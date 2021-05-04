/*! \brief   Implementation of methods of TTStubAlgorithm_official
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Nicola Pozzobon
 *  \author Sebastien Viret
 *  \date   2013, Jul 18
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h"

/// Matching operations
template <>
void TTStubAlgorithm_official<Ref_Phase2TrackerDigi_>::PatternHitCorrelation(
    bool& aConfirmation,
    int& aDisplacement,
    int& anOffset,
    float& anHardBend,
    const TTStub<Ref_Phase2TrackerDigi_>& aTTStub) const {
  /// Calculate average coordinates col/row for inner/outer Cluster
  // These are already corrected for being at the center of each pixel
  MeasurementPoint mp0 = aTTStub.clusterRef(0)->findAverageLocalCoordinates();
  MeasurementPoint mp1 = aTTStub.clusterRef(1)->findAverageLocalCoordinates();

  bool isPS = aTTStub.moduleTypePS();  // get it from the stub now

  /// Get the module position in global coordinates
  // TODO temporary: should use a method from the topology
  DetId stDetId(aTTStub.getDetId());
  const GeomDetUnit* det0 = theTrackerGeom_->idToDetUnit(stDetId + 1);
  const GeomDetUnit* det1 = theTrackerGeom_->idToDetUnit(stDetId + 2);

  /// Find pixel pitch and topology related information
  const PixelGeomDetUnit* pix0 = dynamic_cast<const PixelGeomDetUnit*>(det0);
  const PixelGeomDetUnit* pix1 = dynamic_cast<const PixelGeomDetUnit*>(det1);
  const PixelTopology* top0 = dynamic_cast<const PixelTopology*>(&(pix0->specificTopology()));
  const PixelTopology* top1 = dynamic_cast<const PixelTopology*>(&(pix1->specificTopology()));
  std::pair<float, float> pitch0 = top0->pitch();
  std::pair<float, float> pitch1 = top1->pitch();

  /// Stop if the clusters are not in the same z-segment
  int cols0 = top0->ncolumns();
  int cols1 = top1->ncolumns();
  int ratio = cols0 / cols1;  /// This assumes the ratio is integer!
  int segment0 = floor(mp0.y() / ratio);

  //  if ( ratio == 1 ) /// 2S Modules
  if (!isPS) {
    if (mPerformZMatching2S && (segment0 != floor(mp1.y())))
      return;
  } else  /// PS Modules
  {
    if (mPerformZMatchingPS && (segment0 != floor(mp1.y())))
      return;
  }

  /// Get the Stack radius and z and displacements
  double R0 = det0->position().perp();
  double R1 = det1->position().perp();
  double Z0 = det0->position().z();
  double Z1 = det1->position().z();

  double DR = R1 - R0;
  double DZ = Z1 - Z0;

  double alpha = atan2(DR, DZ);
  double delta = sqrt(DR * DR + DZ * DZ) / (R0 * sin(alpha) + Z0 * cos(alpha));

  int window = 0;

  /// Scale factor is already present in
  /// double mPtScalingFactor = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0*0.0015/mPtThreshold;
  /// hence the formula iis something like
  /// displacement < Delta * 1 / sqrt( ( 1/(mPtScalingFactor*R) )** 2 - 1 )

  /// POSITION IN TERMS OF PITCH MULTIPLES:
  ///       0 1 2 3 4 5 5 6 8 9 ...
  /// COORD: 0 1 2 3 4 5 6 7 8 9 ...
  /// OUT   | | | | | |x| | | | | | | | | |
  ///
  /// IN    | | | |x|x| | | | | | | | | | |
  ///             THIS is 3.5 (COORD) and 4.0 (POS)
  /// 1) disp is the difference between average row coordinates
  ///    in inner and outer stack member, in terms of outer member pitch
  ///    (in case they are the same, this is just a plain coordinate difference)
  double dispD = 2 * (mp1.x() - mp0.x()) * (pitch0.first / pitch1.first);  /// In HALF-STRIP units!
  int dispI = ((dispD > 0) - (dispD < 0)) * floor(std::abs(dispD));        /// In HALF-STRIP units!
  /// 2) offset is the projection with a straight line of the innermost
  ///    hit towards the ourermost stack member, still in terms of outer member pitch
  ///    NOTE: in terms of coordinates, the center of the module is at NROWS/2-0.5 to
  ///    be consistent with the definition given above

  /// In HALF-STRIP units!
  double offsetD = 2 * delta * (mp0.x() - (top0->nrows() / 2 - 0.5)) * (pitch0.first / pitch1.first);
  int offsetI = ((offsetD > 0) - (offsetD < 0)) * floor(std::abs(offsetD));  /// In HALF-STRIP units!

  if (stDetId.subdetId() == StripSubdetector::TOB) {
    int layer = theTrackerTopo_->layer(stDetId);
    int ladder = theTrackerTopo_->tobRod(stDetId);
    int type = 2 * theTrackerTopo_->tobSide(stDetId) - 3;  // -1 for tilted-, 1 for tilted+, 3 for flat
    double corr = 0;

    if (type < 3)  // Only for tilted modules
    {
      corr = (barrelNTilt.at(layer) + 1) / 2.;
      // Corrected ring number, bet 0 and barrelNTilt.at(layer), in ascending |z|
      ladder = corr - (corr - ladder) * type;
      window = 2 * (tiltedCut.at(layer)).at(ladder);
    } else  // Classis barrel window otherwise
    {
      window = 2 * barrelCut.at(layer);
    }

  } else if (stDetId.subdetId() == StripSubdetector::TID) {
    window = 2 * (ringCut.at(theTrackerTopo_->tidWheel(stDetId))).at(theTrackerTopo_->tidRing(stDetId));
  }

  /// Accept the stub if the post-offset correction displacement is smaller than the half-window
  if (std::abs(dispI - offsetI) <= window)  /// In HALF-STRIP units!
  {
    aConfirmation = true;
    aDisplacement = dispI;                                                     /// In HALF-STRIP units!
    anOffset = offsetI;                                                        /// In HALF-STRIP units!
    anHardBend = this->degradeBend(isPS, window, (aDisplacement - anOffset));  // In strips units
  }                                                                            /// End of stub is accepted
}

//--- Does the actual work of degrading the bend. (based on I.Tomalin's code)
template <>
float TTStubAlgorithm_official<Ref_Phase2TrackerDigi_>::degradeBend(bool psModule, int window, int bend) const {
  // Number of bits used to encoded bend output by FE electronics.
  const unsigned int bitsPS_ = 3;
  const unsigned int bits2S_ = 4;

  // Number of degraded bend values should correspond to 3 bits (PS modules) or 4 bits (2S modules),
  // so measuring everything in half-strip units, max integer "window" size that can be encoded without
  // compression given by 2*window+1 <= pow(2,B), where B is number of bits.
  // Hence no compression required if window cut is abs(b) <= 3 (PS) or 7 (2S). Must introduce one merge for
  // each 1 unit increase in "window" beyond this.

  // Bend is measured with granularity of 0.5 strips.
  // Convert it to integer measured in half-strip units for this calculation!

  float degradedB;
  unsigned int numBends = 2 * window + 1;
  unsigned int numAllowed = (psModule) ? pow(2, bitsPS_) : pow(2, bits2S_);

  // Existance of bend = 0 means can only use an odd number of groups.
  numAllowed -= 1;  // NumAllowed can be only based on 3 or 4 bits encoded bends, so 7 or 15 possible values
  if (numBends <= numAllowed) {
    // Can output uncompressed bend info. (So if window is lower or equal than 1.5 in PS, and 3.5 in 2S
    degradedB = static_cast<double>(bend);
  } else  // all other cases, need to compress
  {
    unsigned int inSmallGroup = numBends / numAllowed;
    unsigned int numLargeGroups = numBends % numAllowed;
    unsigned int inLargeGroup = inSmallGroup + 1;
    unsigned int numSmallGroups = numAllowed - numLargeGroups;

    std::vector<unsigned int> groups;

    // At the end we have
    //
    // numBends=inSmallGroup*numSmallGroups+inLargeGroup*numLargeGroups
    // and
    // numAllowed= numSmallGroups+numLargeGroups;
    //
    // Then you alternate large-small-large-small....large. In the middle, you
    // put either large or small, depending if group size is odd or not

    for (unsigned int i = 0; i < numLargeGroups / 2; i++)
      groups.push_back(inLargeGroup);
    for (unsigned int i = 0; i < numSmallGroups / 2; i++)
      groups.push_back(inSmallGroup);

    // Only one of numLargeGroups & numSmallGroups can be odd, since numAllowed is odd.
    // And whichever one is odd is associated to a group with an odd number of elements since numBends is odd,
    if (numLargeGroups % 2 == 1 && inLargeGroup % 2 == 1) {
      groups.push_back(inLargeGroup);
    } else if (numSmallGroups % 2 == 1 && inSmallGroup % 2 == 1) {
      groups.push_back(inSmallGroup);
    } else {
      throw cms::Exception("DegradeBend: logic error with odd numbers");
    }

    for (unsigned int i = 0; i < numSmallGroups / 2; i++)
      groups.push_back(inSmallGroup);
    for (unsigned int i = 0; i < numLargeGroups / 2; i++)
      groups.push_back(inLargeGroup);

    degradedB = 999;
    int iUp = -static_cast<int>(window) - 1;  // Start with the minimal possible bend -1
    int iDown;

    for (unsigned int& inGroup : groups) {
      iUp += inGroup;
      iDown = iUp - inGroup + 1;
      if (bend <= iUp && bend >= iDown) {
        degradedB = 0.5 * (iUp + iDown);
      }
    }
    if (degradedB == 999)
      throw cms::Exception(
          "DegradeStubResolution: error in the group creation, method has been called with wrong inputs");
  }

  // This is degraded bend in full strip units (neglecting bend sign).
  return static_cast<float>(degradedB) / 2.;
}

//--- Check for mistakes.
