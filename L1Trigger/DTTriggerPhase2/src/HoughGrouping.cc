#include "L1Trigger/DTTriggerPhase2/interface/HoughGrouping.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/CMSUnits.h"

#include <cmath>
#include <memory>

#include "TMath.h"

using namespace std;
using namespace edm;
using namespace cmsdt;
using namespace cms_units::operators;

namespace {
  struct {
    bool operator()(ProtoCand a, ProtoCand b) const {
      unsigned short int sumhqa = 0;
      unsigned short int sumhqb = 0;
      unsigned short int sumlqa = 0;
      unsigned short int sumlqb = 0;
      double sumdista = 0;
      double sumdistb = 0;

      for (unsigned short int lay = 0; lay < 8; lay++) {
        sumhqa += (unsigned short int)a.isThereHitInLayer_[lay];
        sumhqb += (unsigned short int)b.isThereHitInLayer_[lay];
        sumlqa += (unsigned short int)a.isThereNeighBourHitInLayer_[lay];
        sumlqb += (unsigned short int)b.isThereNeighBourHitInLayer_[lay];
        sumdista += a.xDistToPattern_[lay];
        sumdistb += b.xDistToPattern_[lay];
      }

      if (a.nLayersWithHits_ != b.nLayersWithHits_)
        return (a.nLayersWithHits_ > b.nLayersWithHits_);  // number of layers with hits
      else if (sumhqa != sumhqb)
        return (sumhqa > sumhqb);  // number of hq hits
      else if (sumlqa != sumlqb)
        return (sumlqa > sumlqb);  // number of lq hits
      else if (a.nHitsDiff_ != b.nHitsDiff_)
        return (a.nHitsDiff_ < b.nHitsDiff_);  // abs. diff. between SL1 & SL3 hits
      else
        return (sumdista < sumdistb);  // abs. dist. to digis
    }
  } const HoughOrdering;
}  // namespace
// ============================================================================
// Constructors and destructor
// ============================================================================
HoughGrouping::HoughGrouping(const ParameterSet& pset, edm::ConsumesCollector& iC) : MotherGrouping(pset, iC) {
  // Obtention of parameters
  debug_ = pset.getUntrackedParameter<bool>("debug");
  if (debug_)
    LogDebug("HoughGrouping") << "HoughGrouping: constructor";

  // HOUGH TRANSFORM CONFIGURATION
  angletan_ = pset.getUntrackedParameter<double>("angletan");
  anglebinwidth_ = pset.getUntrackedParameter<double>("anglebinwidth");
  posbinwidth_ = pset.getUntrackedParameter<double>("posbinwidth");

  // MAXIMA SEARCH CONFIGURATION
  maxdeltaAngDeg_ = pset.getUntrackedParameter<double>("maxdeltaAngDeg");
  maxdeltaPos_ = pset.getUntrackedParameter<double>("maxdeltaPos");
  upperNumber_ = (unsigned short int)pset.getUntrackedParameter<int>("UpperNumber");
  lowerNumber_ = (unsigned short int)pset.getUntrackedParameter<int>("LowerNumber");

  // HITS ASSOCIATION CONFIGURATION
  maxDistanceToWire_ = pset.getUntrackedParameter<double>("MaxDistanceToWire");

  // CANDIDATE QUALITY CONFIGURATION
  minNLayerHits_ = (unsigned short int)pset.getUntrackedParameter<int>("minNLayerHits");
  minSingleSLHitsMax_ = (unsigned short int)pset.getUntrackedParameter<int>("minSingleSLHitsMax");
  minSingleSLHitsMin_ = (unsigned short int)pset.getUntrackedParameter<int>("minSingleSLHitsMin");
  allowUncorrelatedPatterns_ = pset.getUntrackedParameter<bool>("allowUncorrelatedPatterns");
  minUncorrelatedHits_ = (unsigned short int)pset.getUntrackedParameter<int>("minUncorrelatedHits");

  dtGeomH = iC.esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
}

HoughGrouping::~HoughGrouping() {
  if (debug_)
    LogDebug("HoughGrouping") << "HoughGrouping: destructor" << endl;
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void HoughGrouping::initialise(const edm::EventSetup& iEventSetup) {
  if (debug_)
    LogDebug("HoughGrouping") << "initialise";

  resetAttributes();

  maxrads_ = 0.5 * M_PI - atan(angletan_);
  minangle_ = (double)convertDegToRad(anglebinwidth_);
  halfanglebins_ = round(maxrads_ / minangle_ + 1);
  anglebins_ = (unsigned short int)2 * halfanglebins_;
  oneanglebin_ = maxrads_ / halfanglebins_;

  maxdeltaAng_ = maxdeltaAngDeg_ * 2 * M_PI / 360;

  // Initialisation of anglemap. Posmap depends on the size of the chamber.
  double phi = 0;
  anglemap_ = {};
  for (unsigned short int ab = 0; ab < halfanglebins_; ab++) {
    anglemap_[ab] = phi;
    phi += oneanglebin_;
  }

  phi = (M_PI - maxrads_);
  for (unsigned short int ab = halfanglebins_; ab < anglebins_; ab++) {
    anglemap_[ab] = phi;
    phi += oneanglebin_;
  }

  if (debug_) {
    LogDebug("HoughGrouping")
        << "\nHoughGrouping::ResetAttributes - Information from the initialisation of HoughGrouping:";
    LogDebug("HoughGrouping") << "ResetAttributes - maxrads: " << maxrads_;
    LogDebug("HoughGrouping") << "ResetAttributes - anglebinwidth: " << anglebinwidth_;
    LogDebug("HoughGrouping") << "ResetAttributes - minangle: " << minangle_;
    LogDebug("HoughGrouping") << "ResetAttributes - halfanglebins: " << halfanglebins_;
    LogDebug("HoughGrouping") << "ResetAttributes - anglebins: " << anglebins_;
    LogDebug("HoughGrouping") << "ResetAttributes - oneanglebin: " << oneanglebin_;
    LogDebug("HoughGrouping") << "ResetAttributes - posbinwidth: " << posbinwidth_;
  }

  const MuonGeometryRecord& geom = iEventSetup.get<MuonGeometryRecord>();
  dtGeo_ = &geom.get(dtGeomH);
}

void HoughGrouping::run(edm::Event& iEvent,
                        const edm::EventSetup& iEventSetup,
                        const DTDigiCollection& digis,
                        MuonPathPtrs& outMpath) {
  if (debug_)
    LogDebug("HoughGrouping") << "\nHoughGrouping::run";

  resetAttributes();

  if (debug_)
    LogDebug("HoughGrouping") << "run - Beginning digis' loop...";
  LocalPoint wirePosInLay, wirePosInChamber;
  GlobalPoint wirePosGlob;
  for (DTDigiCollection::DigiRangeIterator dtLayerIdIt = digis.begin(); dtLayerIdIt != digis.end(); dtLayerIdIt++) {
    const DTLayer* lay = dtGeo_->layer((*dtLayerIdIt).first);
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerIdIt).second).first;
         digiIt != ((*dtLayerIdIt).second).second;
         digiIt++) {
      if (debug_) {
        LogDebug("HoughGrouping") << "\nHoughGrouping::run - Digi number " << idigi_;
        LogDebug("HoughGrouping") << "run - Wheel: " << (*dtLayerIdIt).first.wheel();
        LogDebug("HoughGrouping") << "run - Chamber: " << (*dtLayerIdIt).first.station();
        LogDebug("HoughGrouping") << "run - Sector: " << (*dtLayerIdIt).first.sector();
        LogDebug("HoughGrouping") << "run - Superlayer: " << (*dtLayerIdIt).first.superLayer();
        LogDebug("HoughGrouping") << "run - Layer: " << (*dtLayerIdIt).first.layer();
        LogDebug("HoughGrouping") << "run - Wire: " << (*digiIt).wire();
        LogDebug("HoughGrouping") << "run - First wire: " << lay->specificTopology().firstChannel();
        LogDebug("HoughGrouping") << "run - Last wire: " << lay->specificTopology().lastChannel();
        LogDebug("HoughGrouping") << "run - First wire x: "
                                  << lay->specificTopology().wirePosition(lay->specificTopology().firstChannel());
        LogDebug("HoughGrouping") << "run - Last wire x: "
                                  << lay->specificTopology().wirePosition(lay->specificTopology().lastChannel());
        LogDebug("HoughGrouping") << "run - Cell width: " << lay->specificTopology().cellWidth();
        LogDebug("HoughGrouping") << "run - Cell height: " << lay->specificTopology().cellHeight();
      }
      if ((*dtLayerIdIt).first.superLayer() == 2)
        continue;

      wirePosInLay = LocalPoint(lay->specificTopology().wirePosition((*digiIt).wire()), 0, 0);
      wirePosGlob = lay->toGlobal(wirePosInLay);
      wirePosInChamber = lay->chamber()->toLocal(wirePosGlob);

      if ((*dtLayerIdIt).first.superLayer() == 3) {
        digimap_[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()] = DTPrimitive();
        digimap_[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setTDCTimeStamp((*digiIt).time());
        digimap_[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setChannelId((*digiIt).wire());
        digimap_[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setLayerId((*dtLayerIdIt).first.layer());
        digimap_[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setSuperLayerId((*dtLayerIdIt).first.superLayer());
        digimap_[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setCameraId((*dtLayerIdIt).first.rawId());
      } else {
        digimap_[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()] = DTPrimitive();
        digimap_[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setTDCTimeStamp((*digiIt).time());
        digimap_[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setChannelId((*digiIt).wire());
        digimap_[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setLayerId((*dtLayerIdIt).first.layer());
        digimap_[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setSuperLayerId((*dtLayerIdIt).first.superLayer());
        digimap_[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setCameraId((*dtLayerIdIt).first.rawId());
      }

      // Obtaining geometrical info of the chosen chamber
      if (xlowlim_ == 0 && xhighlim_ == 0 && zlowlim_ == 0 && zhighlim_ == 0) {
        thewheel_ = (*dtLayerIdIt).first.wheel();
        thestation_ = (*dtLayerIdIt).first.station();
        thesector_ = (*dtLayerIdIt).first.sector();
        obtainGeometricalBorders(lay);
      }

      if (debug_) {
        LogDebug("HoughGrouping") << "run - X position of the cell (chamber frame of reference): "
                                  << wirePosInChamber.x();
        LogDebug("HoughGrouping") << "run - Y position of the cell (chamber frame of reference): "
                                  << wirePosInChamber.y();
        LogDebug("HoughGrouping") << "run - Z position of the cell (chamber frame of reference): "
                                  << wirePosInChamber.z();
      }

      hitvec_.push_back({wirePosInChamber.x() - 1.05, wirePosInChamber.z()});
      hitvec_.push_back({wirePosInChamber.x() + 1.05, wirePosInChamber.z()});
      nhits_ += 2;

      idigi_++;
    }
  }

  if (debug_) {
    LogDebug("HoughGrouping") << "\nHoughGrouping::run - nhits: " << nhits_;
    LogDebug("HoughGrouping") << "run - idigi: " << idigi_;
  }

  if (hitvec_.empty()) {
    if (debug_)
      LogDebug("HoughGrouping") << "run - No digis present in this chamber.";
    return;
  }

  // Perform the Hough transform of the inputs.
  doHoughTransform();

  // Obtain the maxima
  maxima_ = getMaximaVector();
  resetPosElementsOfLinespace();

  if (maxima_.empty()) {
    if (debug_)
      LogDebug("HoughGrouping") << "run - No good maxima found in this event.";
    return;
  }

  DTChamberId TheChambId(thewheel_, thestation_, thesector_);
  const DTChamber* TheChamb = dtGeo_->chamber(TheChambId);
  std::vector<ProtoCand> cands;

  for (unsigned short int ican = 0; ican < maxima_.size(); ican++) {
    if (debug_)
      LogDebug("HoughGrouping") << "\nHoughGrouping::run - Candidate number: " << ican;
    cands.push_back(associateHits(TheChamb, maxima_.at(ican).first, maxima_.at(ican).second));
  }

  // Now we filter them:
  orderAndFilter(cands, outMpath);
  if (debug_)
    LogDebug("HoughGrouping") << "run - now we have our muonpaths! It has " << outMpath.size() << " elements";

  cands.clear();
  return;
}

void HoughGrouping::finish() {
  if (debug_)
    LogDebug("HoughGrouping") << "finish";
  return;
}

// ============================================================================
// Other methods
// ============================================================================
void HoughGrouping::resetAttributes() {
  if (debug_)
    LogDebug("HoughGrouping") << "ResetAttributes";
  // std::vector's:
  maxima_.clear();
  hitvec_.clear();

  // Integer-type variables:
  spacebins_ = 0;
  idigi_ = 0;
  nhits_ = 0;
  xlowlim_ = 0;
  xhighlim_ = 0;
  zlowlim_ = 0;
  zhighlim_ = 0;
  thestation_ = 0;
  thesector_ = 0;
  thewheel_ = 0;

  // Arrays:
  // NOTE: linespace array is treated and reset separately

  // Maps (dictionaries):
  posmap_.clear();
  for (unsigned short int abslay = 0; abslay < 8; abslay++)
    digimap_[abslay].clear();
}

void HoughGrouping::resetPosElementsOfLinespace() {
  if (debug_)
    LogDebug("HoughGrouping") << "ResetPosElementsOfLinespace";
  for (unsigned short int ab = 0; ab < anglebins_; ab++) {
    linespace_[ab].clear();
  }
  linespace_.clear();
}

void HoughGrouping::obtainGeometricalBorders(const DTLayer* lay) {
  if (debug_)
    LogDebug("HoughGrouping") << "ObtainGeometricalBorders";
  LocalPoint FirstWireLocal(lay->chamber()->superLayer(1)->layer(1)->specificTopology().wirePosition(
                                lay->chamber()->superLayer(1)->layer(1)->specificTopology().firstChannel()),
                            0,
                            0);  // TAKING INFO FROM L1 OF SL1 OF THE CHOSEN CHAMBER
  GlobalPoint FirstWireGlobal = lay->chamber()->superLayer(1)->layer(1)->toGlobal(FirstWireLocal);
  LocalPoint FirstWireLocalCh = lay->chamber()->toLocal(FirstWireGlobal);

  LocalPoint LastWireLocal(lay->chamber()->superLayer(1)->layer(1)->specificTopology().wirePosition(
                               lay->chamber()->superLayer(1)->layer(1)->specificTopology().lastChannel()),
                           0,
                           0);
  GlobalPoint LastWireGlobal = lay->chamber()->superLayer(1)->layer(1)->toGlobal(LastWireLocal);
  LocalPoint LastWireLocalCh = lay->chamber()->toLocal(LastWireGlobal);

  //   unsigned short int upsl = thestation == 4 ? 2 : 3;
  unsigned short int upsl = thestation_ == 4 ? 3 : 3;
  if (debug_)
    LogDebug("HoughGrouping") << "ObtainGeometricalBorders - uppersuperlayer: " << upsl;

  LocalPoint FirstWireLocalUp(lay->chamber()->superLayer(upsl)->layer(4)->specificTopology().wirePosition(
                                  lay->chamber()->superLayer(upsl)->layer(4)->specificTopology().firstChannel()),
                              0,
                              0);  // TAKING INFO FROM L1 OF SL1 OF THE CHOSEN CHAMBER
  GlobalPoint FirstWireGlobalUp = lay->chamber()->superLayer(upsl)->layer(4)->toGlobal(FirstWireLocalUp);
  LocalPoint FirstWireLocalChUp = lay->chamber()->toLocal(FirstWireGlobalUp);

  xlowlim_ = FirstWireLocalCh.x() - lay->chamber()->superLayer(1)->layer(1)->specificTopology().cellWidth() / 2;
  xhighlim_ = LastWireLocalCh.x() + lay->chamber()->superLayer(1)->layer(1)->specificTopology().cellWidth() / 2;
  zlowlim_ = FirstWireLocalChUp.z() - lay->chamber()->superLayer(upsl)->layer(4)->specificTopology().cellHeight() / 2;
  zhighlim_ = LastWireLocalCh.z() + lay->chamber()->superLayer(1)->layer(1)->specificTopology().cellHeight() / 2;

  spacebins_ = round(std::abs(xhighlim_ - xlowlim_) / posbinwidth_);
}

void HoughGrouping::doHoughTransform() {
  if (debug_)
    LogDebug("HoughGrouping") << "DoHoughTransform";
  // First we want to obtain the number of bins in angle that we want. To do so, we will consider at first a maximum angle of
  // (in rad.) pi/2 - arctan(0.3) (i.e. ~73º) and a resolution (width of bin angle) of 2º.

  if (debug_) {
    LogDebug("HoughGrouping") << "DoHoughTransform - maxrads: " << maxrads_;
    LogDebug("HoughGrouping") << "DoHoughTransform - minangle: " << minangle_;
    LogDebug("HoughGrouping") << "DoHoughTransform - halfanglebins: " << halfanglebins_;
    LogDebug("HoughGrouping") << "DoHoughTransform - anglebins: " << anglebins_;
    LogDebug("HoughGrouping") << "DoHoughTransform - oneanglebin: " << oneanglebin_;
    LogDebug("HoughGrouping") << "DoHoughTransform - spacebins: " << spacebins_;
  }

  double rho = 0, phi = 0, sbx = 0;
  double lowinitsb = xlowlim_ + posbinwidth_ / 2;

  // Initialisation
  linespace_.resize(anglebins_, std::vector<unsigned short int>(spacebins_));
  for (unsigned short int ab = 0; ab < anglebins_; ab++) {
    sbx = lowinitsb;
    for (unsigned short int sb = 0; sb < spacebins_; sb++) {
      posmap_[sb] = sbx;
      linespace_[ab][sb] = 0;
      sbx += posbinwidth_;
    }
  }

  // Filling of the double array and actually doing the transform
  for (unsigned short int i = 0; i < hitvec_.size(); i++) {
    for (unsigned short int ab = 0; ab < anglebins_; ab++) {
      phi = anglemap_[ab];
      rho = hitvec_.at(i).first * cos(phi) + hitvec_.at(i).second * sin(phi);
      sbx = lowinitsb - posbinwidth_ / 2;
      for (unsigned short int sb = 0; sb < spacebins_; sb++) {
        if (rho < sbx) {
          linespace_[ab][sb]++;
          break;
        }
        sbx += posbinwidth_;
      }
    }
  }
}

PointsInPlane HoughGrouping::getMaximaVector() {
  if (debug_)
    LogDebug("HoughGrouping") << "getMaximaVector";
  PointTuples tmpvec;

  bool flagsearched = false;
  unsigned short int numbertolookat = upperNumber_;

  while (!flagsearched) {
    for (unsigned short int ab = 0; ab < anglebins_; ab++) {
      for (unsigned short int sb = 0; sb < spacebins_; sb++) {
        if (linespace_[ab][sb] >= numbertolookat)
          tmpvec.push_back({anglemap_[ab], posmap_[sb], linespace_[ab][sb]});
      }
    }
    if (((numbertolookat - 1) < lowerNumber_) || (!tmpvec.empty()))
      flagsearched = true;
    else
      numbertolookat--;
  }

  if (tmpvec.empty()) {
    if (debug_)
      LogDebug("HoughGrouping") << "GetMaximaVector - No maxima could be found";
    PointsInPlane finalvec;
    return finalvec;
  } else {
    PointsInPlane finalvec = findTheMaxima(tmpvec);

    // And now obtain the values of m and n of the lines.
    for (unsigned short int i = 0; i < finalvec.size(); i++)
      finalvec.at(i) = transformPair(finalvec.at(i));
    return finalvec;
  }
}

PointInPlane HoughGrouping::transformPair(const PointInPlane& inputpair) {
  if (debug_)
    LogDebug("HoughGrouping") << "transformPair";
  // input: (ang, pos); output: (m, n)
  if (inputpair.first == 0)
    return {1000, -1000 * inputpair.second};
  else
    return {-1. / tan(inputpair.first), inputpair.second / sin(inputpair.first)};
}

PointsInPlane HoughGrouping::findTheMaxima(PointTuples& inputvec) {
  if (debug_)
    LogDebug("HoughGrouping") << "findTheMaxima";
  bool fullyreduced = false;
  unsigned short int ind = 0;

  std::vector<unsigned short int> chosenvec;
  PointsInPlane resultvec;
  PointInPlane finalpair = {};

  if (debug_)
    LogDebug("HoughGrouping") << "findTheMaxima - prewhile";
  while (!fullyreduced) {
    if (debug_) {
      LogDebug("HoughGrouping") << "\nHoughGrouping::findTheMaxima - New iteration";
      LogDebug("HoughGrouping") << "findTheMaxima - inputvec size: " << inputvec.size();
      LogDebug("HoughGrouping") << "findTheMaxima - ind: " << ind;
      LogDebug("HoughGrouping") << "findTheMaxima - maximum deltaang: " << maxdeltaAng_
                                << " and maximum deltapos: " << maxdeltaPos_;
    }
    chosenvec.clear();
    //calculate distances and check out the ones that are near
    if (debug_)
      LogDebug("HoughGrouping") << "findTheMaxima - Ours have " << get<2>(inputvec.at(ind))
                                << " entries, ang.: " << get<0>(inputvec.at(ind))
                                << " and pos.: " << get<1>(inputvec.at(ind));

    for (unsigned short int j = ind + 1; j < inputvec.size(); j++) {
      if (getTwoDelta(inputvec.at(ind), inputvec.at(j)).first <= maxdeltaAng_ &&
          getTwoDelta(inputvec.at(ind), inputvec.at(j)).second <= maxdeltaPos_) {
        chosenvec.push_back(j);
        if (debug_)
          LogDebug("HoughGrouping") << "findTheMaxima -     - Adding num.  " << j
                                    << " with deltaang: " << getTwoDelta(inputvec.at(ind), inputvec.at(j)).first
                                    << ", and deltapos: " << getTwoDelta(inputvec.at(ind), inputvec.at(j)).second
                                    << " and with " << get<2>(inputvec.at(j))
                                    << " entries, ang.: " << get<0>(inputvec.at(j))
                                    << " and pos.: " << get<1>(inputvec.at(j));
      } else if (debug_)
        LogDebug("HoughGrouping") << "findTheMaxima -     - Ignoring num. " << j
                                  << " with deltaang: " << getTwoDelta(inputvec.at(ind), inputvec.at(j)).first
                                  << ", and deltapos: " << getTwoDelta(inputvec.at(ind), inputvec.at(j)).second
                                  << " and with " << get<2>(inputvec.at(j)) << " entries.";
    }

    if (debug_)
      LogDebug("HoughGrouping") << "findTheMaxima - chosenvecsize: " << chosenvec.size();

    if (chosenvec.empty()) {
      if (ind + 1 >= (unsigned short int)inputvec.size())
        fullyreduced = true;
      if ((get<0>(inputvec.at(ind)) <= maxrads_) || (get<0>(inputvec.at(ind)) >= M_PI - maxrads_))
        resultvec.push_back({get<0>(inputvec.at(ind)), get<1>(inputvec.at(ind))});
      else if (debug_)
        LogDebug("HoughGrouping") << "findTheMaxima -     - Candidate dropped due to an excess in angle";
      ind++;
      continue;
    }

    // Now average them
    finalpair = getAveragePoint(inputvec, ind, chosenvec);

    // Erase the ones you used
    inputvec.erase(inputvec.begin() + ind);
    for (short int j = chosenvec.size() - 1; j > -1; j--) {
      if (debug_)
        LogDebug("HoughGrouping") << "findTheMaxima - erasing index: " << chosenvec.at(j) - 1;
      inputvec.erase(inputvec.begin() + chosenvec.at(j) - 1);
    }

    if (debug_)
      LogDebug("HoughGrouping") << "findTheMaxima - inputvec size: " << inputvec.size();

    // And add the one you calculated:
    if ((finalpair.first <= maxrads_) || (finalpair.first >= M_PI - maxrads_))
      resultvec.push_back(finalpair);
    else if (debug_)
      LogDebug("HoughGrouping") << "findTheMaxima -     - Candidate dropped due to an excess in angle";

    if (ind + 1 >= (unsigned short int)inputvec.size())
      fullyreduced = true;
    if (debug_)
      LogDebug("HoughGrouping") << "findTheMaxima - iteration ends";
    ind++;
  }
  if (debug_)
    LogDebug("HoughGrouping") << "findTheMaxima - postwhile";
  return resultvec;
}

PointInPlane HoughGrouping::getTwoDelta(const PointTuple& pair1, const PointTuple& pair2) {
  if (debug_)
    LogDebug("HoughGrouping") << "getTwoDelta";
  return {abs(get<0>(pair1) - get<0>(pair2)), abs(get<1>(pair1) - get<1>(pair2))};
}

PointInPlane HoughGrouping::getAveragePoint(const PointTuples& inputvec,
                                            unsigned short int firstindex,
                                            const std::vector<unsigned short int>& indexlist) {
  if (debug_)
    LogDebug("HoughGrouping") << "getAveragePoint";
  std::vector<double> xs;
  std::vector<double> ys;
  std::vector<double> ws;
  xs.push_back(get<0>(inputvec.at(firstindex)));
  ys.push_back(get<1>(inputvec.at(firstindex)));
  ws.push_back(exp(get<2>(inputvec.at(firstindex))));
  for (unsigned short int i = 0; i < indexlist.size(); i++) {
    xs.push_back(get<0>(inputvec.at(indexlist.at(i))));
    ys.push_back(get<1>(inputvec.at(indexlist.at(i))));
    ws.push_back(exp(get<2>(inputvec.at(indexlist.at(i)))));
  }
  return {TMath::Mean(xs.begin(), xs.end(), ws.begin()), TMath::Mean(ys.begin(), ys.end(), ws.begin())};
}

ProtoCand HoughGrouping::associateHits(const DTChamber* thechamb, double m, double n) {
  if (debug_)
    LogDebug("HoughGrouping") << "associateHits";
  LocalPoint tmpLocal, AWireLocal, AWireLocalCh, tmpLocalCh, thepoint;
  GlobalPoint tmpGlobal, AWireGlobal;
  double tmpx = 0;
  double distleft = 0;
  double distright = 0;
  unsigned short int tmpwire = 0;
  unsigned short int abslay = 0;
  LATERAL_CASES lat = NONE;
  bool isleft = false;
  bool isright = false;

  ProtoCand returnPC;
  for (auto l = 0; l < NUM_LAYERS_2SL; l++) {
    returnPC.isThereHitInLayer_.push_back(false);
    returnPC.isThereNeighBourHitInLayer_.push_back(false);
    returnPC.xDistToPattern_.push_back(0);
    returnPC.dtHits_.push_back(DTPrimitive());
  }

  if (debug_)
    LogDebug("HoughGrouping") << "associateHits - Beginning SL loop";
  for (unsigned short int sl = 1; sl < 3 + 1; sl++) {
    if (sl == 2)
      continue;
    if (debug_)
      LogDebug("HoughGrouping") << "associateHits - SL: " << sl;

    for (unsigned short int l = 1; l < 4 + 1; l++) {
      if (debug_)
        LogDebug("HoughGrouping") << "associateHits - L: " << l;
      isleft = false;
      isright = false;
      lat = NONE;
      distleft = 0;
      distright = 0;
      if (sl == 1)
        abslay = l - 1;
      else
        abslay = l + 3;
      AWireLocal = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(
                                  thechamb->superLayer(sl)->layer(l)->specificTopology().firstChannel()),
                              0,
                              0);
      AWireGlobal = thechamb->superLayer(sl)->layer(l)->toGlobal(AWireLocal);
      AWireLocalCh = thechamb->toLocal(AWireGlobal);
      tmpx = (AWireLocalCh.z() - n) / m;

      if ((tmpx <= xlowlim_) || (tmpx >= xhighlim_)) {
        returnPC.dtHits_[abslay] = DTPrimitive();  // empty primitive
        continue;
      }

      thepoint = LocalPoint(tmpx, 0, AWireLocalCh.z());
      tmpwire = thechamb->superLayer(sl)->layer(l)->specificTopology().channel(thepoint);
      if (debug_)
        LogDebug("HoughGrouping") << "associateHits - Wire number: " << tmpwire;
      if (debug_)
        LogDebug("HoughGrouping") << "associateHits - First channel in layer: "
                                  << thechamb->superLayer(sl)->layer(l)->specificTopology().firstChannel();
      if ((digimap_[abslay]).count(tmpwire)) {
        // OK, we have a digi, let's choose the laterality, if we can:
        tmpLocal = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire), 0, 0);
        tmpGlobal = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal);
        tmpLocalCh = thechamb->toLocal(tmpGlobal);

        if (abs(tmpLocalCh.x() - thepoint.x()) >= maxDistanceToWire_) {
          // The distance where lateralities are not put is 0.03 cm, which is a conservative threshold for the resolution of the cells.
          if ((tmpLocalCh.x() - thepoint.x()) > 0)
            lat = LEFT;
          else
            lat = RIGHT;
        }

        // Filling info
        returnPC.nLayersWithHits_++;
        returnPC.isThereHitInLayer_[abslay] = true;
        returnPC.isThereNeighBourHitInLayer_[abslay] = true;
        if (lat == LEFT)
          returnPC.xDistToPattern_[abslay] = abs(tmpx - (tmpLocalCh.x() - 1.05));
        else if (lat == RIGHT)
          returnPC.xDistToPattern_[abslay] = abs(tmpx - (tmpLocalCh.x() + 1.05));
        else
          returnPC.xDistToPattern_[abslay] = abs(tmpx - tmpLocalCh.x());
        returnPC.dtHits_[abslay] = DTPrimitive(digimap_[abslay][tmpwire]);
        returnPC.dtHits_[abslay].setLaterality(lat);
      } else {
        if (debug_)
          LogDebug("HoughGrouping") << "associateHits - No hit in the crossing cell";
        if ((digimap_[abslay]).count(tmpwire - 1))
          isleft = true;
        if ((digimap_[abslay]).count(tmpwire + 1))
          isright = true;
        if (debug_)
          LogDebug("HoughGrouping") << "associateHits - There is in the left: " << (int)isleft;
        if (debug_)
          LogDebug("HoughGrouping") << "associateHits - There is in the right: " << (int)isright;

        if ((isleft) && (!isright)) {
          tmpLocal = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire - 1), 0, 0);
          tmpGlobal = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal);
          tmpLocalCh = thechamb->toLocal(tmpGlobal);

          // Filling info
          returnPC.nLayersWithHits_++;
          returnPC.isThereNeighBourHitInLayer_[abslay] = true;
          returnPC.xDistToPattern_[abslay] = abs(tmpx - (tmpLocalCh.x() + 1.05));
          returnPC.dtHits_[abslay] = DTPrimitive(digimap_[abslay][tmpwire - 1]);
          returnPC.dtHits_[abslay].setLaterality(RIGHT);

        } else if ((!isleft) && (isright)) {
          tmpLocal = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire + 1), 0, 0);
          tmpGlobal = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal);
          tmpLocalCh = thechamb->toLocal(tmpGlobal);

          // Filling info
          returnPC.nLayersWithHits_++;
          returnPC.isThereNeighBourHitInLayer_[abslay] = true;
          returnPC.xDistToPattern_[abslay] = abs(tmpx - (tmpLocalCh.x() - 1.05));
          returnPC.dtHits_[abslay] = DTPrimitive(digimap_[abslay][tmpwire + 1]);
          returnPC.dtHits_[abslay].setLaterality(LEFT);
        } else if ((isleft) && (isright)) {
          LocalPoint tmpLocal_l =
              LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire - 1), 0, 0);
          GlobalPoint tmpGlobal_l = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal_l);
          LocalPoint tmpLocalCh_l = thechamb->toLocal(tmpGlobal_l);

          LocalPoint tmpLocal_r =
              LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire + 1), 0, 0);
          GlobalPoint tmpGlobal_r = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal_r);
          LocalPoint tmpLocalCh_r = thechamb->toLocal(tmpGlobal_r);

          distleft = std::abs(thepoint.x() - tmpLocalCh_l.x());
          distright = std::abs(thepoint.x() - tmpLocalCh_r.x());

          // Filling info
          returnPC.nLayersWithHits_++;
          returnPC.isThereNeighBourHitInLayer_[abslay] = true;

          returnPC.xDistToPattern_[abslay] = abs(tmpx - (tmpLocalCh.x() - 1.05));
          returnPC.dtHits_[abslay] = DTPrimitive(digimap_[abslay][tmpwire + 1]);
          returnPC.dtHits_[abslay].setLaterality(LEFT);

          if (distleft < distright) {
            returnPC.xDistToPattern_[abslay] = std::abs(tmpx - (tmpLocalCh.x() + 1.05));
            returnPC.dtHits_[abslay] = DTPrimitive(digimap_[abslay][tmpwire - 1]);
            returnPC.dtHits_[abslay].setLaterality(RIGHT);
          } else {
            returnPC.xDistToPattern_[abslay] = std::abs(tmpx - (tmpLocalCh.x() - 1.05));
            returnPC.dtHits_[abslay] = DTPrimitive(digimap_[abslay][tmpwire + 1]);
            returnPC.dtHits_[abslay].setLaterality(LEFT);
          }
        } else {                                     // case where there are no digis
          returnPC.dtHits_[abslay] = DTPrimitive();  // empty primitive
        }
      }
    }
  }

  setDifferenceBetweenSL(returnPC);
  if (debug_) {
    LogDebug("HoughGrouping") << "associateHits - Finishing with the candidate. We have found the following of it:";
    LogDebug("HoughGrouping") << "associateHits - # of layers with hits: " << returnPC.nLayersWithHits_;
    for (unsigned short int lay = 0; lay < 8; lay++) {
      LogDebug("HoughGrouping") << "associateHits - For absolute layer: " << lay;
      LogDebug("HoughGrouping") << "associateHits - # of HQ hits: " << returnPC.isThereHitInLayer_[lay];
      LogDebug("HoughGrouping") << "associateHits - # of LQ hits: " << returnPC.isThereNeighBourHitInLayer_[lay];
    }
    LogDebug("HoughGrouping") << "associateHits - Abs. diff. between SL1 and SL3 hits: " << returnPC.nHitsDiff_;
    for (unsigned short int lay = 0; lay < 8; lay++) {
      LogDebug("HoughGrouping") << "associateHits - For absolute layer: " << lay;
      LogDebug("HoughGrouping") << "associateHits - Abs. distance to digi: " << returnPC.xDistToPattern_[lay];
    }
  }
  return returnPC;
}

void HoughGrouping::setDifferenceBetweenSL(ProtoCand& tupl) {
  if (debug_)
    LogDebug("HoughGrouping") << "setDifferenceBetweenSL";
  short int absres = 0;
  for (unsigned short int lay = 0; lay < 8; lay++) {
    if (tupl.dtHits_[lay].channelId() > 0) {
      if (lay <= 3)
        absres++;
      else
        absres--;
    }
  }

  if (absres >= 0)
    tupl.nHitsDiff_ = absres;
  else
    tupl.nHitsDiff_ = (unsigned short int)(-absres);
}

void HoughGrouping::orderAndFilter(std::vector<ProtoCand>& invector, MuonPathPtrs& outMuonPath) {
  if (debug_)
    LogDebug("HoughGrouping") << "orderAndFilter";
  // 0: # of layers with hits.
  // 1: # of hits of high quality (the expected line crosses the cell).
  // 2: # of hits of low quality (the expected line is in a neighbouring cell).
  // 3: absolute diff. between the number of hits in SL1 and SL3.
  // 4: absolute distance to all hits of the segment.
  // 5: DTPrimitive of the candidate.

  std::vector<unsigned short int> elstoremove;
  // ordering:
  if (debug_)
    LogDebug("HoughGrouping") << "orderAndFilter - First ordering";
  std::sort(invector.begin(), invector.end(), HoughOrdering);

  // Now filtering:
  unsigned short int ind = 0;
  bool filtered = false;
  if (debug_)
    LogDebug("HoughGrouping") << "orderAndFilter - Entering while";
  while (!filtered) {
    if (debug_)
      LogDebug("HoughGrouping") << "\nHoughGrouping::orderAndFilter - New iteration with ind: " << ind;
    elstoremove.clear();
    for (unsigned short int i = ind + 1; i < invector.size(); i++) {
      if (debug_)
        LogDebug("HoughGrouping") << "orderAndFilter - Checking index: " << i;
      for (unsigned short int lay = 0; lay < 8; lay++) {
        if (debug_)
          LogDebug("HoughGrouping") << "orderAndFilter - Checking layer number: " << lay;
        if (invector.at(i).dtHits_[lay].channelId() == invector.at(ind).dtHits_[lay].channelId() &&
            invector.at(ind).dtHits_[lay].channelId() != -1) {
          invector.at(i).nLayersWithHits_--;
          invector.at(i).isThereHitInLayer_[lay] = false;
          invector.at(i).isThereNeighBourHitInLayer_[lay] = false;
          setDifferenceBetweenSL(invector.at(i));
          // We check that if its a different laterality, the best candidate of the two of them changes its laterality to not-known (that is, both).
          if (invector.at(i).dtHits_[lay].laterality() != invector.at(ind).dtHits_[lay].laterality())
            invector.at(ind).dtHits_[lay].setLaterality(NONE);

          invector.at(i).dtHits_[lay] = DTPrimitive();
        }
      }
      if (debug_)
        LogDebug("HoughGrouping")
            << "orderAndFilter - Finished checking all the layers, now seeing if we should remove the "
               "candidate";

      if (!areThereEnoughHits(invector.at(i))) {
        if (debug_)
          LogDebug("HoughGrouping") << "orderAndFilter - This candidate shall be removed!";
        elstoremove.push_back((unsigned short int)i);
      }
    }

    if (debug_)
      LogDebug("HoughGrouping") << "orderAndFilter - We are gonna erase " << elstoremove.size() << " elements";

    for (short int el = (elstoremove.size() - 1); el > -1; el--) {
      invector.erase(invector.begin() + elstoremove.at(el));
    }

    if (ind + 1 == (unsigned short int)invector.size())
      filtered = true;
    else
      std::sort(invector.begin() + ind + 1, invector.end(), HoughOrdering);
    ind++;
  }

  // Ultimate filter: if the remaining do not fill the requirements (configurable through pset arguments), they are removed also.
  for (short int el = (invector.size() - 1); el > -1; el--) {
    if (!areThereEnoughHits(invector.at(el))) {
      invector.erase(invector.begin() + el);
    }
  }

  if (invector.empty()) {
    if (debug_)
      LogDebug("HoughGrouping") << "OrderAndFilter - We do not have candidates with the minimum hits required.";
    return;
  } else if (debug_)
    LogDebug("HoughGrouping") << "OrderAndFilter - At the end, we have only " << invector.size() << " good paths!";

  // Packing dt primitives
  for (unsigned short int i = 0; i < invector.size(); i++) {
    DTPrimitivePtrs ptrPrimitive;
    unsigned short int tmplowfill = 0;
    unsigned short int tmpupfill = 0;
    for (unsigned short int lay = 0; lay < 8; lay++) {
      auto dtAux = std::make_shared<DTPrimitive>(invector.at(i).dtHits_[lay]);
      ptrPrimitive.push_back(std::move(dtAux));
      if (debug_) {
        LogDebug("HoughGrouping") << "\nHoughGrouping::OrderAndFilter - cameraid: " << ptrPrimitive[lay]->cameraId();
        LogDebug("HoughGrouping") << "OrderAndFilter - channelid (GOOD): " << ptrPrimitive[lay]->channelId();
        LogDebug("HoughGrouping") << "OrderAndFilter - channelid (AM):   " << ptrPrimitive[lay]->channelId() - 1;
      }
      // Fixing channel ID to AM conventions...
      if (ptrPrimitive[lay]->channelId() != -1)
        ptrPrimitive[lay]->setChannelId(ptrPrimitive[lay]->channelId() - 1);

      if (ptrPrimitive[lay]->cameraId() > 0) {
        if (lay < 4)
          tmplowfill++;
        else
          tmpupfill++;
      }
    }

    auto ptrMuonPath = std::make_shared<MuonPath>(ptrPrimitive, tmplowfill, tmpupfill);
    outMuonPath.push_back(ptrMuonPath);
    if (debug_) {
      for (unsigned short int lay = 0; lay < 8; lay++) {
        LogDebug("HoughGrouping") << "OrderAndFilter - Final cameraID: "
                                  << outMuonPath.back()->primitive(lay)->cameraId();
        LogDebug("HoughGrouping") << "OrderAndFilter - Final channelID: "
                                  << outMuonPath.back()->primitive(lay)->channelId();
        LogDebug("HoughGrouping") << "OrderAndFilter - Final time: "
                                  << outMuonPath.back()->primitive(lay)->tdcTimeStamp();
      }
    }
  }
  return;
}

bool HoughGrouping::areThereEnoughHits(const ProtoCand& tupl) {
  if (debug_)
    LogDebug("HoughGrouping") << "areThereEnoughHits";
  unsigned short int numhitssl1 = 0;
  unsigned short int numhitssl3 = 0;
  for (unsigned short int lay = 0; lay < 8; lay++) {
    if ((tupl.dtHits_[lay].channelId() > 0) && (lay < 4))
      numhitssl1++;
    else if (tupl.dtHits_[lay].channelId() > 0)
      numhitssl3++;
  }

  if (debug_)
    LogDebug("HoughGrouping") << "areThereEnoughHits - Hits in SL1: " << numhitssl1;
  if (debug_)
    LogDebug("HoughGrouping") << "areThereEnoughHits - Hits in SL3: " << numhitssl3;

  if ((numhitssl1 != 0) && (numhitssl3 != 0)) {  // Correlated candidates
    if ((numhitssl1 + numhitssl3) >= minNLayerHits_) {
      if (numhitssl1 > numhitssl3) {
        return ((numhitssl1 >= minSingleSLHitsMax_) && (numhitssl3 >= minSingleSLHitsMin_));
      } else if (numhitssl3 > numhitssl1) {
        return ((numhitssl3 >= minSingleSLHitsMax_) && (numhitssl1 >= minSingleSLHitsMin_));
      } else
        return true;
    }
  } else if (allowUncorrelatedPatterns_) {  // Uncorrelated candidates
    return ((numhitssl1 + numhitssl3) >= minNLayerHits_);
  }
  return false;
}
