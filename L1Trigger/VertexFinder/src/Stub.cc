#include "L1Trigger/VertexFinder/interface/Stub.h"

namespace l1tVertexFinder {

  //=== Store useful info about this stub.

  Stub::Stub()
      : settings_(nullptr),
        phi_(0.),
        r_(0.),
        z_(0.),
        idDet_(0),
        moduleMinR_(0.),
        moduleMaxR_(0.),
        moduleMinPhi_(0.),
        moduleMaxPhi_(0.),
        moduleMinZ_(0.),
        moduleMaxZ_(0.),
        psModule_(false),
        layerId_(0),
        endcapRing_(0),
        barrel_(false),
        sigmaPerp_(0.),
        sigmaPar_(0.),
        stripPitch_(0.),
        stripLength_(0.),
        nStrips_(0),
        sensorWidth_(0.),
        outerModuleAtSmallerR_(false) {}

  Stub::Stub(const TTStubRef& ttStubRef,
             const AnalysisSettings& settings,
             const TrackerGeometry* trackerGeometry,
             const TrackerTopology* trackerTopology)
      : TTStubRef(ttStubRef), settings_(&settings) {
    auto geoDetId = trackerGeometry->idToDet(ttStubRef->clusterRef(0)->getDetId())->geographicalId();
    auto theGeomDet = trackerGeometry->idToDet(geoDetId);
    auto measurementPoint = ttStubRef->clusterRef(0)->findAverageLocalCoordinatesCentered();
    auto pos = theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(measurementPoint));

    phi_ = pos.phi();
    r_ = pos.perp();
    z_ = pos.z();

    if (r_ < settings_->trackerInnerRadius() || r_ > settings_->trackerOuterRadius() ||
        std::abs(z_) > settings_->trackerHalfLength()) {
      throw cms::Exception(
          "Stub: Stub found outside assumed tracker volume. Please update tracker dimensions specified in Settings.h!")
          << " r=" << r_ << " z=" << z_ << " " << ttStubRef->getDetId().subdetId() << std::endl;
    }

    // Set info about the module this stub is in
    this->setModuleInfo(trackerGeometry, trackerTopology, geoDetId);
  }

  //=== Note which tracking particle(s), if any, produced this stub.
  void Stub::fillTruth(edm::Handle<TTStubAssMap> mcTruthTTStubHandle,
                       edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle) {
    const TTStubRef& ttStubRef(*this);  // Cast to base class

    //--- Fill assocTP_ info. If both clusters in this stub were produced by the same single tracking particle, find out which one it was.

    // Require same TP contributed to both clusters.
    if (mcTruthTTStubHandle->isGenuine(ttStubRef)) {
      assocTP_ = mcTruthTTStubHandle->findTrackingParticlePtr(ttStubRef);
    }

    // Fill assocTPs_ info.
    if (settings_->stubMatchStrict()) {
      // We consider only stubs in which this TP contributed to both clusters.
      if (!assocTP_.isNull())
        assocTPs_.insert(assocTP_);
    } else {
      // We consider stubs in which this TP contributed to either cluster.

      for (unsigned int iClus = 0; iClus <= 1; iClus++) {  // Loop over both clusters that make up stub.
        const TTClusterRef& ttClusterRef = ttStubRef->clusterRef(iClus);

        // Now identify all TP's contributing to either cluster in stub.
        std::vector<edm::Ptr<TrackingParticle>> vecTpPtr =
            mcTruthTTClusterHandle->findTrackingParticlePtrs(ttClusterRef);

        for (const edm::Ptr<TrackingParticle>& tpPtr : vecTpPtr) {
          assocTPs_.insert(tpPtr);
        }
      }
    }
  }

  void Stub::setModuleInfo(const TrackerGeometry* trackerGeometry,
                           const TrackerTopology* trackerTopology,
                           const DetId& detId) {
    idDet_ = detId();

    // Get min & max (r,phi,z) coordinates of the centre of the two sensors containing this stub.
    const GeomDetUnit* det0 = trackerGeometry->idToDetUnit(detId);
    const GeomDetUnit* det1 = trackerGeometry->idToDetUnit(trackerTopology->partnerDetId(detId));

    float R0 = det0->position().perp();
    float R1 = det1->position().perp();
    float PHI0 = det0->position().phi();
    float PHI1 = det1->position().phi();
    float Z0 = det0->position().z();
    float Z1 = det1->position().z();
    moduleMinR_ = std::min(R0, R1);
    moduleMaxR_ = std::max(R0, R1);
    moduleMinPhi_ = std::min(PHI0, PHI1);
    moduleMaxPhi_ = std::max(PHI0, PHI1);
    moduleMinZ_ = std::min(Z0, Z1);
    moduleMaxZ_ = std::max(Z0, Z1);

    // Note if module is PS or 2S, and whether in barrel or endcap.
    psModule_ =
        trackerGeometry->getDetectorType(detId) ==
        TrackerGeometry::ModuleType::
            Ph2PSP;  // From https://github.com/cms-sw/cmssw/blob/CMSSW_8_1_X/Geometry/TrackerGeometryBuilder/README.md
    barrel_ = detId.subdetId() == StripSubdetector::TOB || detId.subdetId() == StripSubdetector::TIB;

    // Encode layer ID.
    if (barrel_) {
      layerId_ = trackerTopology->layer(detId);  // barrel layer 1-6 encoded as 1-6
    } else {
      // layerId_ = 10*detId.iSide() + detId.iDisk(); // endcap layer 1-5 encoded as 11-15 (endcap A) or 21-25 (endcapB)
      // EJC This seems to give the same encoding
      layerId_ = 10 * trackerTopology->side(detId) + trackerTopology->tidWheel(detId);
    }

    // Note module ring in endcap
    // endcapRing_ = barrel_  ?  0  :  detId.iRing();
    endcapRing_ = barrel_ ? 0 : trackerTopology->tidRing(detId);

    // Get sensor strip or pixel pitch using innermost sensor of pair.

    const PixelGeomDetUnit* unit = reinterpret_cast<const PixelGeomDetUnit*>(det0);
    const PixelTopology& topo = unit->specificTopology();
    const Bounds& bounds = det0->surface().bounds();

    std::pair<float, float> pitch = topo.pitch();
    stripPitch_ = pitch.first;      // Strip pitch (or pixel pitch along shortest axis)
    stripLength_ = pitch.second;    //  Strip length (or pixel pitch along longest axis)
    nStrips_ = topo.nrows();        // No. of strips in sensor
    sensorWidth_ = bounds.width();  // Width of sensitive region of sensor (= stripPitch * nStrips).

    outerModuleAtSmallerR_ = false;
    if (barrel_ && det0->position().perp() > det1->position().perp()) {
      outerModuleAtSmallerR_ = true;
    }

    sigmaPerp_ = stripPitch_ / sqrt(12.);  // resolution perpendicular to strip (or to longest pixel axis)
    sigmaPar_ = stripLength_ / sqrt(12.);  // resolution parallel to strip (or to longest pixel axis)
  }

}  // end namespace l1tVertexFinder
