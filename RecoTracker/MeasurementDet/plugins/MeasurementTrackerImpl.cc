#include "MeasurementTrackerImpl.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "Geometry/CommonDetUnit/interface/StackGeomDet.h"
#include "Geometry/CommonDetUnit/interface/DoubleSensGeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/Common/interface/ContainerMask.h"

#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TkStripMeasurementDet.h"
#include "TkPixelMeasurementDet.h"
#include "TkPhase2OTMeasurementDet.h"
#include "TkGluedMeasurementDet.h"
#include "TkStackMeasurementDet.h"
#include "TkDoubleSensMeasurementDet.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"

#include <iostream>
#include <typeinfo>
#include <map>
#include <algorithm>

//

using namespace std;

namespace {

  class StrictWeakOrdering {
  public:
    bool operator()(uint32_t p, const uint32_t& i) const { return p < i; }
  };

  struct CmpTKD {
    bool operator()(MeasurementDet const* rh, MeasurementDet const* lh) {
      return rh->fastGeomDet().geographicalId().rawId() < lh->fastGeomDet().geographicalId().rawId();
    }
    bool operator()(MeasurementDet const& rh, MeasurementDet const& lh) {
      return rh.fastGeomDet().geographicalId().rawId() < lh.fastGeomDet().geographicalId().rawId();
    }
  };

  template <typename TKD>
  void sortTKD(std::vector<TKD*>& det) {
    std::sort(det.begin(), det.end(), CmpTKD());
  }
  template <typename TKD>
  void sortTKD(std::vector<TKD>& det) {
    std::sort(det.begin(), det.end(), CmpTKD());
  }

}  // namespace

MeasurementTrackerImpl::MeasurementTrackerImpl(const BadStripCutsDet& badStripCuts,
                                               const PixelClusterParameterEstimator* pixelCPE,
                                               const StripClusterParameterEstimator* stripCPE,
                                               const SiStripRecHitMatcher* hitMatcher,
                                               const TrackerTopology* trackerTopology,
                                               const TrackerGeometry* trackerGeom,
                                               const GeometricSearchTracker* geometricSearchTracker,
                                               const SiStripQuality* stripQuality,
                                               int stripQualityFlags,
                                               int stripQualityDebugFlags,
                                               const SiPixelQuality* pixelQuality,
                                               const SiPixelFedCabling* pixelCabling,
                                               int pixelQualityFlags,
                                               int pixelQualityDebugFlags,
                                               const ClusterParameterEstimator<Phase2TrackerCluster1D>* phase2OTCPE)
    : MeasurementTracker(trackerGeom, geometricSearchTracker),
      theStDetConditions(hitMatcher, stripCPE),
      thePxDetConditions(pixelCPE),
      thePhase2DetConditions(phase2OTCPE) {
  this->initialize(trackerTopology);
  this->initializeStripStatus(badStripCuts, stripQuality, stripQualityFlags, stripQualityDebugFlags);
  this->initializePixelStatus(pixelQuality, pixelCabling, pixelQualityFlags, pixelQualityDebugFlags);
}

MeasurementTrackerImpl::~MeasurementTrackerImpl() {}

void MeasurementTrackerImpl::initialize(const TrackerTopology* trackerTopology) {
  bool subIsPixel = false;
  //FIXME:just temporary solution for phase2 :
  //the OT is defined as PixelSubDetector!
  bool subIsOT = false;

  //if the TkGeometry has the subDet vector filled, the theDetMap is filled, otherwise nothing should happen
  if (!theTrackerGeom->detsPXB().empty()) {
    subIsPixel = GeomDetEnumerators::isTrackerPixel(
        theTrackerGeom->geomDetSubDetector(theTrackerGeom->detsPXB().front()->geographicalId().subdetId()));
    addDets(theTrackerGeom->detsPXB(), subIsPixel, subIsOT);
  }

  if (!theTrackerGeom->detsPXF().empty()) {
    subIsPixel = GeomDetEnumerators::isTrackerPixel(
        theTrackerGeom->geomDetSubDetector(theTrackerGeom->detsPXF().front()->geographicalId().subdetId()));
    addDets(theTrackerGeom->detsPXF(), subIsPixel, subIsOT);
  }

  subIsOT = true;

  if (!theTrackerGeom->detsTIB().empty()) {
    subIsPixel = GeomDetEnumerators::isTrackerPixel(
        theTrackerGeom->geomDetSubDetector(theTrackerGeom->detsTIB().front()->geographicalId().subdetId()));
    addDets(theTrackerGeom->detsTIB(), subIsPixel, subIsOT);
  }

  if (!theTrackerGeom->detsTID().empty()) {
    subIsPixel = GeomDetEnumerators::isTrackerPixel(
        theTrackerGeom->geomDetSubDetector(theTrackerGeom->detsTID().front()->geographicalId().subdetId()));
    addDets(theTrackerGeom->detsTID(), subIsPixel, subIsOT);
  }

  if (!theTrackerGeom->detsTOB().empty()) {
    subIsPixel = GeomDetEnumerators::isTrackerPixel(
        theTrackerGeom->geomDetSubDetector(theTrackerGeom->detsTOB().front()->geographicalId().subdetId()));
    addDets(theTrackerGeom->detsTOB(), subIsPixel, subIsOT);
  }

  if (!theTrackerGeom->detsTEC().empty()) {
    subIsPixel = GeomDetEnumerators::isTrackerPixel(
        theTrackerGeom->geomDetSubDetector(theTrackerGeom->detsTEC().front()->geographicalId().subdetId()));
    addDets(theTrackerGeom->detsTEC(), subIsPixel, subIsOT);
  }

  // fist all stripdets
  sortTKD(theStripDets);
  initStMeasurementConditionSet(theStripDets);
  for (unsigned int i = 0; i != theStripDets.size(); ++i)
    theDetMap[theStDetConditions.id(i)] = &theStripDets[i];

  // now the glued dets
  sortTKD(theGluedDets);
  for (unsigned int i = 0; i != theGluedDets.size(); ++i)
    initGluedDet(theGluedDets[i], trackerTopology);

  // then the pixels
  sortTKD(thePixelDets);
  initPxMeasurementConditionSet(thePixelDets);
  for (unsigned int i = 0; i != thePixelDets.size(); ++i)
    theDetMap[thePxDetConditions.id(i)] = &thePixelDets[i];

  // then the phase2 dets
  sortTKD(thePhase2Dets);
  initPhase2OTMeasurementConditionSet(thePhase2Dets);
  for (unsigned int i = 0; i != thePhase2Dets.size(); ++i)
    theDetMap[thePhase2DetConditions.id(i)] = &thePhase2Dets[i];

  // and then the stack dets, at last
  sortTKD(theStackDets);
  for (unsigned int i = 0; i != theStackDets.size(); ++i)
    initStackDet(theStackDets[i]);

  // and then the double sensor dets
  sortTKD(theDoubleSensGeomDets);
  for (unsigned int i = 0; i != theDoubleSensGeomDets.size(); ++i)
    initDoubleSensDet(theDoubleSensGeomDets[i]);

  if (!checkDets())
    throw MeasurementDetException("Number of dets in MeasurementTracker not consistent with TrackerGeometry!");
}

void MeasurementTrackerImpl::initStMeasurementConditionSet(std::vector<TkStripMeasurementDet>& stripDets) {
  // assume vector is full and ordered!
  int size = stripDets.size();
  theStDetConditions.init(size);
  for (int i = 0; i != size; ++i) {
    auto& mdet = stripDets[i];
    mdet.setIndex(i);
    //intialize the detId !
    theStDetConditions.id_[i] = mdet.specificGeomDet().geographicalId().rawId();
    theStDetConditions.subId_[i] = SiStripDetId(theStDetConditions.id_[i]).subdetId() - 3;
    //initalize the total number of strips
    theStDetConditions.totalStrips_[i] = mdet.specificGeomDet().specificTopology().nstrips();
  }
}

void MeasurementTrackerImpl::initPxMeasurementConditionSet(std::vector<TkPixelMeasurementDet>& pixelDets) {
  // assume vector is full and ordered!
  int size = pixelDets.size();
  thePxDetConditions.init(size);

  for (int i = 0; i != size; ++i) {
    auto& mdet = pixelDets[i];
    mdet.setIndex(i);
    thePxDetConditions.id_[i] = mdet.specificGeomDet().geographicalId().rawId();
  }
}

void MeasurementTrackerImpl::initPhase2OTMeasurementConditionSet(std::vector<TkPhase2OTMeasurementDet>& phase2Dets) {
  // assume vector is full and ordered!
  int size = phase2Dets.size();
  thePhase2DetConditions.init(size);

  for (int i = 0; i != size; ++i) {
    auto& mdet = phase2Dets[i];
    mdet.setIndex(i);
    thePhase2DetConditions.id_[i] = mdet.specificGeomDet().geographicalId().rawId();
  }
}

void MeasurementTrackerImpl::addDets(const TrackingGeometry::DetContainer& dets, bool subIsPixel, bool subIsOT) {
  //in phase2, we can have composed subDetector made by Pixel or Strip
  for (TrackerGeometry::DetContainer::const_iterator gd = dets.begin(); gd != dets.end(); gd++) {
    const GeomDetUnit* gdu = dynamic_cast<const GeomDetUnit*>(*gd);

    //Pixel or Strip GeomDetUnit
    if (gdu->isLeaf()) {
      if (subIsPixel) {
        if (!subIsOT) {
          addPixelDet(*gd);
        } else {
          addPhase2Det(*gd);
        }
      } else {
        addStripDet(*gd);
      }
    } else {
      //Glued or Stack GeomDet
      const GluedGeomDet* gluedDet = dynamic_cast<const GluedGeomDet*>(*gd);
      const StackGeomDet* stackDet = dynamic_cast<const StackGeomDet*>(*gd);
      const DoubleSensGeomDet* doubleSensGeomDet = dynamic_cast<const DoubleSensGeomDet*>(*gd);

      if ((gluedDet == nullptr && stackDet == nullptr && doubleSensGeomDet == nullptr) ||
          (gluedDet != nullptr && stackDet != nullptr && doubleSensGeomDet != nullptr)) {
        throw MeasurementDetException(
            "MeasurementTracker ERROR: GeomDet neither DetUnit nor GluedDet nor StackDet nor DoubleSensGeomDet");
      }
      if (gluedDet != nullptr)
        addGluedDet(gluedDet);
      else if (stackDet != nullptr)
        addStackDet(stackDet);
      else
        addDoubleSensGeomDet(doubleSensGeomDet);
    }
  }
}

bool MeasurementTrackerImpl::checkDets() {
  if (theTrackerGeom->dets().size() == theDetMap.size())
    return true;
  return false;
}

void MeasurementTrackerImpl::addStripDet(const GeomDet* gd) {
  try {
    theStripDets.push_back(TkStripMeasurementDet(gd, theStDetConditions));
  } catch (MeasurementDetException& err) {
    edm::LogError("MeasurementDet") << "Oops, got a MeasurementDetException: " << err.what();
  }
}

void MeasurementTrackerImpl::addPixelDet(const GeomDet* gd) {
  try {
    thePixelDets.push_back(TkPixelMeasurementDet(gd, thePxDetConditions));
  } catch (MeasurementDetException& err) {
    edm::LogError("MeasurementDet") << "Oops, got a MeasurementDetException: " << err.what();
  }
}

void MeasurementTrackerImpl::addPhase2Det(const GeomDet* gd) {
  try {
    thePhase2Dets.push_back(TkPhase2OTMeasurementDet(gd, thePhase2DetConditions));
  } catch (MeasurementDetException& err) {
    edm::LogError("MeasurementDet") << "Oops, got a MeasurementDetException: " << err.what();
  }
}

void MeasurementTrackerImpl::addGluedDet(const GluedGeomDet* gd) {
  theGluedDets.push_back(TkGluedMeasurementDet(gd, theStDetConditions.matcher(), theStDetConditions.stripCPE()));
}

void MeasurementTrackerImpl::addStackDet(const StackGeomDet* gd) {
  //since the Stack will be composed by PS or 2S,
  //both cluster parameter estimators are needed? - right now just the thePixelCPE is used.
  theStackDets.push_back(TkStackMeasurementDet(gd, thePxDetConditions.pixelCPE()));
}

void MeasurementTrackerImpl::addDoubleSensGeomDet(const DoubleSensGeomDet* gd) {
  theDoubleSensGeomDets.push_back(TkDoubleSensMeasurementDet(gd, thePxDetConditions.pixelCPE()));
}

void MeasurementTrackerImpl::initGluedDet(TkGluedMeasurementDet& det, const TrackerTopology* trackerTopology) {
  const GluedGeomDet& gd = det.specificGeomDet();
  const MeasurementDet* monoDet = findDet(gd.monoDet()->geographicalId());
  const MeasurementDet* stereoDet = findDet(gd.stereoDet()->geographicalId());
  if (monoDet == nullptr || stereoDet == nullptr) {
    edm::LogError("MeasurementDet") << "MeasurementTracker ERROR: GluedDet components not found as MeasurementDets ";
    throw MeasurementDetException("MeasurementTracker ERROR: GluedDet components not found as MeasurementDets");
  }
  det.init(monoDet, stereoDet, trackerTopology);
  theDetMap[gd.geographicalId()] = &det;
}

void MeasurementTrackerImpl::initStackDet(TkStackMeasurementDet& det) {
  const StackGeomDet& gd = det.specificGeomDet();
  const MeasurementDet* lowerDet = findDet(gd.lowerDet()->geographicalId());
  const MeasurementDet* upperDet = findDet(gd.upperDet()->geographicalId());
  if (lowerDet == nullptr || upperDet == nullptr) {
    edm::LogError("MeasurementDet") << "MeasurementTracker ERROR: StackDet components not found as MeasurementDets ";
    throw MeasurementDetException("MeasurementTracker ERROR: StackDet components not found as MeasurementDets");
  }
  det.init(lowerDet, upperDet);
  theDetMap[gd.geographicalId()] = &det;
}

void MeasurementTrackerImpl::initDoubleSensDet(TkDoubleSensMeasurementDet& det) {
  const DoubleSensGeomDet& gd = det.specificGeomDet();
  const MeasurementDet* firstDet = findDet(gd.firstDet()->geographicalId());
  const MeasurementDet* secondDet = findDet(gd.secondDet()->geographicalId());
  if (firstDet == nullptr || secondDet == nullptr) {
    edm::LogError("MeasurementDet")
        << "MeasurementTracker ERROR: DoubleSensDet components not found as MeasurementDets ";
    throw MeasurementDetException("MeasurementTracker ERROR: DoubleSensDet components not found as MeasurementDets");
  }
  det.init(firstDet, secondDet);
  theDetMap[gd.geographicalId()] = &det;
}

void MeasurementTrackerImpl::initializeStripStatus(const BadStripCutsDet& badStripCuts,
                                                   const SiStripQuality* quality,
                                                   int qualityFlags,
                                                   int qualityDebugFlags) {
  if (qualityFlags & BadStrips) {
    theStDetConditions.badStripCuts_[SiStripDetId::TIB - 3] = badStripCuts.tib;
    theStDetConditions.badStripCuts_[SiStripDetId::TOB - 3] = badStripCuts.tob;
    theStDetConditions.badStripCuts_[SiStripDetId::TID - 3] = badStripCuts.tid;
    theStDetConditions.badStripCuts_[SiStripDetId::TEC - 3] = badStripCuts.tec;
  }
  theStDetConditions.setMaskBad128StripBlocks((qualityFlags & MaskBad128StripBlocks) != 0);

  if ((quality != nullptr) && (qualityFlags != 0)) {
    edm::LogInfo("MeasurementTracker") << "qualityFlags = " << qualityFlags;
    unsigned int on = 0, tot = 0;
    unsigned int foff = 0, ftot = 0, aoff = 0, atot = 0;
    for (int i = 0; i != theStDetConditions.nDet(); i++) {
      uint32_t detid = theStDetConditions.id(i);
      if (qualityFlags & BadModules) {
        bool isOn = quality->IsModuleUsable(detid);
        theStDetConditions.setActive(i, isOn);
        tot++;
        on += (unsigned int)isOn;
        if (qualityDebugFlags & BadModules) {
          edm::LogInfo("MeasurementTracker")
              << "MeasurementTrackerImpl::initializeStripStatus : detid " << detid << " is " << (isOn ? "on" : "off");
        }
      } else {
        theStDetConditions.setActive(i, true);
      }
      // first turn all APVs and fibers ON
      theStDetConditions.set128StripStatus(i, true);
      if (qualityFlags & BadAPVFibers) {
        short badApvs = quality->getBadApvs(detid);
        short badFibers = quality->getBadFibers(detid);
        for (int j = 0; j < 6; j++) {
          atot++;
          if (badApvs & (1 << j)) {
            theStDetConditions.set128StripStatus(i, false, j);
            aoff++;
          }
        }
        for (int j = 0; j < 3; j++) {
          ftot++;
          if (badFibers & (1 << j)) {
            theStDetConditions.set128StripStatus(i, false, 2 * j);
            theStDetConditions.set128StripStatus(i, false, 2 * j + 1);
            foff++;
          }
        }
      }
      auto& badStrips = theStDetConditions.getBadStripBlocks(i);
      badStrips.clear();
      if (qualityFlags & BadStrips) {
        SiStripBadStrip::Range range = quality->getRange(detid);
        for (SiStripBadStrip::ContainerIterator bit = range.first; bit != range.second; ++bit) {
          badStrips.push_back(quality->decode(*bit));
        }
      }
    }
    if (qualityDebugFlags & BadModules) {
      edm::LogInfo("MeasurementTracker StripModuleStatus")
          << " Total modules: " << tot << ", active " << on << ", inactive " << (tot - on);
    }
    if (qualityDebugFlags & BadAPVFibers) {
      edm::LogInfo("MeasurementTracker StripAPVStatus")
          << " Total APVs: " << atot << ", active " << (atot - aoff) << ", inactive " << (aoff);
      edm::LogInfo("MeasurementTracker StripFiberStatus")
          << " Total Fibers: " << ftot << ", active " << (ftot - foff) << ", inactive " << (foff);
    }
  } else {
    for (int i = 0; i != theStDetConditions.nDet(); i++) {
      theStDetConditions.setActive(i, true);          // module ON
      theStDetConditions.set128StripStatus(i, true);  // all APVs and fibers ON
    }
  }
}

void MeasurementTrackerImpl::initializePixelStatus(const SiPixelQuality* quality,
                                                   const SiPixelFedCabling* pixelCabling,
                                                   int qualityFlags,
                                                   int qualityDebugFlags) {
  if ((quality != nullptr) && (qualityFlags != 0)) {
    edm::LogInfo("MeasurementTracker") << "qualityFlags = " << qualityFlags;
    unsigned int on = 0, tot = 0, badrocs = 0;
    for (std::vector<TkPixelMeasurementDet>::iterator i = thePixelDets.begin(); i != thePixelDets.end(); i++) {
      uint32_t detid = ((*i).geomDet().geographicalId()).rawId();
      if (qualityFlags & BadModules) {
        bool isOn = quality->IsModuleUsable(detid);
        (i)->setActive(isOn);
        tot++;
        on += (unsigned int)isOn;
        if (qualityDebugFlags & BadModules) {
          edm::LogInfo("MeasurementTracker")
              << "MeasurementTrackerImpl::initializePixelStatus : detid " << detid << " is " << (isOn ? "on" : "off");
        }
      } else {
        (i)->setActive(true);
      }
      if ((qualityFlags & BadROCs) && (quality->getBadRocs(detid) != 0)) {
        std::vector<LocalPoint> badROCs = quality->getBadRocPositions(detid, *theTrackerGeom, pixelCabling);
        badrocs += badROCs.size();
        (i)->setBadRocPositions(badROCs);
      } else {
        (i)->clearBadRocPositions();
      }
    }
    if (qualityDebugFlags & BadModules) {
      edm::LogInfo("MeasurementTracker PixelModuleStatus")
          << " Total modules: " << tot << ", active " << on << ", inactive " << (tot - on);
    }
    if (qualityDebugFlags & BadROCs) {
      edm::LogInfo("MeasurementTracker PixelROCStatus") << " Total of bad ROCs: " << badrocs;
    }
  } else {
    for (std::vector<TkPixelMeasurementDet>::iterator i = thePixelDets.begin(); i != thePixelDets.end(); i++) {
      (i)->setActive(true);  // module ON
    }
  }
}
