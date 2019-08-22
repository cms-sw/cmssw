#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripApvGainBuilderFromTag.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include "SiStripFakeAPVParameters.h"

SiStripApvGainBuilderFromTag::SiStripApvGainBuilderFromTag(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)), pset_(iConfig) {
  tTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  tGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  inputApvGainToken_ = esConsumes<SiStripApvGain, SiStripApvGainRcd>();
}

void SiStripApvGainBuilderFromTag::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  const auto& tTopo = iSetup.getData(tTopoToken_);
  const auto& tGeom = iSetup.getData(tGeomToken_);

  //   unsigned int run=evt.id().run();

  std::string genMode = pset_.getParameter<std::string>("genMode");
  bool applyTuning = pset_.getParameter<bool>("applyTuning");

  double meanGain_ = pset_.getParameter<double>("MeanGain");
  double sigmaGain_ = pset_.getParameter<double>("SigmaGain");
  double minimumPosValue_ = pset_.getParameter<double>("MinPositiveGain");

  uint32_t printdebug_ = pset_.getUntrackedParameter<uint32_t>("printDebug", 5);

  //parameters for layer/disk level correction; not used if applyTuning=false
  SiStripFakeAPVParameters correct{pset_, "correct"};

  // Read the gain from the given tag
  const auto& inputApvGain = iSetup.getData(inputApvGainToken_);
  std::vector<uint32_t> inputDetIds;
  inputApvGain.getDetIds(inputDetIds);

  // Prepare the new object
  SiStripApvGain* obj = new SiStripApvGain();

  uint32_t count = 0;
  for (const auto det : tGeom.detUnits()) {
    const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(det);
    if (stripDet != nullptr) {
      const DetId detid = stripDet->geographicalId();
      // Find if this DetId is in the input tag and if so how many are the Apvs for which it contains information
      SiStripApvGain::Range inputRange;
      size_t inputRangeSize = 0;
      if (find(inputDetIds.begin(), inputDetIds.end(), detid) != inputDetIds.end()) {
        inputRange = inputApvGain.getRange(detid);
        inputRangeSize = distance(inputRange.first, inputRange.second);
      }

      std::vector<float> theSiStripVector;
      for (unsigned short j = 0; j < (stripDet->specificTopology().nstrips() / 128); j++) {
        double gainValue = meanGain_;

        if (j < inputRangeSize) {
          gainValue = inputApvGain.getApvGain(j, inputRange);
          // cout << "Gain = " << gainValue <<" from input tag for DetId = " << detid.rawId() << " and apv = " << j << endl;
        }
        // else {
        //   cout << "No gain in input tag for DetId = " << detid << " and apv = " << j << " using value from cfg = " << gainValue << endl;
        // }

        // corrections at layer/disk level:
        SiStripFakeAPVParameters::index sl = SiStripFakeAPVParameters::getIndex(&tTopo, detid);
        //unsigned short nApvs = (stripDet->specificTopology().nstrips()/128);
        if (applyTuning) {
          double correction = correct.get(sl);
          gainValue *= correction;
        }

        // smearing:
        if (genMode == "gaussian") {
          gainValue = CLHEP::RandGauss::shoot(gainValue, sigmaGain_);
          if (gainValue <= minimumPosValue_)
            gainValue = minimumPosValue_;
        } else if (genMode != "default") {
          LogDebug("SiStripApvGain") << "ERROR: wrong genMode specifier : " << genMode
                                     << ", please select one of \"default\" or \"gaussian\"" << std::endl;
          exit(1);
        }

        if (count < printdebug_) {
          edm::LogInfo("SiStripApvGainGeneratorFromTag")
              << "detid: " << detid.rawId() << " Apv: " << j << " gain: " << gainValue << std::endl;
        }
        theSiStripVector.push_back(gainValue);
      }
      count++;
      SiStripApvGain::Range range(theSiStripVector.begin(), theSiStripVector.end());
      if (!obj->put(detid, range))
        edm::LogError("SiStripApvGainGeneratorFromTag") << " detid already exists" << std::endl;
    }
  }

  //End now write data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripApvGainRcd2")) {
      mydbservice->createNewIOV<SiStripApvGain>(
          obj, mydbservice->beginOfTime(), mydbservice->endOfTime(), "SiStripApvGainRcd2");
    } else {
      mydbservice->appendSinceTime<SiStripApvGain>(obj, mydbservice->currentTime(), "SiStripApvGainRcd2");
    }
  } else {
    edm::LogError("SiStripApvGainBuilderFromTag") << "Service is unavailable" << std::endl;
  }
}
