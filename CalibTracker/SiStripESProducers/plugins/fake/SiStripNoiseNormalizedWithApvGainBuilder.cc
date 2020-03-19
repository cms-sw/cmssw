#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripNoiseNormalizedWithApvGainBuilder.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SiStripFakeAPVParameters.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

SiStripNoiseNormalizedWithApvGainBuilder::SiStripNoiseNormalizedWithApvGainBuilder(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)),
      pset_(iConfig),
      electronsPerADC_(0.),
      minimumPosValue_(0.),
      stripLengthMode_(true),
      printDebug_(0) {
  tTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  tGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  inputApvGainToken_ = esConsumes<SiStripApvGain, SiStripApvGainRcd>();
}

void SiStripNoiseNormalizedWithApvGainBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  const auto& tTopo = iSetup.getData(tTopoToken_);
  const auto& tGeom = iSetup.getData(tGeomToken_);
  // Read the gain from the given tag
  const auto& inputApvGain = iSetup.getData(inputApvGainToken_);
  std::vector<uint32_t> inputDetIds;
  inputApvGain.getDetIds(inputDetIds);

  // Prepare the new object
  SiStripNoises* obj = new SiStripNoises();

  stripLengthMode_ = pset_.getParameter<bool>("StripLengthMode");

  //parameters for random noise generation. not used if Strip length mode is chosen
  SiStripFakeAPVParameters meanNoise{pset_, "MeanNoise"};
  SiStripFakeAPVParameters sigmaNoise{pset_, "SigmaNoise"};
  minimumPosValue_ = pset_.getParameter<double>("MinPositiveNoise");

  //parameters for strip length proportional noise generation. not used if random mode is chosen
  SiStripFakeAPVParameters noiseStripLengthLinearSlope{pset_, "NoiseStripLengthSlope"};
  SiStripFakeAPVParameters noiseStripLengthLinearQuote{pset_, "NoiseStripLengthQuote"};
  electronsPerADC_ = pset_.getParameter<double>("electronPerAdc");

  printDebug_ = pset_.getUntrackedParameter<uint32_t>("printDebug", 5);

  unsigned int count = 0;
  for (const auto det : tGeom.detUnits()) {
    const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(det);
    if (stripDet != nullptr) {
      const DetId detId = stripDet->geographicalId();
      // Find if this DetId is in the input tag and if so how many are the Apvs for which it contains information
      SiStripApvGain::Range inputRange(inputApvGain.getRange(detId));

      //Generate Noises for det detid
      SiStripNoises::InputVector theSiStripVector;
      float noise = 0.;
      SiStripFakeAPVParameters::index sl = SiStripFakeAPVParameters::getIndex(&tTopo, detId);
      unsigned short nApvs = stripDet->specificTopology().nstrips() / 128;

      if (stripLengthMode_) {
        // Use strip length
        double linearSlope = noiseStripLengthLinearSlope.get(sl);
        double linearQuote = noiseStripLengthLinearQuote.get(sl);
        double stripLength = stripDet->specificTopology().stripLength();
        for (unsigned short j = 0; j < nApvs; ++j) {
          double gain = inputApvGain.getApvGain(j, inputRange);

          for (unsigned short stripId = 0; stripId < 128; ++stripId) {
            noise = ((linearSlope * stripLength + linearQuote) / electronsPerADC_) * gain;
            if (count < printDebug_)
              printLog(detId, stripId + 128 * j, noise);
            obj->setData(noise, theSiStripVector);
          }
        }
      } else {
        // Use random generator
        double meanN = meanNoise.get(sl);
        double sigmaN = sigmaNoise.get(sl);
        for (unsigned short j = 0; j < nApvs; ++j) {
          double gain = inputApvGain.getApvGain(j, inputRange);

          for (unsigned short stripId = 0; stripId < 128; ++stripId) {
            noise = (CLHEP::RandGauss::shoot(meanN, sigmaN)) * gain;
            if (noise <= minimumPosValue_)
              noise = minimumPosValue_;
            if (count < printDebug_)
              printLog(detId, stripId + 128 * j, noise);
            obj->setData(noise, theSiStripVector);
          }
        }
      }
      ++count;

      if (!obj->put(detId, theSiStripVector)) {
        edm::LogError("SiStripNoisesFakeESSource::produce ") << " detid already exists" << std::endl;
      }
    }
  }

  //End now write data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripNoisesRcd")) {
      mydbservice->createNewIOV<SiStripNoises>(
          obj, mydbservice->beginOfTime(), mydbservice->endOfTime(), "SiStripNoisesRcd");
    } else {
      mydbservice->appendSinceTime<SiStripNoises>(obj, mydbservice->currentTime(), "SiStripNoisesRcd");
    }
  } else {
    edm::LogError("SiStripNoiseNormalizedWithApvGainBuilder") << "Service is unavailable" << std::endl;
  }
}
