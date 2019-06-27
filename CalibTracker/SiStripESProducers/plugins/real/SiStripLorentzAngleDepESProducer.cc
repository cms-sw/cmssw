// -*- C++ -*-
//
// Package:    SiStripLorentzAngleDepESProducer
// Class:      SiStripLorentzAngleDepESProducer
//
/**\class SiStripLorentzAngleDepESProducer SiStripLorentzAngleDepESProducer.h CalibTracker/SiStripESProducers/plugins/real/SiStripLorentzAngleDepESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Segala and Rebeca Gonzalez Suarez
//         Created:  15/02/2011
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

class SiStripLorentzAngleDepESProducer : public edm::ESProducer {
public:
  SiStripLorentzAngleDepESProducer(const edm::ParameterSet&);
  ~SiStripLorentzAngleDepESProducer() override{};

  std::unique_ptr<SiStripLorentzAngle> produce(const SiStripLorentzAngleDepRcd&);

private:
  edm::ParameterSet getLatency;
  edm::ParameterSet getPeak;
  edm::ParameterSet getDeconv;
};

SiStripLorentzAngleDepESProducer::SiStripLorentzAngleDepESProducer(const edm::ParameterSet& iConfig)
    : getLatency(iConfig.getParameter<edm::ParameterSet>("LatencyRecord")),
      getPeak(iConfig.getParameter<edm::ParameterSet>("LorentzAnglePeakMode")),
      getDeconv(iConfig.getParameter<edm::ParameterSet>("LorentzAngleDeconvMode")) {
  setWhatProduced(this);

  edm::LogInfo("SiStripLorentzAngleDepESProducer") << "ctor" << std::endl;
}

std::unique_ptr<SiStripLorentzAngle> SiStripLorentzAngleDepESProducer::produce(
    const SiStripLorentzAngleDepRcd& iRecord) {
  std::unique_ptr<SiStripLorentzAngle> siStripLA;
  edm::LogInfo("SiStripLorentzAngleDepESProducer") << "Producer called" << std::endl;

  std::string latencyRecordName = getLatency.getParameter<std::string>("record");
  std::string latencyLabel = getLatency.getUntrackedParameter<std::string>("label");
  bool peakMode = false;

  if (latencyRecordName == "SiStripLatencyRcd") {
    edm::ESHandle<SiStripLatency> latency;
    iRecord.getRecord<SiStripLatencyRcd>().get(latencyLabel, latency);
    if (latency->singleReadOutMode() == 1)
      peakMode = true;
  } else
    edm::LogError("SiStripLorentzAngleDepESProducer")
        << "[SiStripLorentzAngleDepESProducer::produce] No Latency Record found " << std::endl;

  std::string lorentzAngleRecordName;
  std::string lorentzAngleLabel;

  if (peakMode) {
    lorentzAngleRecordName = getPeak.getParameter<std::string>("record");
    lorentzAngleLabel = getPeak.getUntrackedParameter<std::string>("label");
  } else {
    lorentzAngleRecordName = getDeconv.getParameter<std::string>("record");
    lorentzAngleLabel = getDeconv.getUntrackedParameter<std::string>("label");
  }

  if (lorentzAngleRecordName == "SiStripLorentzAngleRcd") {
    edm::ESHandle<SiStripLorentzAngle> siStripLorentzAngle;
    iRecord.getRecord<SiStripLorentzAngleRcd>().get(lorentzAngleLabel, siStripLorentzAngle);
    siStripLA.reset(new SiStripLorentzAngle(*(siStripLorentzAngle.product())));
  } else
    edm::LogError("SiStripLorentzAngleDepESProducer")
        << "[SiStripLorentzAngleDepESProducer::produce] No Lorentz Angle Record found " << std::endl;

  return siStripLA;
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripLorentzAngleDepESProducer);
