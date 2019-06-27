// -*- C++ -*-
//
// Package:    SiStripBackPlaneCorrectionDepESProducer
// Class:      SiStripBackPlaneCorrectionDepESProducer
//
/**\class SiStripBackPlaneCorrectionDepESProducer SiStripBackPlaneCorrectionDepESProducer.h CalibTracker/SiStripESProducers/plugins/real/SiStripBackPlaneCorrectionDepESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic Quertenmont inspired from the SiStripLorentzAngleDepESProducer
//         Created:  07/03/2013
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
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

class SiStripBackPlaneCorrectionDepESProducer : public edm::ESProducer {
public:
  SiStripBackPlaneCorrectionDepESProducer(const edm::ParameterSet&);
  ~SiStripBackPlaneCorrectionDepESProducer() override{};

  std::unique_ptr<SiStripBackPlaneCorrection> produce(const SiStripBackPlaneCorrectionDepRcd&);

private:
  edm::ParameterSet getLatency;
  edm::ParameterSet getPeak;
  edm::ParameterSet getDeconv;
};

SiStripBackPlaneCorrectionDepESProducer::SiStripBackPlaneCorrectionDepESProducer(const edm::ParameterSet& iConfig)
    : getLatency(iConfig.getParameter<edm::ParameterSet>("LatencyRecord")),
      getPeak(iConfig.getParameter<edm::ParameterSet>("BackPlaneCorrectionPeakMode")),
      getDeconv(iConfig.getParameter<edm::ParameterSet>("BackPlaneCorrectionDeconvMode")) {
  setWhatProduced(this);

  edm::LogInfo("SiStripBackPlaneCorrectionDepESProducer") << "ctor" << std::endl;
}

std::unique_ptr<SiStripBackPlaneCorrection> SiStripBackPlaneCorrectionDepESProducer::produce(
    const SiStripBackPlaneCorrectionDepRcd& iRecord) {
  std::unique_ptr<SiStripBackPlaneCorrection> siStripBPC;
  edm::LogInfo("SiStripBackPlaneCorrectionDepESProducer") << "Producer called" << std::endl;

  std::string latencyRecordName = getLatency.getParameter<std::string>("record");
  std::string latencyLabel = getLatency.getUntrackedParameter<std::string>("label");
  bool peakMode = false;

  if (latencyRecordName == "SiStripLatencyRcd") {
    edm::ESHandle<SiStripLatency> latency;
    iRecord.getRecord<SiStripLatencyRcd>().get(latencyLabel, latency);
    if (latency->singleReadOutMode() == 1)
      peakMode = true;
  } else
    edm::LogError("SiStripBackPlaneCorrectionDepESProducer")
        << "[SiStripBackPlaneCorrectionDepESProducer::produce] No Latency Record found " << std::endl;

  std::string backPlaneCorrectionRecordName;
  std::string backPlaneCorrectionLabel;

  if (peakMode) {
    backPlaneCorrectionRecordName = getPeak.getParameter<std::string>("record");
    backPlaneCorrectionLabel = getPeak.getUntrackedParameter<std::string>("label");
  } else {
    backPlaneCorrectionRecordName = getDeconv.getParameter<std::string>("record");
    backPlaneCorrectionLabel = getDeconv.getUntrackedParameter<std::string>("label");
  }

  if (backPlaneCorrectionRecordName == "SiStripBackPlaneCorrectionRcd") {
    edm::ESHandle<SiStripBackPlaneCorrection> siStripBackPlaneCorrection;
    iRecord.getRecord<SiStripBackPlaneCorrectionRcd>().get(backPlaneCorrectionLabel, siStripBackPlaneCorrection);
    siStripBPC = std::make_unique<SiStripBackPlaneCorrection>(*(siStripBackPlaneCorrection.product()));
  } else
    edm::LogError("SiStripBackPlaneCorrectionDepESProducer")
        << "[SiStripBackPlaneCorrectionDepESProducer::produce] No Lorentz Angle Record found " << std::endl;

  return siStripBPC;
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripBackPlaneCorrectionDepESProducer);
