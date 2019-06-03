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
  edm::ESGetToken<SiStripLatency, SiStripLatencyRcd> latencyToken_;
  edm::ESGetToken<SiStripBackPlaneCorrection, SiStripBackPlaneCorrectionRcd> backPlaneCorrectionPeakToken_;
  edm::ESGetToken<SiStripBackPlaneCorrection, SiStripBackPlaneCorrectionRcd> backPlaneCorrectionDeconvToken_;
};

SiStripBackPlaneCorrectionDepESProducer::SiStripBackPlaneCorrectionDepESProducer(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this);

  edm::LogInfo("SiStripBackPlaneCorrectionDepESProducer") << "ctor";

  auto getLatency = iConfig.getParameter<edm::ParameterSet>("LatencyRecord");
  // How useful the "record" parameter really is?
  if(getLatency.getParameter<std::string>("record") == "SiStripLatencyRcd") {
    // Shouldn't the label be a tracked parameter?
    cc.setConsumes(latencyToken_, edm::ESInputTag{"", getLatency.getUntrackedParameter<std::string>("label")});
  }
  // Would it make sense to elevate this as an exception?
  else edm::LogError("SiStripBackPlaneCorrectionDepESProducer") << "[SiStripBackPlaneCorrectionDepESProducer::ctor] No Latency Record found ";

  auto getPeak = iConfig.getParameter<edm::ParameterSet>("BackPlaneCorrectionPeakMode");
  if(getPeak.getParameter<std::string>("record") == "SiStripBackPlaneCorrectionRcd") {
    // Shouldn't the label be a tracked parameter?
    cc.setConsumes(backPlaneCorrectionPeakToken_, edm::ESInputTag{"", getPeak.getUntrackedParameter<std::string>("label")});
  // Would it make sense to elevate this as an exception?
  } else edm::LogError("SiStripBackPlaneCorrectionDepESProducer") << "[SiStripBackPlaneCorrectionDepESProducer::ctor] No Lorentz Angle Record found " << std::endl;


  auto getDeconv = iConfig.getParameter<edm::ParameterSet>("BackPlaneCorrectionDeconvMode");
  // How useful the "record" parameter really is?
  if(getDeconv.getParameter<std::string>("record") == "SiStripBackPlaneCorrectionRcd") {
    // Shouldn't the label be a tracked parameter?
    cc.setConsumes(backPlaneCorrectionDeconvToken_, edm::ESInputTag{"", getDeconv.getUntrackedParameter<std::string>("label")});
  // Would it make sense to elevate this as an exception?
  } else edm::LogError("SiStripBackPlaneCorrectionDepESProducer") << "[SiStripBackPlaneCorrectionDepESProducer::ctor] No Lorentz Angle Record found " << std::endl;
}


std::unique_ptr<SiStripBackPlaneCorrection> SiStripBackPlaneCorrectionDepESProducer::produce(const SiStripBackPlaneCorrectionDepRcd& iRecord)
{
  edm::LogInfo("SiStripBackPlaneCorrectionDepESProducer") << "Producer called" << std::endl;
  
  auto const *tokenPtr = &backPlaneCorrectionDeconvToken_;
  
  if(latencyToken_.isInitialized()) {
    const auto& latency = iRecord.get(latencyToken_);
    if(latency.singleReadOutMode() == 1) {
      tokenPtr = &backPlaneCorrectionPeakToken_;
    }
  }

  if (tokenPtr->isInitialized()) {
    // Is this copy really needed? (there is a way to return the very same object)
    return std::make_unique<SiStripBackPlaneCorrection>(iRecord.get(*tokenPtr));
  }
  return nullptr;
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripBackPlaneCorrectionDepESProducer);
