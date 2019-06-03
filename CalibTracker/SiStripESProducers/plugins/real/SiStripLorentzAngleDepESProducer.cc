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
  edm::ESGetToken<SiStripLatency, SiStripLatencyRcd> latencyToken_;
  edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleRcd> lorentzAnglePeakToken_;
  edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleRcd> lorentzAngleDeconvToken_;
};

SiStripLorentzAngleDepESProducer::SiStripLorentzAngleDepESProducer(const edm::ParameterSet& iConfig)
{  
  auto cc = setWhatProduced(this);
  
  edm::LogInfo("SiStripLorentzAngleDepESProducer") << "ctor";

  auto getLatency = iConfig.getParameter<edm::ParameterSet>("LatencyRecord");
  // How useful the "record" parameter really is?
  if(getLatency.getParameter<std::string>("record") == "SiStripLatencyRcd") {
    // Shouldn't the label be a tracked parameter?
    cc.setConsumes(latencyToken_, edm::ESInputTag{"", getLatency.getUntrackedParameter<std::string>("label")});
  }
  // Would it make sense to elevate this as an exception?
  else edm::LogError("SiStripLorentzAngleDepESProducer") << "[SiStripLorentzAngleDepESProducer::ctor] No Latency Record found ";

  auto getPeak = iConfig.getParameter<edm::ParameterSet>("BackPlaneCorrectionPeakMode");
  if(getPeak.getParameter<std::string>("record") == "SiStripBackPlaneCorrectionRcd") {
    // Shouldn't the label be a tracked parameter?
    cc.setConsumes(lorentzAnglePeakToken_, edm::ESInputTag{"", getPeak.getUntrackedParameter<std::string>("label")});
  // Would it make sense to elevate this as an exception?
  } else edm::LogError("SiStripLorentzAngleDepESProducer") << "[SiStripLorentzAngleDepESProducer::ctor] No Lorentz Angle Record found " << std::endl;

  auto getDeconv = iConfig.getParameter<edm::ParameterSet>("BackPlaneCorrectionDeconvMode");
  // How useful the "record" parameter really is?
  if(getDeconv.getParameter<std::string>("record") == "SiStripBackPlaneCorrectionRcd") {
    // Shouldn't the label be a tracked parameter?
    cc.setConsumes(lorentzAngleDeconvToken_, edm::ESInputTag{"", getDeconv.getUntrackedParameter<std::string>("label")});
  // Would it make sense to elevate this as an exception?
  } else edm::LogError("SiStripLorentzAngleDepESProducer") << "[SiStripLorentzAngleDepESProducer::ctor] No Lorentz Angle Record found " << std::endl;
}


std::unique_ptr<SiStripLorentzAngle> SiStripLorentzAngleDepESProducer::produce(const SiStripLorentzAngleDepRcd& iRecord)
{
  edm::LogInfo("SiStripLorentzAngleDepESProducer") << "Producer called" << std::endl;
  
  auto const *tokenPtr = &lorentzAngleDeconvToken_;
  
  if(latencyToken_.isInitialized()) {
    const auto& latency = iRecord.get(latencyToken_);
    if(latency.singleReadOutMode() == 1) {
      tokenPtr = &lorentzAnglePeakToken_;
    }
  }

  if (tokenPtr->isInitialized()) {
    // Is this copy really needed? (there is a way to return the very same object)
    return std::make_unique<SiStripLorentzAngle>(iRecord.get(*tokenPtr));
  }
  return nullptr;
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripLorentzAngleDepESProducer);
