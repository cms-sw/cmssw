// -*- C++ -*-
//
// Package:    L1Trigger/L1TZDC
// Class:      L1TZDC
//
/**\class L1TZDC L1TZDCMapper.cc L1Trigger/L1TZDC/plugins/L1TZDCMapper.cc

 Description: ZDC Mapper for L1 Trigger emulation

 Implementation:
 Modified L1TZDCProducer to work w/ hcalTps w/ ZDC included at abs(ieta) 42
 ZDC EtSums found at iphi 99
*/
//
// Copied/modded from L1TZDCProducer by: Chris McGinn
//        Copy Made: Wed, 18 Sep 2023
//        Contact: christopher.mc.ginn@cern.ch or
//                 cfmcginn on github for bugs/issues
//
#include <memory>
#include <iostream>
// user include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2FirmwareFactory.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2MainProcessor.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsO2ORcd.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

//
// class declaration
//

using namespace l1t;

class L1TZDCMapper : public edm::stream::EDProducer<> {
public:
  explicit L1TZDCMapper(const edm::ParameterSet& ps);
  ~L1TZDCMapper() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void beginRun(edm::Run const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------

  // input tokens
  //Add the hcalTP token
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalTPSource;

  //input ints
  int bxFirst_;
  int bxLast_;

  // put tokens
  edm::EDPutTokenT<EtSumBxCollection> etToken_;
};

L1TZDCMapper::L1TZDCMapper(const edm::ParameterSet& ps)
  : hcalTPSource(consumes<HcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("hcalTPDigis"))) {
  // register what you produce
  etToken_ = produces<EtSumBxCollection>();

  bxFirst_ = ps.getParameter<int>("bxFirst");
  bxLast_ = ps.getParameter<int>("bxLast");
}

// ------------ method called to produce the data  ------------
void L1TZDCMapper::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace l1t;

  LogDebug("L1TZDCMapper") << "L1TZDCMapper::produce function called..." << std::endl;

  // reduced collection to be emplaced in output
  EtSumBxCollection etsumsReduced(0, bxFirst_, bxLast_);

  edm::Handle<HcalTrigPrimDigiCollection> hcalTPs;
  iEvent.getByToken(hcalTPSource, hcalTPs);
  
  //Check if the hcal tokens are valid
  if (hcalTPs.isValid()) {      
    
    //If valid, process
    for (const auto& hcalTp : *hcalTPs) {
      int ieta = hcalTp.id().ieta();	
      uint32_t absIEta = std::abs(ieta);

      //absIEta position 42 is used for the ZDC; -42 for ZDCM, +42 for ZDCP
      if(absIEta != 42) continue;
      
      int iphi = hcalTp.id().iphi();
      //iphi position 99 is used for the etSums
      if(iphi != 99) continue;

      //Get number of samples and number of presamples; bx 0 is nPresamples      
      int nSamples = hcalTp.size();
      int nPresamples = hcalTp.presamples();
      
      for (int ibx = 0; ibx < nSamples; ibx++) {
	if (ibx >= nPresamples + bxFirst_ && ibx <= nPresamples + bxLast_) {
	  HcalTriggerPrimitiveSample hcalTpSample = hcalTp.sample(ibx);	  	  
	  int ietIn = hcalTpSample.compressedEt();
	  	  
	  l1t::EtSum tempEt = l1t::EtSum();
	  tempEt.setHwPt(ietIn);
	  tempEt.setHwPhi(0.);

	  //ieta < 0 is ZDCMinus; > 0 is ZDCPlus
	  if(ieta < 0){
	    tempEt.setHwEta(-1.);
	    tempEt.setType(EtSum::EtSumType::kZDCM);
	  }
	  else{
	    tempEt.setHwEta(1.);
	    tempEt.setType(EtSum::EtSumType::kZDCP);
	  }

	  //By construction, nPresamples is 0 bx (since presamples span 0 to nPresamples-1)
	  etsumsReduced.push_back(ibx - nPresamples, CaloTools::etSumP4Demux(tempEt));
	}	  
      }
    }
  }
  else{   
    // If the collection is not valid issue a warning before putting an empty collection
    edm::LogWarning("L1TZDCMapper") << "hcalTps not valid; return empty ZDC Et Sum BXCollection" << std::endl;
  }

  // Emplace even if !hcalTps.isValid()
  // Output in this case will be an empty collection
  iEvent.emplace(etToken_, std::move(etsumsReduced));
}

// ------------ method called when starting to processes a run  ------------
void L1TZDCMapper::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TZDCMapper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hcalTPDigis", edm::InputTag("simHcalTriggerPrimitiveDigis"));
  desc.add<int>("bxFirst", -2);
  desc.add<int>("bxLast", 2);
  descriptions.add("l1tZDCMapper", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TZDCMapper);
