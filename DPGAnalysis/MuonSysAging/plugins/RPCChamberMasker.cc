// -*- C++ -*-
//
// Package:    L1Trigger/L1IntegratedMuonTrigger/RPCChamberMasker
// Class:      RPCChamberMasker
// 
/**\class RPCChamberMasker RPCChamberMasker.cc L1Trigger/L1IntegratedMuonTrigger/RPCChamberMasker/plugins/RPCChamberMasker.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Borislav Pavlov
//         Created:  Tue, 16 Jun 2015 13:02:20 GMT
//
//


// system include files
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <regex>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/MuonSystemAging/interface/MuonSystemAging.h"
#include "CondFormats/DataRecord/interface/MuonSystemAgingRcd.h"
//
// class declaration
//

class RPCChamberMasker : public edm::EDProducer {
   public:
      explicit RPCChamberMasker(const edm::ParameterSet&);
      ~RPCChamberMasker();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
      
      virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
  edm::InputTag digiTag_;
  edm::EDGetTokenT<RPCDigiCollection> m_digiTag;
  std::vector<int> m_maskedRPCIDs;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
RPCChamberMasker::RPCChamberMasker(const edm::ParameterSet& iConfig) : 
  digiTag_(iConfig.getParameter<edm::InputTag>("digiTag") )
{
   m_digiTag = consumes<RPCDigiCollection>(digiTag_);
 
  //std::cout<<"RPCChamberMasker::RPCChamberMasker"<<std::endl;
  produces<RPCDigiCollection>();

/*  std::cout<<"IDs of masked RPCs"<<std::endl;
  for ( auto rpc_ids : iConfig.getParameter<std::vector<int>>("maskedRPCIDs"))
    {
      m_maskedRPCIDs.push_back(rpc_ids);
      std::cout<<rpc_ids<<std::endl;
    }
*/
  
}


RPCChamberMasker::~RPCChamberMasker()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
RPCChamberMasker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  std::unique_ptr<RPCDigiCollection> filteredDigis(new RPCDigiCollection());
  if (!digiTag_.label().empty())
    {
      edm::Handle<RPCDigiCollection> rpcDigis;
      iEvent.getByToken(m_digiTag, rpcDigis);
      
      RPCDigiCollection::DigiRangeIterator rpcLayerIdIt  = rpcDigis->begin();
      RPCDigiCollection::DigiRangeIterator rpcLayerIdEnd = rpcDigis->end();
      
      for (; rpcLayerIdIt != rpcLayerIdEnd; ++rpcLayerIdIt)
	  {
          int id = ((*rpcLayerIdIt).first).rawId();
          
          if(std::find(m_maskedRPCIDs.begin(),m_maskedRPCIDs.end(),id) == m_maskedRPCIDs.end()){
              filteredDigis->put((*rpcLayerIdIt).second,(*rpcLayerIdIt).first);
          }
      }
    } 

 
      iEvent.put(std::move(filteredDigis));
}

// ------------ method called once each job just before starting event loop  ------------
void 
RPCChamberMasker::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
RPCChamberMasker::endJob() {
}

// ------------ method called when starting to processes a run  ------------

void
RPCChamberMasker::beginRun(edm::Run const& run, edm::EventSetup const& iSetup)
{
  edm::ESHandle<MuonSystemAging> mcData;
  iSetup.get<MuonSystemAgingRcd>().get(mcData);
  const MuonSystemAging* myMC=mcData.product();
  std::vector<int> mcV = myMC->m_RPCchambers;
  for(unsigned int i = 0; i < mcV.size();++i)m_maskedRPCIDs.push_back(mcV.at(i));

}

 
// ------------ method called when ending the processing of a run  ------------

void
RPCChamberMasker::endRun(edm::Run const&, edm::EventSetup const&)
{
}

 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
RPCChamberMasker::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
RPCChamberMasker::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
RPCChamberMasker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCChamberMasker);
