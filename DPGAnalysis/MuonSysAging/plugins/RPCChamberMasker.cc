// -*- C++ -*-
//
// Package:    DPGAnalysis/MuonSysAging/
// Class:      RPCChamberMasker
// 
/**\class RPCChamberMasker RPCChamberMasker.cc DPGAnalysis/MuonSysAging/plugins/RPCChamberMasker.cc

 Description:

 Implementation:
     Class to mask RPC digis on a for single DetIds

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
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandomEngine.h"
//
// class declaration
//

class RPCChamberMasker : public edm::EDProducer 
{

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

      // ----------member data ---------------------------
  edm::InputTag digiTag_;
  edm::EDGetTokenT<RPCDigiCollection> m_digiTag;
  std::map<RPCDetId, float> m_ChEffs;
  bool theRE31_off;
  bool theRE41_off; 

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
  produces<RPCDigiCollection>();

  theRE31_off = iConfig.getParameter<bool>("descopeRE31");
  theRE41_off =iConfig.getParameter<bool>("descopeRE41");

}


RPCChamberMasker::~RPCChamberMasker()
{

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
RPCChamberMasker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  edm::Service<edm::RandomNumberGenerator> randGenService;
  CLHEP::HepRandomEngine& randGen = randGenService->getEngine(iEvent.streamID());

  
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
	  auto chEffIt = m_ChEffs.find(id);
	  if ((chEffIt != m_ChEffs.end()) && (randGen.flat() <= chEffIt->second))
	    filteredDigis->put((*rpcLayerIdIt).second,(*rpcLayerIdIt).first);
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
RPCChamberMasker::endJob() 
{

}

// ------------ method called when starting to processes a run  ------------

void
RPCChamberMasker::beginRun(edm::Run const& run, edm::EventSetup const& iSetup)
{

  m_ChEffs.clear();
  
  edm::ESHandle<RPCGeometry> rpcGeom;
  iSetup.get<MuonGeometryRecord>().get(rpcGeom);
  
  edm::ESHandle<MuonSystemAging> agingObj;
  iSetup.get<MuonSystemAgingRcd>().get(agingObj);
  
  const auto rolls = rpcGeom->rolls();
  
  for ( const auto * roll : rolls)
    {
      RPCDetId rollId = roll->id();
      uint32_t rollRawId = rollId.rawId();
      
      Float_t chamberEff = 1.; 
      for ( auto & agingPair : agingObj->m_RPCChambEffs)
	{
	  
	  if ( agingPair.first == rollRawId)
	    {
	      chamberEff = agingPair.second;
	      break;
	    }
	  
	  if(theRE31_off && ( roll->isIRPC() && (rollId.station()==3) ) ){
	    chamberEff = 0;
	  }
	  if(theRE41_off && ( roll->isIRPC() && (rollId.station()==4) ) ){
	    chamberEff = 0;
	  }	   
	}
      m_ChEffs[rollId] = chamberEff;
    }

}

 
// ------------ method called when ending the processing of a run  ------------

void
RPCChamberMasker::endRun(edm::Run const&, edm::EventSetup const&)
{

}
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
RPCChamberMasker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("simMuonRPCDigis"));
  desc.add<bool>("descopeRE31", false);
  desc.add<bool>("descopeRE41", false);
  descriptions.addDefault(desc);

}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCChamberMasker);
