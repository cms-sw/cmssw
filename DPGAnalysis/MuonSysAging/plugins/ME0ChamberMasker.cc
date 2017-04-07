// -*- C++ -*-
// Class:      ME0ChamberMasker
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

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"
#include "CondFormats/MuonSystemAging/interface/MuonSystemAging.h"
#include "CondFormats/DataRecord/interface/MuonSystemAgingRcd.h"
//
// class declaration
//

class ME0ChamberMasker : public edm::stream::EDProducer<>
{

   public:
      explicit ME0ChamberMasker(const edm::ParameterSet&);
      ~ME0ChamberMasker();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      
      virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void endRun(edm::Run const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
  bool ME0Minus_;
  bool ME0Plus_;
  edm::InputTag digiTag_;
  edm::EDGetTokenT<ME0DigiPreRecoCollection> m_digiTag;
  std::map<unsigned int, float> m_maskedME0IDs;

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
ME0ChamberMasker::ME0ChamberMasker(const edm::ParameterSet& iConfig) : 
 ME0Minus_(iConfig.getParameter<bool>("ME0Minus") ),
 ME0Plus_(iConfig.getParameter<bool>("ME0Plus") ), 
 digiTag_(iConfig.getParameter<edm::InputTag>("digiTag") )
{

  m_digiTag = consumes<ME0DigiPreRecoCollection>(digiTag_); 
  produces<ME0DigiPreRecoCollection>();

}


ME0ChamberMasker::~ME0ChamberMasker()
{

}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
ME0ChamberMasker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  std::unique_ptr<ME0DigiPreRecoCollection> filteredDigis(new ME0DigiPreRecoCollection());

  if (!digiTag_.label().empty())
    {
      edm::Handle<ME0DigiPreRecoCollection> me0Digis;
      iEvent.getByToken(m_digiTag, me0Digis);
      
      ME0DigiPreRecoCollection::DigiRangeIterator me0LayerIdIt  = me0Digis->begin();
      ME0DigiPreRecoCollection::DigiRangeIterator me0LayerIdEnd = me0Digis->end();
      
      for (; me0LayerIdIt != me0LayerIdEnd; ++me0LayerIdIt)
	{

	  auto chambId = (*me0LayerIdIt).first.chamberId();

	  bool keepDigi = (!ME0Minus_  && chambId.region() < 0 ) || 
	                  (!ME0Plus_   && chambId.region() > 0 ) ;
	  
	  uint32_t rawId = chambId.rawId();
	  if(keepDigi || m_maskedME0IDs.find(rawId) == m_maskedME0IDs.end())
	    {
	      filteredDigis->put((*me0LayerIdIt).second,(*me0LayerIdIt).first);
	    }
          
	}
    } 
  
  iEvent.put(std::move(filteredDigis));

}

// ------------ method called when starting to processes a run  ------------

void
ME0ChamberMasker::beginRun(edm::Run const& run, edm::EventSetup const& iSetup)
{

  edm::ESHandle<MuonSystemAging> agingObj;
  iSetup.get<MuonSystemAgingRcd>().get(agingObj);

  m_maskedME0IDs = agingObj->m_ME0ChambEffs;

}

 
// ------------ method called when ending the processing of a run  ------------

void
ME0ChamberMasker::endRun(edm::Run const&, edm::EventSetup const&)
{

}

 
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ME0ChamberMasker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("simMuonME0Digis"));
  desc.add<bool>("ME0Minus", true);
  desc.add<bool>("ME0Plus",  true);
  descriptions.addDefault(desc);

}

//define this as a plug-in
DEFINE_FWK_MODULE(ME0ChamberMasker);
