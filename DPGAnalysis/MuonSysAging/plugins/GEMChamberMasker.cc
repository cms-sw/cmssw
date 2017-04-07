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

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "CondFormats/MuonSystemAging/interface/MuonSystemAging.h"
#include "CondFormats/DataRecord/interface/MuonSystemAgingRcd.h"

//
// class declaration
//

class GEMChamberMasker : public edm::stream::EDProducer<>
{

   public:
      explicit GEMChamberMasker(const edm::ParameterSet&);
      ~GEMChamberMasker();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      
      virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void endRun(edm::Run const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
  edm::InputTag digiTag_;
  bool GE11Minus_;
  bool GE11Plus_; 
  bool GE21Minus_;
  bool GE21Plus_; 

  edm::EDGetTokenT<GEMDigiCollection> m_digiTag;
  std::map<unsigned int, float> m_maskedGEMIDs;

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
GEMChamberMasker::GEMChamberMasker(const edm::ParameterSet& iConfig) : 
  digiTag_(iConfig.getParameter<edm::InputTag>("digiTag") ),
  GE11Minus_(iConfig.getParameter<bool>("GE11Minus") ),
  GE11Plus_(iConfig.getParameter<bool>("GE11Plus") ),
  GE21Minus_(iConfig.getParameter<bool>("GE21Minus") ),
  GE21Plus_(iConfig.getParameter<bool>("GE21Plus") )
{

  m_digiTag = consumes<GEMDigiCollection>(digiTag_); 
  produces<GEMDigiCollection>();
  
}


GEMChamberMasker::~GEMChamberMasker()
{ 

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GEMChamberMasker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  std::unique_ptr<GEMDigiCollection> filteredDigis(new GEMDigiCollection());

  if (!digiTag_.label().empty())
    {
      edm::Handle<GEMDigiCollection> gemDigis;
      iEvent.getByToken(m_digiTag, gemDigis);
      
      GEMDigiCollection::DigiRangeIterator gemLayerIdIt  = gemDigis->begin();
      GEMDigiCollection::DigiRangeIterator gemLayerIdEnd = gemDigis->end();


      
      for (; gemLayerIdIt != gemLayerIdEnd; ++gemLayerIdIt)
	{
	  auto chambId = (*gemLayerIdIt).first.chamberId();

	  bool keepDigi = (!GE11Minus_  && chambId.station()==1 && chambId.region()<0 ) ||
		          (!GE11Plus_   && chambId.station()==1 && chambId.region()>0 ) ||
		          (!GE21Minus_  && chambId.station()==2 && chambId.region()<0 ) ||
		          (!GE21Plus_   && chambId.station()==2 && chambId.region()>0 ) ;

          uint32_t rawId = chambId.rawId();
          if(keepDigi || m_maskedGEMIDs.find(rawId) == m_maskedGEMIDs.end())
	    {
              filteredDigis->put((*gemLayerIdIt).second,(*gemLayerIdIt).first);
	    }
	}

    } 

  iEvent.put(std::move(filteredDigis));

}

// ------------ method called when starting to processes a run  ------------

void
GEMChamberMasker::beginRun(edm::Run const& run, edm::EventSetup const& iSetup)
{

  edm::ESHandle<MuonSystemAging> agingObj;
  iSetup.get<MuonSystemAgingRcd>().get(agingObj);

  m_maskedGEMIDs = agingObj->m_GEMChambEffs;

}

 
// ------------ method called when ending the processing of a run  ------------

void
GEMChamberMasker::endRun(edm::Run const&, edm::EventSetup const&)
{

}

void
GEMChamberMasker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("simMuonGEMDigis"));
  desc.add<bool>("GE11Minus", true);
  desc.add<bool>("GE11Plus",  true);
  desc.add<bool>("GE21Minus", true);
  desc.add<bool>("GE21Plus",  true);

  descriptions.addDefault(desc);

}

//define this as a plug-in
DEFINE_FWK_MODULE(GEMChamberMasker);
