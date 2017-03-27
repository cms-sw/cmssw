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

class ME0ChamberMasker : public edm::EDProducer {
   public:
      explicit ME0ChamberMasker(const edm::ParameterSet&);
      ~ME0ChamberMasker();

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
  bool ME0Minus_;
  bool ME0Plus_;
  edm::InputTag digiTag_;
  edm::EDGetTokenT<ME0DigiPreRecoCollection> m_digiTag;
  std::vector<int> m_maskedME0IDs;
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
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

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
           int id = ((*me0LayerIdIt).first).chamberId().rawId();
                   if(std::find(m_maskedME0IDs.begin(),m_maskedME0IDs.end(),id) == m_maskedME0IDs.end()){
                          filteredDigis->put((*me0LayerIdIt).second,(*me0LayerIdIt).first);
             }
          
     }
    } 

 
      iEvent.put(std::move(filteredDigis));
}

// ------------ method called once each job just before starting event loop  ------------
void 
ME0ChamberMasker::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
ME0ChamberMasker::endJob() {
}

// ------------ method called when starting to processes a run  ------------

void
ME0ChamberMasker::beginRun(edm::Run const& run, edm::EventSetup const& iSetup)
{
  edm::ESHandle<MuonSystemAging> mcData;
  iSetup.get<MuonSystemAgingRcd>().get(mcData);
  const MuonSystemAging* myMC=mcData.product();
  std::vector<int> mcV;
  if(ME0Minus_){
  mcV = myMC->m_ME0Minuschambers;
  for(unsigned int i = 0; i < mcV.size();++i)m_maskedME0IDs.push_back(mcV.at(i));
  }

  if(ME0Plus_){
  mcV = myMC->m_ME0Pluschambers;
  for(unsigned int i = 0; i < mcV.size();++i)m_maskedME0IDs.push_back(mcV.at(i));
  }
}

 
// ------------ method called when ending the processing of a run  ------------

void
ME0ChamberMasker::endRun(edm::Run const&, edm::EventSetup const&)
{
}

 
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ME0ChamberMasker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ME0ChamberMasker);
