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

class GEMChamberMasker : public edm::EDProducer {
   public:
      explicit GEMChamberMasker(const edm::ParameterSet&);
      ~GEMChamberMasker();

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
  bool GE11Minus_;
  bool GE11Plus_; 
  bool GE21Minus_;
  bool GE21Plus_; 

  edm::EDGetTokenT<GEMDigiCollection> m_digiTag;
  std::vector<int> m_maskedGEMIDs;
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
/*
  std::cout<<"IDs of masked GEMs"<<std::endl;
  for ( auto gem_ids : iConfig.getParameter<std::vector<int>>("maskedGEMIDs"))
    {
      m_maskedGEMIDs.push_back(gem_ids);
      std::cout<<gem_ids<<std::endl;
    }
*/
  
}


GEMChamberMasker::~GEMChamberMasker()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GEMChamberMasker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
 // std::cout<<"***********new event*************"<<std::endl;
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
          //GEMDetId chambid = ((*gemLayerIdIt).first).chamberId();
          int id = ((*gemLayerIdIt).first).chamberId().rawId();
   //       std::cout<<"GEM det ID = "<<id<<std::endl; 
          if(std::find(m_maskedGEMIDs.begin(),m_maskedGEMIDs.end(),id) == m_maskedGEMIDs.end()){
     //         std::cout<<"Selected GEM det ID = "<<id<<std::endl;
              filteredDigis->put((*gemLayerIdIt).second,(*gemLayerIdIt).first);
          }
      }
    } 

 
      iEvent.put(std::move(filteredDigis));
}

// ------------ method called once each job just before starting event loop  ------------
void 
GEMChamberMasker::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
GEMChamberMasker::endJob() {
}

// ------------ method called when starting to processes a run  ------------

void
GEMChamberMasker::beginRun(edm::Run const& run, edm::EventSetup const& iSetup)
{
  edm::ESHandle<MuonSystemAging> mcData;
  iSetup.get<MuonSystemAgingRcd>().get(mcData);
  const MuonSystemAging* myMC=mcData.product();
  std::vector<int> mcV;
  if(GE11Minus_){
  mcV = myMC->m_GE11Minuschambers;
  for(unsigned int i = 0; i < mcV.size();++i)m_maskedGEMIDs.push_back(mcV.at(i));
  }

  if(GE11Plus_){
  mcV = myMC->m_GE11Pluschambers;
  for(unsigned int i = 0; i < mcV.size();++i)m_maskedGEMIDs.push_back(mcV.at(i));
  }

  if(GE21Minus_){
  mcV = myMC->m_GE21Minuschambers;
  for(unsigned int i = 0; i < mcV.size();++i)m_maskedGEMIDs.push_back(mcV.at(i));
  }

  if(GE21Plus_){
  mcV = myMC->m_GE21Pluschambers;
  for(unsigned int i = 0; i < mcV.size();++i)m_maskedGEMIDs.push_back(mcV.at(i));
  }
 for(unsigned int i = 0; i < m_maskedGEMIDs.size();++i)std::cout<<"chamber to be masked = "<<m_maskedGEMIDs.at(i)<<std::endl;
}

 
// ------------ method called when ending the processing of a run  ------------

void
GEMChamberMasker::endRun(edm::Run const&, edm::EventSetup const&)
{
}

 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
GEMChamberMasker::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
GEMChamberMasker::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
GEMChamberMasker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GEMChamberMasker);
