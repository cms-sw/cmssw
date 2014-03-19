// -*- C++ -*-
//
// Package:    L1Trigger/skeleton
// Class:      skeleton
// 
/**\class skeleton skeleton.cc L1Trigger/skeleton/plugins/skeleton.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  James Brooke
//         Created:  Thu, 05 Dec 2013 17:39:27 GMT
//
//


// system include files
#include <boost/shared_ptr.hpp>

// user include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer1FirmwareFactory.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2PreProcessor.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

//
// class declaration
//

namespace l1t {
  
  class Stage2Layer1Producer : public edm::EDProducer { 
  public:
    explicit Stage2Layer1Producer(const edm::ParameterSet& ps);
    ~Stage2Layer1Producer();
    
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions)
      ;
    
  private:
    virtual void beginJob() override;
    virtual void produce(edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;
    
    virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    
    // ----------member data ---------------------------

    // input token
    edm::EDGetToken m_towerToken;

    // parameters
    unsigned long long m_paramsCacheId;
    FirmwareVersion m_fwv;
    CaloParams* m_params;

    // the processor
    l1t::Stage2Layer1FirmwareFactory m_factory;
    boost::shared_ptr<Stage2PreProcessor> m_processor;
     
  }; 
  
}

l1t::Stage2Layer1Producer::Stage2Layer1Producer(const edm::ParameterSet& ps) {

  // register what you produce
  produces<l1t::CaloTowerBxCollection> ();
  
  // register what you consume and keep token for later access:
  m_towerToken = consumes<l1t::CaloTowerBxCollection>(ps.getParameter<edm::InputTag>("towerToken"));
  
  // placeholder for the parameters
  m_params = new CaloParams;

  // set firmware version from python config for now
  m_fwv.setFirmwareVersion(ps.getParameter<int>("firmware"));
 
}

l1t::Stage2Layer1Producer::~Stage2Layer1Producer() {
  
  delete m_params;

}

// ------------ method called to produce the data  ------------
void
l1t::Stage2Layer1Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  LogDebug("l1t|stage 2") << "Stage2Layer1Producer::produce function called..." << std::endl;
  
  
  //inputs
  Handle< BXVector<l1t::CaloTower> > inTowers;
  iEvent.getByToken(m_towerToken,inTowers);
  
  int bxFirst = inTowers->getFirstBX();
  int bxLast  = inTowers->getLastBX();

  LogDebug("L1TDebug") << "First BX=" << bxFirst << ", last BX=" << bxLast << std::endl;
  
  //outputs
  std::auto_ptr<l1t::CaloTowerBxCollection> outTowers (new l1t::CaloTowerBxCollection(0, bxFirst, bxLast));
  
  // loop over BX
  for(int ibx = bxFirst; ibx < bxLast+1; ++ibx) {
    std::auto_ptr< std::vector<l1t::CaloTower> > localInTowers (new std::vector<l1t::CaloTower>);
    std::auto_ptr< std::vector<l1t::CaloTower> > localOutTowers (new std::vector<l1t::CaloTower>);
    
    for(std::vector<l1t::CaloTower>::const_iterator tower = inTowers->begin(ibx);
	tower != inTowers->end(ibx);
	++tower)
      localInTowers->push_back(*tower);

    LogDebug("L1TDebug") << "BX=" << ibx << ", N(Towers)=" << inTowers->size(ibx) << "=" << localInTowers->size() << std::endl;    

    m_processor->processEvent(*localInTowers, *localOutTowers);
    
    for(std::vector<l1t::CaloTower>::const_iterator tower = localOutTowers->begin(); 
	tower != localOutTowers->end(); 
	++tower) 
      outTowers->push_back(ibx, *tower);

    LogDebug("L1TDebug") << "BX=" << ibx << ", N(Tower in)=" << localInTowers->size() << ", N(Tower out)=" << localOutTowers->size() << std::endl;

  }
  
  iEvent.put(outTowers);
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
l1t::Stage2Layer1Producer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
l1t::Stage2Layer1Producer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
l1t::Stage2Layer1Producer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{

  // update parameters and algorithms at run start, if they have changed
  // update params first because the firmware factory relies on pointer to params

  // parameters

  unsigned long long id = iSetup.get<L1TCaloParamsRcd>().cacheIdentifier();  
  
  if (id != m_paramsCacheId) {

    m_paramsCacheId = id;

    edm::ESHandle<CaloParams> paramsHandle;
    iSetup.get<L1TCaloParamsRcd>().get(paramsHandle);

    // replace our local copy of the parameters with a new one using placement new
    m_params->~CaloParams();
    m_params = new (m_params) CaloParams(*paramsHandle.product());
    
    LogDebug("L1TDebug") << *m_params << std::endl;

    if (! m_params){
      edm::LogError("l1t|caloStage2") << "Could not retrieve params from Event Setup" << std::endl;            
    }

  }

  // firmware

  if ( !m_processor ) { // in future, also check if the firmware cache ID has changed !
    
    //     m_fwv = ; // get new firmware version in future
    
    // Set the current algorithm version based on DB pars from database:
    m_processor = m_factory.create(m_fwv, m_params);
    
    if (! m_processor) {
      // we complain here once per run
      edm::LogError("l1t|caloStage2") << "Layer 1 firmware could not be configured.\n";
    }
    
    LogDebug("L1TDebug") << "Processor object : " << (m_processor?1:0) << std::endl;
    
  }
  

}


// ------------ method called when ending the processing of a run  ------------
void
l1t::Stage2Layer1Producer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
l1t::Stage2Layer1Producer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup cons
t&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
l1t::Stage2Layer1Producer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&
)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
l1t::Stage2Layer1Producer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::Stage2Layer1Producer);
