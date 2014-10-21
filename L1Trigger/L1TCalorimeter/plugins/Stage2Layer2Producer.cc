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

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2FirmwareFactory.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2MainProcessor.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

//
// class declaration
//

namespace l1t {
  
  class Stage2Layer2Producer : public edm::EDProducer { 
  public:
    explicit Stage2Layer2Producer(const edm::ParameterSet& ps);
    ~Stage2Layer2Producer();
    
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
    unsigned m_fwv;
    CaloParams* m_params;

    // the processor
    l1t::Stage2Layer2FirmwareFactory m_factory;
    boost::shared_ptr<Stage2MainProcessor> m_processor;
     
  }; 
  
}

l1t::Stage2Layer2Producer::Stage2Layer2Producer(const edm::ParameterSet& ps) {

  // register what you produce
  produces<l1t::CaloTowerBxCollection> ();
  produces<l1t::CaloClusterBxCollection> ();
  produces<l1t::EGammaBxCollection> ();
  produces<l1t::TauBxCollection> ();
  produces<l1t::JetBxCollection> ();
  produces<l1t::EtSumBxCollection> ();
  
  // register what you consume and keep token for later access:
  m_towerToken = consumes<l1t::CaloTowerBxCollection>(ps.getParameter<edm::InputTag>("towerToken"));
  
  // placeholder for the parameters
  m_params = new CaloParams;

  // set firmware version from python config for now
  m_fwv = ps.getParameter<int>("firmware");
 
}

l1t::Stage2Layer2Producer::~Stage2Layer2Producer() {
  
  delete m_params;

}

// ------------ method called to produce the data  ------------
void
l1t::Stage2Layer2Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  LogDebug("l1t|stage 2") << "Stage2Layer2Producer::produce function called..." << std::endl;
  
  
  //inputs
  Handle< BXVector<l1t::CaloTower> > towers;
  iEvent.getByToken(m_towerToken,towers);
  
  int bxFirst = towers->getFirstBX();
  int bxLast = towers->getLastBX();

  LogDebug("L1TDebug") << "First BX=" << bxFirst << ", last BX=" << bxLast << std::endl;
  
  //outputs
  std::auto_ptr<l1t::CaloTowerBxCollection> outTowers (new l1t::CaloTowerBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::CaloClusterBxCollection> clusters (new l1t::CaloClusterBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::EGammaBxCollection> egammas (new l1t::EGammaBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::TauBxCollection> taus (new l1t::TauBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::JetBxCollection> jets (new l1t::JetBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::EtSumBxCollection> etsums (new l1t::EtSumBxCollection(0, bxFirst, bxLast));
  
  // loop over BX
  for(int ibx = bxFirst; ibx < bxLast+1; ++ibx) {
    std::auto_ptr< std::vector<l1t::CaloTower> > localTowers (new std::vector<l1t::CaloTower>);
    std::auto_ptr< std::vector<l1t::CaloTower> > localOutTowers (new std::vector<l1t::CaloTower>);
    std::auto_ptr< std::vector<l1t::CaloCluster> > localClusters (new std::vector<l1t::CaloCluster>);
    std::auto_ptr< std::vector<l1t::EGamma> > localEGammas (new std::vector<l1t::EGamma>);
    std::auto_ptr< std::vector<l1t::Tau> > localTaus (new std::vector<l1t::Tau>);
    std::auto_ptr< std::vector<l1t::Jet> > localJets (new std::vector<l1t::Jet>);
    std::auto_ptr< std::vector<l1t::EtSum> > localEtSums (new std::vector<l1t::EtSum>);
    
    LogDebug("L1TDebug") << "BX=" << ibx << ", N(Towers)=" << towers->size(ibx) << std::endl;

    for(std::vector<l1t::CaloTower>::const_iterator tower = towers->begin(ibx);
	tower != towers->end(ibx);
	++tower) {
      localTowers->push_back(*tower);
    }

    LogDebug("L1TDebug") << "BX=" << ibx << ", N(Towers)=" << localTowers->size() << std::endl;    

    m_processor->processEvent(*localTowers,
			      *localOutTowers, *localClusters, *localEGammas, *localTaus, *localJets, *localEtSums);
    
    for(std::vector<l1t::CaloTower>::const_iterator tow = localOutTowers->begin(); tow != localOutTowers->end(); ++tow) outTowers->push_back(ibx, *tow);
    for(std::vector<l1t::CaloCluster>::const_iterator clus = localClusters->begin(); clus != localClusters->end(); ++clus) clusters->push_back(ibx, *clus);
    for(std::vector<l1t::EGamma>::const_iterator eg = localEGammas->begin(); eg != localEGammas->end(); ++eg) egammas->push_back(ibx, *eg);
    for(std::vector<l1t::Tau>::const_iterator tau = localTaus->begin(); tau != localTaus->end(); ++tau) taus->push_back(ibx, *tau);
    for(std::vector<l1t::Jet>::const_iterator jet = localJets->begin(); jet != localJets->end(); ++jet) jets->push_back(ibx, *jet);
    for(std::vector<l1t::EtSum>::const_iterator etsum = localEtSums->begin(); etsum != localEtSums->end(); ++etsum) etsums->push_back(ibx, *etsum);

    LogDebug("L1TDebug") << "BX=" << ibx << ", N(Cluster)=" << localClusters->size() << ", N(EG)=" << localEGammas->size() << ", N(Tau)=" << localTaus->size() << ", N(Jet)=" << localJets->size() << ", N(Sums)=" << localEtSums->size() << std::endl;    

  }
  
  iEvent.put(outTowers);
  iEvent.put(clusters);
  iEvent.put(egammas);
  iEvent.put(taus);
  iEvent.put(jets);
  iEvent.put(etsums);
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
l1t::Stage2Layer2Producer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
l1t::Stage2Layer2Producer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
l1t::Stage2Layer2Producer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
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
      edm::LogError("l1t|caloStage2") << "Firmware could not be configured.\n";
    }
    
    LogDebug("L1TDebug") << "Processor object : " << (m_processor?1:0) << std::endl;
    
  }
  

}


// ------------ method called when ending the processing of a run  ------------
void
l1t::Stage2Layer2Producer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
l1t::Stage2Layer2Producer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup cons
t&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
l1t::Stage2Layer2Producer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&
)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
l1t::Stage2Layer2Producer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::Stage2Layer2Producer);
