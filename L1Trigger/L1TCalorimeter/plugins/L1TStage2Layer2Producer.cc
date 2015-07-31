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

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

//
// class declaration
//

using namespace l1t;

  class L1TStage2Layer2Producer : public edm::EDProducer {
  public:
    explicit L1TStage2Layer2Producer(const edm::ParameterSet& ps);
    ~L1TStage2Layer2Producer();

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
    CaloParamsHelper* m_params;

    // the processor
    Stage2Layer2FirmwareFactory m_factory;
    boost::shared_ptr<Stage2MainProcessor> m_processor;

  };



L1TStage2Layer2Producer::L1TStage2Layer2Producer(const edm::ParameterSet& ps) {

  // register what you produce
  produces<CaloTowerBxCollection> ("MP");
  produces<CaloClusterBxCollection> ("MP");
  produces<EGammaBxCollection> ("MP");
  produces<TauBxCollection> ("MP");
  produces<JetBxCollection> ("MP");
  produces<EtSumBxCollection> ("MP");
  produces<EGammaBxCollection> ();
  produces<TauBxCollection> ();
  produces<JetBxCollection> ();
  produces<EtSumBxCollection> ();

  // register what you consume and keep token for later access:
  m_towerToken = consumes<CaloTowerBxCollection>(ps.getParameter<edm::InputTag>("towerToken"));

  // placeholder for the parameters
  m_params = new CaloParamsHelper;

  // set firmware version from python config for now
  m_fwv = ps.getParameter<int>("firmware");

}

L1TStage2Layer2Producer::~L1TStage2Layer2Producer() {

  delete m_params;

}

// ------------ method called to produce the data  ------------
void
L1TStage2Layer2Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  LogDebug("l1t|stage 2") << "L1TStage2Layer2Producer::produce function called..." << std::endl;


  //inputs
  Handle< BXVector<CaloTower> > towers;
  iEvent.getByToken(m_towerToken,towers);

  int bxFirst = towers->getFirstBX();
  int bxLast = towers->getLastBX();

  LogDebug("L1TDebug") << "First BX=" << bxFirst << ", last BX=" << bxLast << std::endl;

  //outputs
  std::auto_ptr<CaloTowerBxCollection> outTowers (new CaloTowerBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<CaloClusterBxCollection> clusters (new CaloClusterBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<EGammaBxCollection> mpegammas (new EGammaBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<TauBxCollection> mptaus (new TauBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<JetBxCollection> mpjets (new JetBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<EtSumBxCollection> mpsums (new EtSumBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<EGammaBxCollection> egammas (new EGammaBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<TauBxCollection> taus (new TauBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<JetBxCollection> jets (new JetBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<EtSumBxCollection> etsums (new EtSumBxCollection(0, bxFirst, bxLast));

  // loop over BX
  for(int ibx = bxFirst; ibx < bxLast+1; ++ibx) {
    std::auto_ptr< std::vector<CaloTower> > localTowers (new std::vector<CaloTower>);
    std::auto_ptr< std::vector<CaloTower> > localOutTowers (new std::vector<CaloTower>);
    std::auto_ptr< std::vector<CaloCluster> > localClusters (new std::vector<CaloCluster>);
    std::auto_ptr< std::vector<EGamma> > localMPEGammas (new std::vector<EGamma>);
    std::auto_ptr< std::vector<Tau> > localMPTaus (new std::vector<Tau>);
    std::auto_ptr< std::vector<Jet> > localMPJets (new std::vector<Jet>);
    std::auto_ptr< std::vector<EtSum> > localMPEtSums (new std::vector<EtSum>);
    std::auto_ptr< std::vector<EGamma> > localEGammas (new std::vector<EGamma>);
    std::auto_ptr< std::vector<Tau> > localTaus (new std::vector<Tau>);
    std::auto_ptr< std::vector<Jet> > localJets (new std::vector<Jet>);
    std::auto_ptr< std::vector<EtSum> > localEtSums (new std::vector<EtSum>);

    LogDebug("L1TDebug") << "BX=" << ibx << ", N(Towers)=" << towers->size(ibx) << std::endl;

    for(std::vector<CaloTower>::const_iterator tower = towers->begin(ibx);
	tower != towers->end(ibx);
	++tower) {
      localTowers->push_back(*tower);
    }

    LogDebug("L1TDebug") << "BX=" << ibx << ", N(Towers)=" << localTowers->size() << std::endl;

    m_processor->processEvent(*localTowers,
			      *localOutTowers,
			      *localClusters,
			      *localMPEGammas,
			      *localMPTaus,
			      *localMPJets,
			      *localMPEtSums,
			      *localEGammas,
			      *localTaus,
			      *localJets,
			      *localEtSums);

    for(std::vector<CaloTower>::const_iterator tow = localOutTowers->begin(); tow != localOutTowers->end(); ++tow) outTowers->push_back(ibx, *tow);
    for(std::vector<CaloCluster>::const_iterator clus = localClusters->begin(); clus != localClusters->end(); ++clus) clusters->push_back(ibx, *clus);
    for(std::vector<EGamma>::const_iterator eg = localMPEGammas->begin(); eg != localMPEGammas->end(); ++eg) mpegammas->push_back(ibx, *eg);
    for(std::vector<Tau>::const_iterator tau = localMPTaus->begin(); tau != localMPTaus->end(); ++tau) mptaus->push_back(ibx, *tau);
    for(std::vector<Jet>::const_iterator jet = localMPJets->begin(); jet != localMPJets->end(); ++jet) mpjets->push_back(ibx, *jet);
    for(std::vector<EtSum>::const_iterator etsum = localMPEtSums->begin(); etsum != localMPEtSums->end(); ++etsum) mpsums->push_back(ibx, *etsum);
    for(std::vector<EGamma>::const_iterator eg = localEGammas->begin(); eg != localEGammas->end(); ++eg) egammas->push_back(ibx, *eg);
    for(std::vector<Tau>::const_iterator tau = localTaus->begin(); tau != localTaus->end(); ++tau) taus->push_back(ibx, *tau);
    for(std::vector<Jet>::const_iterator jet = localJets->begin(); jet != localJets->end(); ++jet) jets->push_back(ibx, *jet);
    for(std::vector<EtSum>::const_iterator etsum = localEtSums->begin(); etsum != localEtSums->end(); ++etsum) etsums->push_back(ibx, *etsum);


    LogDebug("L1TDebug") << "BX=" << ibx << ", N(Cluster)=" << localClusters->size() << ", N(EG)=" << localEGammas->size() << ", N(Tau)=" << localTaus->size() << ", N(Jet)=" << localJets->size() << ", N(Sums)=" << localEtSums->size() << std::endl;

  }

  iEvent.put(outTowers, "MP");
  iEvent.put(clusters, "MP");
  iEvent.put(mpegammas, "MP");
  iEvent.put(mptaus, "MP");
  iEvent.put(mpjets, "MP");
  iEvent.put(mpsums, "MP");
  iEvent.put(egammas);
  iEvent.put(taus);
  iEvent.put(jets);
  iEvent.put(etsums);

}

// ------------ method called once each job just before starting event loop  ------------
void
L1TStage2Layer2Producer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TStage2Layer2Producer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TStage2Layer2Producer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
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
    m_params->~CaloParamsHelper();
    m_params = new (m_params) CaloParamsHelper(*paramsHandle.product());

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
L1TStage2Layer2Producer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TStage2Layer2Producer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup cons
t&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TStage2Layer2Producer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&
)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TStage2Layer2Producer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TStage2Layer2Producer);
