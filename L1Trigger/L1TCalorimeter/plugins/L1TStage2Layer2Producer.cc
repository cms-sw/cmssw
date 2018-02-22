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
#include <memory>

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

//
// class declaration
//

using namespace l1t;

 class L1TStage2Layer2Producer : public edm::stream::EDProducer<> {
  public:
    explicit L1TStage2Layer2Producer(const edm::ParameterSet& ps);
    ~L1TStage2Layer2Producer() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions)
      ;

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;

    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void endRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

    // ----------member data ---------------------------

    // input token
    edm::EDGetTokenT<CaloTowerBxCollection> m_towerToken;

    // put tokens
    edm::EDPutTokenT<CaloTowerBxCollection> m_towerMPToken;
    edm::EDPutTokenT<CaloClusterBxCollection> m_clusterMPToken;
    edm::EDPutTokenT<EGammaBxCollection> m_egammaMPToken;
    edm::EDPutTokenT<TauBxCollection> m_tauMPToken;
    edm::EDPutTokenT<JetBxCollection> m_jetMPToken;
    edm::EDPutTokenT<EtSumBxCollection> m_etMPToken;
    edm::EDPutTokenT<EGammaBxCollection> m_egammaToken;
    edm::EDPutTokenT<TauBxCollection> m_tauToken;
    edm::EDPutTokenT<JetBxCollection> m_jetToken;
    edm::EDPutTokenT<EtSumBxCollection> m_etToken;


    // parameters
    unsigned long long m_paramsCacheId;
    unsigned m_fwv;
    CaloParamsHelper* m_params;

    // the processor
    std::shared_ptr<Stage2MainProcessor> m_processor;

    // use static config for fw testing
    bool m_useStaticConfig;

  };



L1TStage2Layer2Producer::L1TStage2Layer2Producer(const edm::ParameterSet& ps) {

  // register what you produce
  m_towerMPToken = produces<CaloTowerBxCollection> ("MP");
  m_clusterMPToken = produces<CaloClusterBxCollection> ("MP");
  m_egammaMPToken = produces<EGammaBxCollection> ("MP");
  m_tauMPToken = produces<TauBxCollection> ("MP");
  m_jetMPToken = produces<JetBxCollection> ("MP");
  m_etMPToken = produces<EtSumBxCollection> ("MP");
  m_egammaToken = produces<EGammaBxCollection> ();
  m_tauToken = produces<TauBxCollection> ();
  m_jetToken = produces<JetBxCollection> ();
  m_etToken = produces<EtSumBxCollection> ();

  // register what you consume and keep token for later access:
  m_towerToken = consumes<CaloTowerBxCollection>(ps.getParameter<edm::InputTag>("towerToken"));

  // placeholder for the parameters
  m_params = new CaloParamsHelper;

  // set firmware version from python config for now
  m_fwv = ps.getParameter<int>("firmware");

  // get static config flag
  m_useStaticConfig = ps.getParameter<bool>("useStaticConfig");

  //initialize
  m_paramsCacheId=0;

}

L1TStage2Layer2Producer::~L1TStage2Layer2Producer() {

  delete m_params;

}

// ------------ method called to produce the data  ------------
void
L1TStage2Layer2Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;

  using namespace l1t;
  
  LogDebug("l1t|stage 2") << "L1TStage2Layer2Producer::produce function called..." << std::endl;


  //inputs
  Handle< BXVector<CaloTower> > towers;
  iEvent.getByToken(m_towerToken,towers);

  int bxFirst = towers->getFirstBX();
  int bxLast = towers->getLastBX();

  LogDebug("L1TDebug") << "First BX=" << bxFirst << ", last BX=" << bxLast << std::endl;

  //outputs
  CaloTowerBxCollection outTowers(0, bxFirst, bxLast);
  CaloClusterBxCollection clusters(0, bxFirst, bxLast);
  EGammaBxCollection mpegammas(0, bxFirst, bxLast);
  TauBxCollection mptaus(0, bxFirst, bxLast);
  JetBxCollection mpjets(0, bxFirst, bxLast);
  EtSumBxCollection mpsums(0, bxFirst, bxLast);
  EGammaBxCollection egammas(0, bxFirst, bxLast);
  TauBxCollection taus(0, bxFirst, bxLast);
  JetBxCollection jets(0, bxFirst, bxLast);
  EtSumBxCollection etsums(0, bxFirst, bxLast);

  // loop over BX
  for(int ibx = bxFirst; ibx < bxLast+1; ++ibx) {
    std::vector<CaloTower>  localTowers (CaloTools::caloTowerHashMax()+1);
    std::vector<CaloTower>  localOutTowers;
    std::vector<CaloCluster>  localClusters;
    std::vector<EGamma>  localMPEGammas;
    std::vector<Tau>  localMPTaus;
    std::vector<Jet>  localMPJets;
    std::vector<EtSum>  localMPEtSums;
    std::vector<EGamma>  localEGammas;
    std::vector<Tau>  localTaus;
    std::vector<Jet>  localJets;
    std::vector<EtSum>  localEtSums;

    LogDebug("L1TDebug") << "BX=" << ibx << ", N(Towers)=" << towers->size(ibx) << std::endl;

    for(std::vector<CaloTower>::const_iterator tower = towers->begin(ibx);
	tower != towers->end(ibx);
	++tower) {

      CaloTower tow(tower->p4(),
		    tower->etEm(),
		    tower->etHad(),
		    tower->hwPt(),
		    tower->hwEta(),
		    tower->hwPhi(),
		    tower->hwQual(),
		    tower->hwEtEm(),
		    tower->hwEtHad(),
		    tower->hwEtRatio());
      
      localTowers.at(CaloTools::caloTowerHash(tow.hwEta(),tow.hwPhi())) = tow;
    }

    LogDebug("L1TDebug") << "BX=" << ibx << ", N(Towers)=" << localTowers.size() << std::endl;

    m_processor->processEvent(localTowers,
			      localOutTowers,
			      localClusters,
			      localMPEGammas,
			      localMPTaus,
			      localMPJets,
			      localMPEtSums,
			      localEGammas,
			      localTaus,
			      localJets,
			      localEtSums);
    
    for( auto tow = localOutTowers.begin(); tow != localOutTowers.end(); ++tow)
      outTowers.push_back(ibx, *tow);
    for( auto clus = localClusters.begin(); clus != localClusters.end(); ++clus)
      clusters.push_back(ibx, *clus);
    for( auto eg = localMPEGammas.begin(); eg != localMPEGammas.end(); ++eg)
      mpegammas.push_back(ibx, CaloTools::egP4MP(*eg));
    for( auto tau = localMPTaus.begin(); tau != localMPTaus.end(); ++tau) 
      mptaus.push_back(ibx, CaloTools::tauP4MP(*tau));
    for( auto jet = localMPJets.begin(); jet != localMPJets.end(); ++jet) 
      mpjets.push_back(ibx, CaloTools::jetP4MP(*jet));
    for( auto etsum = localMPEtSums.begin(); etsum != localMPEtSums.end(); ++etsum)
      mpsums.push_back(ibx, CaloTools::etSumP4MP(*etsum));
    for( auto eg = localEGammas.begin(); eg != localEGammas.end(); ++eg)
      egammas.push_back(ibx, CaloTools::egP4Demux(*eg));
    for( auto tau = localTaus.begin(); tau != localTaus.end(); ++tau)
      taus.push_back(ibx, CaloTools::tauP4Demux(*tau));
    for( auto jet = localJets.begin(); jet != localJets.end(); ++jet)
      jets.push_back(ibx, CaloTools::jetP4Demux(*jet));
    for( auto etsum = localEtSums.begin(); etsum != localEtSums.end(); ++etsum) 
      etsums.push_back(ibx, CaloTools::etSumP4Demux(*etsum));

  
    LogDebug("L1TDebug") << "BX=" << ibx << ", N(Cluster)=" << localClusters.size() << ", N(EG)=" << localEGammas.size() << ", N(Tau)=" << localTaus.size() << ", N(Jet)=" << localJets.size() << ", N(Sums)=" << localEtSums.size() << std::endl;

  }

  iEvent.emplace(m_towerMPToken, std::move(outTowers));
  iEvent.emplace(m_clusterMPToken, std::move(clusters));
  iEvent.emplace(m_egammaMPToken, std::move(mpegammas));
  iEvent.emplace(m_tauMPToken, std::move(mptaus));
  iEvent.emplace(m_jetMPToken, std::move(mpjets));
  iEvent.emplace(m_etMPToken, std::move(mpsums));
  iEvent.emplace(m_egammaToken, std::move(egammas));
  iEvent.emplace(m_tauToken, std::move(taus));
  iEvent.emplace(m_jetToken, std::move(jets));
  iEvent.emplace(m_etToken, std::move(etsums));

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

    // fetch payload corresponding to the current run from the CondDB
    edm::ESHandle<CaloParams> candidateHandle;
    iSetup.get<L1TCaloParamsRcd>().get(candidateHandle);
    std::unique_ptr<l1t::CaloParams> candidate(new l1t::CaloParams( *candidateHandle.product() ));


    if(!m_useStaticConfig){

      // fetch the latest greatest prototype (equivalent of static payload)
      edm::ESHandle<CaloParams> o2oProtoHandle;
      iSetup.get<L1TCaloParamsO2ORcd>().get(o2oProtoHandle);
      std::unique_ptr<l1t::CaloParams> prototype(new l1t::CaloParams( *o2oProtoHandle.product() ));

      // prepare to set the emulator's configuration
      //  and then replace our local copy of the parameters with a new one using placement new
      m_params->~CaloParamsHelper();

      // compare the candidate payload misses some of the pnodes compared to the prototype,
      // if this is the case - the candidate is an old payload that'll crash the Stage2 emulator
      // and we better use the prototype for the emulator's configuration
      if( ((CaloParamsHelper*)candidate.get())->getNodes().size() < ((CaloParamsHelper*)prototype.get())->getNodes().size())
	m_params = new (m_params) CaloParamsHelper( *o2oProtoHandle.product() );
      else
	m_params = new (m_params) CaloParamsHelper( *candidateHandle.product() );
      // KK: the nifty tricks above (placement new) work as long as current definition of
      //     CaloParams takes more space than the one obtained from the record
      
    } else {
      m_params->~CaloParamsHelper();
      m_params = new (m_params) CaloParamsHelper( *candidateHandle.product() );
    }

    LogDebug("L1TDebug") << *m_params << std::endl;

    if (! m_params){
      edm::LogError("l1t|caloStage2") << "Could not retrieve params from Event Setup" << std::endl;
    }

  }

  // firmware

  if ( !m_processor ) { // in future, also check if the firmware cache ID has changed !

    //     m_fwv = ; // get new firmware version in future

    // Set the current algorithm version based on DB pars from database:
    Stage2Layer2FirmwareFactory m_factory;
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
