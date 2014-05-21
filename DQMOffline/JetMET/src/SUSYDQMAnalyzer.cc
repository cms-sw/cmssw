//authors:  Francesco Costanza (DESY)
//          Dirk Kruecker (DESY)
//date:     05/05/11

//================================================================  
// CMS FW and Data Handlers

#include "DataFormats/Common/interface/Handle.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//================================================================  
// Jet & Jet collections  // MET & MET collections

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
//#include "DataFormats/JetReco/interface/JPTJet.h"
//#include "DataFormats/JetReco/interface/JPTJetCollection.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"

//================================================================  
// SUSY Classes

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DQMOffline/JetMET/interface/SUSYDQMAnalyzer.h"
#include "DQMOffline/JetMET/interface/SusyDQM/alpha_T.h"
#include "DQMOffline/JetMET/interface/SusyDQM/HT.h"

//================================================================  
// ROOT Classes

#include "TH1.h"
#include "TVector2.h"
#include "TLorentzVector.h"

//================================================================  
// Ordinary C++ stuff

#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

using namespace edm;
using namespace reco;
using namespace math;
using namespace std;

SUSYDQMAnalyzer::SUSYDQMAnalyzer( const edm::ParameterSet& pSet)
{
  iConfig = pSet;
  
  SUSYFolder = iConfig.getParameter<std::string>("folderName");
  // Load parameters 
  thePFMETCollectionToken     = consumes<reco::PFMETCollection>   (iConfig.getParameter<edm::InputTag>("PFMETCollectionLabel"));
  theCaloMETCollectionToken   = consumes<reco::CaloMETCollection> (iConfig.getParameter<edm::InputTag>("CaloMETCollectionLabel"));

  //remove TCMET and JPT related variables due to anticipated changes in RECO
  //theTCMETCollectionToken     = consumes<reco::METCollection>     (iConfig.getParameter<edm::InputTag>("TCMETCollectionLabel"));

  theCaloJetCollectionToken   = consumes<reco::CaloJetCollection>   (iConfig.getParameter<edm::InputTag>("CaloJetCollectionLabel"));
  //theJPTJetCollectionToken    = consumes<reco::JPTJetCollection>    (iConfig.getParameter<edm::InputTag>("JPTJetCollectionLabel"));
  thePFJetCollectionToken     = consumes<std::vector<reco::PFJet> > (iConfig.getParameter<edm::InputTag>("PFJetCollectionLabel"));

  _ptThreshold = iConfig.getParameter<double>("ptThreshold");
  _maxNJets = iConfig.getParameter<double>("maxNJets");
  _maxAbsEta = iConfig.getParameter<double>("maxAbsEta");

}

const char* SUSYDQMAnalyzer::messageLoggerCatregory = "SUSYDQM";


void SUSYDQMAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
				     edm::Run const & iRun,
				     edm::EventSetup const & ) {
  //  if( dqm ) {
    //===========================================================                                                                                 
    // book HT histos.                                                                                                                           
    std::string dir=SUSYFolder;
    dir+="HT"; 
    ibooker.setCurrentFolder(dir);
    hCaloHT = ibooker.book1D("Calo_HT", "", 500, 0., 2000);
    hPFHT   = ibooker.book1D("PF_HT"  , "", 500, 0., 2000);
    //hJPTHT  = ibooker.book1D("JPT_HT" , "", 500, 0., 2000);
    //===========================================================                                                                                 
    // book MET histos.                                                                                                                           

    dir=SUSYFolder;
    dir+="MET";
    ibooker.setCurrentFolder(dir);
    hCaloMET = ibooker.book1D("Calo_MET", "", 500, 0., 1000);
    hPFMET   = ibooker.book1D("PF_MET"  , "", 500, 0., 1000);
    //hTCMET   = ibooker.book1D("TC_MET"  , "", 500, 0., 1000);

    //===========================================================                                                                                 
    // book MHT histos.                                                                                                                           

    dir=SUSYFolder;
    dir+="MHT";
    ibooker.setCurrentFolder(dir);
    hCaloMHT = ibooker.book1D("Calo_MHT", "", 500, 0., 1000);
    hPFMHT   = ibooker.book1D("PF_MHT"  , "", 500, 0., 1000);
    //hJPTMHT  = ibooker.book1D("JPT_MHT" , "", 500, 0., 1000);

    //===========================================================                                                                                 
    // book alpha_T histos.                                                                                                                       

    dir=SUSYFolder;
    dir+="Alpha_T";
    ibooker.setCurrentFolder(dir);
    hCaloAlpha_T = ibooker.book1D("Calo_AlphaT", "", 100, 0., 1.);
    //hJPTAlpha_T  = ibooker.book1D("PF_AlphaT"  , "", 100, 0., 1.);
    hPFAlpha_T   = ibooker.book1D("PF_AlphaT"  , "", 100, 0., 1.);
    //  }
}

void SUSYDQMAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //###########################################################
  // HTand MHT
  
  //===========================================================
  // Calo HT, MHT and alpha_T

  edm::Handle<reco::CaloJetCollection> CaloJetcoll;

  iEvent.getByToken(theCaloJetCollectionToken, CaloJetcoll);

  if(!CaloJetcoll.isValid()) return;
  
  std::vector<math::XYZTLorentzVector> Ps;
  for (reco::CaloJetCollection::const_iterator jet = CaloJetcoll->begin(); jet!=CaloJetcoll->end(); ++jet){
    if ((jet->pt()>_ptThreshold) && (abs(jet->eta()) < _maxAbsEta)){
      if(Ps.size()>_maxNJets) {
	edm::LogInfo(messageLoggerCatregory)<<"NMax Jets exceded..";
        break;
      }
      Ps.push_back(jet->p4());
    }
  }

  hCaloAlpha_T->Fill( alpha_T()(Ps));

  HT< reco::CaloJetCollection > CaloHT(CaloJetcoll, _ptThreshold, _maxAbsEta);

  hCaloHT->Fill(CaloHT.ScalarSum);
  hCaloMHT->Fill(CaloHT.v.Mod());

  //===========================================================
  // PF HT, MHT and alpha_T

  edm::Handle<reco::PFJetCollection> PFjetcoll;

  iEvent.getByToken(thePFJetCollectionToken, PFjetcoll);

  if(!PFjetcoll.isValid()) return;

  Ps.clear();
  for (reco::PFJetCollection::const_iterator jet = PFjetcoll->begin(); jet!=PFjetcoll->end(); ++jet){
    if ((jet->pt()>_ptThreshold) && (abs(jet->eta()) < _maxAbsEta)){
      if(Ps.size()>_maxNJets) {
	edm::LogInfo(messageLoggerCatregory)<<"NMax Jets exceded..";
	break;
      }
      Ps.push_back(jet->p4());
    }
  }
  hPFAlpha_T->Fill( alpha_T()(Ps));

  HT<reco::PFJetCollection> PFHT(PFjetcoll, _ptThreshold, _maxAbsEta);

  hPFHT->Fill(PFHT.ScalarSum);
  hPFMHT->Fill(PFHT.v.Mod());
  
  //===========================================================
  // JPT HT, MHT and alpha_T

  //edm::Handle<reco::JPTJetCollection> JPTjetcoll;

  //iEvent.getByToken(theJPTJetCollectionToken, JPTjetcoll);

  //if(!JPTjetcoll.isValid()) return;

  //Ps.clear();
  //for (reco::JPTJetCollection::const_iterator jet = JPTjetcoll->begin(); jet!=JPTjetcoll->end(); ++jet){
  //if ((jet->pt()>_ptThreshold) && (abs(jet->eta())<_maxAbsEta)){
  //  if(Ps.size()>_maxNJets) {
  //	edm::LogInfo(messageLoggerCatregory)<<"NMax Jets exceded..";
  //    break;
  //  }
  //  Ps.push_back(jet->p4());
  //}
  //}
  //hJPTAlpha_T->Fill( alpha_T()(Ps));

  //HT<reco::JPTJetCollection> JPTHT(JPTjetcoll, _ptThreshold, _maxAbsEta);

  //hJPTHT->Fill(JPTHT.ScalarSum);
  //hJPTMHT->Fill(JPTHT.v.Mod());

  //###########################################################
  // MET

  //===========================================================  
  // Calo MET

  edm::Handle<reco::CaloMETCollection> calometcoll;
  iEvent.getByToken(theCaloMETCollectionToken, calometcoll);

  if(!calometcoll.isValid()) return;

  const CaloMETCollection *calometcol = calometcoll.product();
  const CaloMET *calomet;
  calomet = &(calometcol->front());
  
  hCaloMET->Fill(calomet->pt());

  //===========================================================
  // PF MET

  edm::Handle<reco::PFMETCollection> pfmetcoll;
  iEvent.getByToken(thePFMETCollectionToken, pfmetcoll);
  
  if(!pfmetcoll.isValid()) return;

  const PFMETCollection *pfmetcol = pfmetcoll.product();
  const PFMET *pfmet;
  pfmet = &(pfmetcol->front());

  hPFMET->Fill(pfmet->pt());

  //===========================================================
  // TC MET

  //edm::Handle<reco::METCollection> tcmetcoll;
  //iEvent.getByToken(theTCMETCollectionToken, tcmetcoll);
  
  //if(!tcmetcoll.isValid()) return;

  //const METCollection *tcmetcol = tcmetcoll.product();
  //const MET *tcmet;
  //tcmet = &(tcmetcol->front());

  //hTCMET->Fill(tcmet->pt());

}


SUSYDQMAnalyzer::~SUSYDQMAnalyzer(){
}
