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
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"

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
  dqm = edm::Service<DQMStore>().operator->();
  //std::cout<<"here 1!"<<std::endl;
}


void SUSYDQMAnalyzer::beginJob(void){

  // Load parameters
  //std::cout<<"here 2!"<<std::endl;
  thePFMETCollectionLabel     = iConfig.getParameter<edm::InputTag>("PFMETCollectionLabel");
  theCaloMETCollectionLabel   = iConfig.getParameter<edm::InputTag>("CaloMETCollectionLabel");
  theTCMETCollectionLabel     = iConfig.getParameter<edm::InputTag>("TCMETCollectionLabel");

  theCaloJetCollectionLabel   = iConfig.getParameter<edm::InputTag>("CaloJetCollectionLabel");
  theJPTJetCollectionLabel    = iConfig.getParameter<edm::InputTag>("JPTJetCollectionLabel");
  thePFJetCollectionLabel     = iConfig.getParameter<edm::InputTag>("PFJetCollectionLabel");

  _ptThreshold = iConfig.getParameter<double>("ptThreshold");
  _maxNJets = iConfig.getParameter<double>("maxNJets");
  _maxAbsEta = iConfig.getParameter<double>("maxAbsEta");

  //std::cout<<"here 3!"<<std::endl;
  //std::cout<<thePFJetCollectionLabel<<std::endl;
}

void SUSYDQMAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup){
  //std::cout<<"here 4!"<<std::endl;
  if( dqm ) {
    //std::cout<<"here 5!"<<std::endl;
    //===========================================================  
    // book HT histos.

    std::string dir=SUSYFolder;
    dir+="HT";
    dqm->setCurrentFolder(dir);
    hCaloHT = dqm->book1D("Calo_HT", "", 500, 0., 2000);
    hPFHT   = dqm->book1D("PF_HT"  , "", 500, 0., 2000);
    hJPTHT  = dqm->book1D("JPT_HT" , "", 500, 0., 2000);

    //===========================================================  
    // book MET histos.

    dir=SUSYFolder;
    dir+="MET";
    dqm->setCurrentFolder(dir);
    hCaloMET = dqm->book1D("Calo_MET", "", 500, 0., 1000);
    hPFMET   = dqm->book1D("PF_MET"  , "", 500, 0., 1000);
    hTCMET   = dqm->book1D("TC_MET"  , "", 500, 0., 1000);

    //===========================================================  
    // book MHT histos.

    dir=SUSYFolder;
    dir+="MHT"; 
    dqm->setCurrentFolder(dir);
    hCaloMHT = dqm->book1D("Calo_MHT", "", 500, 0., 1000);
    hPFMHT   = dqm->book1D("PF_MHT"  , "", 500, 0., 1000);
    hJPTMHT  = dqm->book1D("JPT_MHT" , "", 500, 0., 1000);
   
    //===========================================================  
    // book alpha_T histos.

    dir=SUSYFolder;
    dir+="Alpha_T";
    dqm->setCurrentFolder(dir);
    hCaloAlpha_T = dqm->book1D("Calo_AlphaT", "", 100, 0., 1.);
    hJPTAlpha_T  = dqm->book1D("PF_AlphaT"  , "", 100, 0., 1.);
    hPFAlpha_T   = dqm->book1D("JPT_AlphaT"  , "", 100, 0., 1.);
    //std::cout<<"here 6!"<<std::endl;
  }
}

void SUSYDQMAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //std::cout<<"here 7!"<<std::endl;
  // Commenting out unused but initialized variables. [Suchandra Dutta]
//  EventID TheEvent = iEvent.id();
  //int BXN = iEvent.bunchCrossing() ;
  //bool Dump = TextFileName.size();
  //int TheEventNumber = TheEvent.event();
  //int Lumi = iEvent.luminosityBlock();
  //int Run  = iEvent.run();

  //###########################################################
  // HTand MHT
  
  //===========================================================
  // Calo HT, MHT and alpha_T

  edm::Handle<reco::CaloJetCollection> CaloJetcoll;

  iEvent.getByLabel(theCaloJetCollectionLabel, CaloJetcoll);

  if(!CaloJetcoll.isValid()) return;
  
  std::vector<math::XYZTLorentzVector> Ps;
  for (reco::CaloJetCollection::const_iterator jet = CaloJetcoll->begin(); jet!=CaloJetcoll->end(); ++jet){
    if ((jet->pt()>_ptThreshold) && (abs(jet->eta()) < _maxAbsEta)){
      if(Ps.size()>_maxNJets) {
        std::cout<<"NMax Jets exceded.."<<std::endl;
        break;
      }
      Ps.push_back(jet->p4());
    }
  }

  hCaloAlpha_T->Fill( alpha_T()(Ps));
  //std::cout<<"Calo Alpha_T "<<alpha_T()(Ps)<<std::endl;

  HT< reco::CaloJetCollection > CaloHT(CaloJetcoll, _ptThreshold, _maxAbsEta);

  //std::cout << "Calo MHT " << CaloHT.v.Mod() << std::endl;
  //std::cout << "Calo HT " << CaloHT.ScalarSum << std::endl;
  hCaloHT->Fill(CaloHT.ScalarSum);
  hCaloMHT->Fill(CaloHT.v.Mod());

  //===========================================================
  // PF HT, MHT and alpha_T

  edm::Handle<reco::PFJetCollection> PFjetcoll;

  iEvent.getByLabel(thePFJetCollectionLabel, PFjetcoll);

  if(!PFjetcoll.isValid()) return;

  Ps.clear();
  for (reco::PFJetCollection::const_iterator jet = PFjetcoll->begin(); jet!=PFjetcoll->end(); ++jet){
    if ((jet->pt()>_ptThreshold) && (abs(jet->eta()) < _maxAbsEta)){
      if(Ps.size()>_maxNJets) {
	std::cout<<"NMax Jets exceded.."<<std::endl;
	break;
      }
      Ps.push_back(jet->p4());
    }
  }
  //std::cout<<Ps.size()<<" "<<Ps.max_size()<<std::endl;
  hPFAlpha_T->Fill( alpha_T()(Ps));
  //std::cout<<"PF Alpha_T "<<alpha_T()(Ps)<<std::endl;

  HT<reco::PFJetCollection> PFHT(PFjetcoll, _ptThreshold, _maxAbsEta);

  //std::cout << "PF MHT " << PFHT.v.Mod() << std::endl;
  //std::cout << "PF HT " << PFHT.ScalarSum << std::endl;
  hPFHT->Fill(PFHT.ScalarSum);
  hPFMHT->Fill(PFHT.v.Mod());
  
  //===========================================================
  // JPT HT, MHT and alpha_T

  edm::Handle<reco::JPTJetCollection> JPTjetcoll;

  iEvent.getByLabel(theJPTJetCollectionLabel, JPTjetcoll);

  if(!JPTjetcoll.isValid()) return;

  Ps.clear();
  for (reco::JPTJetCollection::const_iterator jet = JPTjetcoll->begin(); jet!=JPTjetcoll->end(); ++jet){
    if ((jet->pt()>_ptThreshold) && (abs(jet->eta())<_maxAbsEta)){
      if(Ps.size()>_maxNJets) {
        std::cout<<"NMax Jets exceded..."<<std::endl;
        break;
      }
      Ps.push_back(jet->p4());
    }
  }
  hJPTAlpha_T->Fill( alpha_T()(Ps));
  //std::cout<<"JPT Alpha_T "<<alpha_T()(Ps)<<std::endl;

  HT<reco::JPTJetCollection> JPTHT(JPTjetcoll, _ptThreshold, _maxAbsEta);

  //std::cout << "JPT MHT " << JPTHT.v.Mod() << std::endl;
  //std::cout << "JPT HT " << JPTHT.ScalarSum << std::endl;
  hJPTHT->Fill(JPTHT.ScalarSum);
  hJPTMHT->Fill(JPTHT.v.Mod());

  //###########################################################
  // MET

  //===========================================================  
  // Calo MET

  edm::Handle<reco::CaloMETCollection> calometcoll;
  iEvent.getByLabel(theCaloMETCollectionLabel, calometcoll);

  if(!calometcoll.isValid()) return;

  const CaloMETCollection *calometcol = calometcoll.product();
  const CaloMET *calomet;
  calomet = &(calometcol->front());
  
  hCaloMET->Fill(calomet->pt());
  //std::cout<<"Calo MET "<<calomet->pt()<<std::endl;

  //===========================================================
  // PF MET

  edm::Handle<reco::PFMETCollection> pfmetcoll;
  iEvent.getByLabel(thePFMETCollectionLabel, pfmetcoll);
  
  if(!pfmetcoll.isValid()) return;

  const PFMETCollection *pfmetcol = pfmetcoll.product();
  const PFMET *pfmet;
  pfmet = &(pfmetcol->front());

  hPFMET->Fill(pfmet->pt());
  //std::cout<<"PF MET "<<pfmet->pt()<<std::endl;

  //===========================================================
  // TC MET

  edm::Handle<reco::METCollection> tcmetcoll;
  iEvent.getByLabel(theTCMETCollectionLabel, tcmetcoll);
  
  if(!tcmetcoll.isValid()) return;

  const METCollection *tcmetcol = tcmetcoll.product();
  const MET *tcmet;
  tcmet = &(tcmetcol->front());

  hTCMET->Fill(tcmet->pt());
  //std::cout<<"TC MET "<<tcmet->pt()<<std::endl;

}

void SUSYDQMAnalyzer::endRun(const edm::Run&, const edm::EventSetup&){  
}

SUSYDQMAnalyzer::~SUSYDQMAnalyzer(){
}
