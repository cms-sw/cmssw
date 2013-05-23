#include "RecoJets/JetAnalyzers/interface/myFilter.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// #include "DataFormats/PhotonReco/interface/PhotonFwd.h"
// #include "DataFormats/PhotonReco/interface/Photon.h"



using namespace edm;
using namespace reco;
using namespace std;

#define DEBUG false
#define INVALID 9999.

typedef struct RBX_struct {
  double et;
  double hadEnergy;
  double emEnergy;
  float  hcalTime;
  float  ecalTime;
  int    nTowers;
} RBX ;

typedef struct HPD_struct {
  double et;
  double hadEnergy;
  double emEnergy;
  double time;
  float  hcalTime;
  float  ecalTime;
  int    nTowers;
} HPD ;




//enum HcalSubdetector { HcalEmpty=0, HcalBarrel=1, HcalEndcap=2, HcalOuter=3, HcalForward=4, HcalTriggerTower=5, HcalOther=7 };

//enum SectorId { HBplus=1, HBminus=2, 
// 		HEplus=3, HEminus=4, 
//		HO2plus=5, HO1plus=6, HOzero=7, HO1minus=8, HO2minus=9, 
//		HFplus=10, HFminus=11 }; 


myFilter::myFilter(const edm::ParameterSet& cfg) :
  CaloJetAlgorithm( cfg.getParameter<string>( "CaloJetAlgorithm" ) )
{
  _nEvent      = 0;
  _acceptedEvt = 0;
  _passPt      = 0;
  _passNJets   = 0;
  _passNTrks   = 0;
  _passEMF     = 0;
  _passNTowers = 0;
  _passMET     = 0;
  _passMETSig  = 0;
  _passHighPtTower    = 0;
  _passNRBX    = 0;
  _passHLT     = 0;

  theTriggerResultsLabel = cfg.getParameter<edm::InputTag>("TriggerResultsLabel");
}

myFilter::~myFilter() {
}

void myFilter::beginJob() {
}

void myFilter::endJob() {

  std::cout << "=============================================================" << std::endl;
  std::cout << "myFilter: accepted " 
	    << _acceptedEvt << " / " <<  _nEvent <<  " events." << std::endl;
  std::cout << "Pt           = " << _passPt          << std::endl;
  std::cout << "NJets        = " << _passNJets       << std::endl;
  std::cout << "NTrks        = " << _passNTrks       << std::endl;
  std::cout << "EMF          = " << _passEMF         << std::endl;
  std::cout << "NTowers      = " << _passNTowers     << std::endl;
  std::cout << "MET          = " << _passMET         << std::endl;
  std::cout << "METSig       = " << _passMETSig      << std::endl;
  std::cout << "HighPtTower  = " << _passHighPtTower << std::endl;
  std::cout << "NRBX         = " << _passNRBX        << std::endl;
  std::cout << "=============================================================" << std::endl;

}

bool
myFilter::filter(edm::Event& evt, edm::EventSetup const& es) {

  double HFThreshold   = 4.0;
  //double HOThreshold   = 1.0;


  bool result         = false;
  bool filter_Pt      = false;
  bool filter_NTrks   = false;
  bool filter_EMF     = false;
  bool filter_NJets   = false;
  //bool filter_NTowers = false;
  bool filter_MET     = false;
  bool filter_METSig  = false;
  bool filter_HighPtTower  = false;
  bool filter_NRBX         = false;
  bool filter_HLT          = false;


  bool Pass = false;
  if (evt.id().run() == 124009) {
    if ( (evt.bunchCrossing() == 51)  ||
	 (evt.bunchCrossing() == 151) ||
         (evt.bunchCrossing() == 2824) ) {
      Pass = true;
    }
  }
  if (evt.id().run() == 124020) {
    if ( (evt.bunchCrossing() == 51)  ||
	 (evt.bunchCrossing() == 151) ||
         (evt.bunchCrossing() == 2824) ) {
      Pass = true;
    }
  }
  if (evt.id().run() == 124024) {
    if ( (evt.bunchCrossing() == 51)  ||
	 (evt.bunchCrossing() == 151) ||
         (evt.bunchCrossing() == 2824) ) {
      Pass = true;
    }
  }

  if ( (evt.bunchCrossing() == 51)  ||
       (evt.bunchCrossing() == 151) ||
       (evt.bunchCrossing() == 2824) ) {
    Pass = true;
  }


  // ***********************
  // ***********************

  double HFM_ETime, HFP_ETime;
  double HFM_E, HFP_E;
  double HF_PMM;

  HFM_ETime = 0.;
  HFM_E = 0.;
  HFP_ETime = 0.;
  HFP_E = 0.;

  try {
    std::vector<edm::Handle<HFRecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<HFRecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HFRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
        if (j->id().subdet() == HcalForward) {

          if (j->id().ieta()<0) {
            if (j->energy() > HFThreshold) {
              HFM_ETime += j->energy()*j->time();
              HFM_E     += j->energy();
            }
          } else {
            if (j->energy() > HFThreshold) {
              HFP_ETime += j->energy()*j->time();
              HFP_E     += j->energy();
            }
          }


        }
      }
    }
  } catch (...) {
    cout << "No HF RecHits." << endl;
  }

  if ((HFP_E > 0.) && (HFM_E > 0.)) {
    HF_PMM = (HFP_ETime / HFP_E) - (HFM_ETime / HFM_E);
  } else {
    HF_PMM = INVALID;
  }


  if (fabs(HF_PMM) < 10.) {
    Pass = true;
  } else {
    Pass = false;
  }



  if (Pass) {
  std::cout << ">>>> FIL: Run = "    << evt.id().run()
            << " Event = " << evt.id().event()
            << " Bunch Crossing = " << evt.bunchCrossing()
            << " Orbit Number = "  << evt.orbitNumber()
            <<  std::endl;

  // *********************************************************
  // --- Event Classification
  // *********************************************************

  RBX RBXColl[36];
  HPD HPDColl[144];

  int evtType = 0;

  Handle<CaloTowerCollection> caloTowers;
  evt.getByLabel( "towerMaker", caloTowers );

  for (int i=0;i<36;i++) {
    RBXColl[i].et        = 0;
    RBXColl[i].hadEnergy = 0;
    RBXColl[i].emEnergy  = 0;
    RBXColl[i].hcalTime  = 0;
    RBXColl[i].ecalTime  = 0;
    RBXColl[i].nTowers   = 0;
  }
  for (int i=0;i<144;i++) {
    HPDColl[i].et        = 0;
    HPDColl[i].hadEnergy = 0;
    HPDColl[i].emEnergy  = 0;
    HPDColl[i].hcalTime  = 0;
    HPDColl[i].ecalTime  = 0;
    HPDColl[i].nTowers   = 0;
  }



  for (CaloTowerCollection::const_iterator tower = caloTowers->begin();
       tower != caloTowers->end(); tower++) {

    if (tower->hadEnergy() < 0.) {
    }
    if (tower->emEnergy() < 0.) {
    }
    

    if (tower->et()>0.5) {

      int iRBX = tower->iphi();
      iRBX = iRBX-2;
      if (iRBX == 0)  iRBX = 17;
      if (iRBX == -1) iRBX = 18;
      iRBX = (iRBX-1)/4;

      if (tower->ieta() < 0) iRBX += 18;
      if (iRBX < 36) {
        RBXColl[iRBX].et        += tower->et();
        RBXColl[iRBX].hadEnergy += tower->hadEnergy();
        RBXColl[iRBX].emEnergy  += tower->emEnergy();
        RBXColl[iRBX].hcalTime  += tower->hcalTime();
        RBXColl[iRBX].ecalTime  += tower->ecalTime();
        RBXColl[iRBX].nTowers++;
      }
      /***
      std::cout << "iRBX = " << iRBX << " "
                << "ieta/iphi = " <<  tower->ieta() << " / "  << tower->iphi()
                << " et = " << tower->et()
                << std::endl;
      ***/
      int iHPD = tower->iphi();
      if (tower->ieta() < 0) iHPD = iHPD + 72;
      if (iHPD < 144) {
        HPDColl[iHPD].et        += tower->et();
        HPDColl[iHPD].hadEnergy += tower->hadEnergy();
        HPDColl[iHPD].emEnergy  += tower->emEnergy();
        HPDColl[iHPD].hcalTime  += tower->hcalTime();
        HPDColl[iHPD].ecalTime  += tower->ecalTime();
        HPDColl[iHPD].nTowers++;
      }
      /***
      std::cout << "iHPD = " << iHPD << " "
                << "ieta/iphi = " <<  tower->ieta() << " / "  << tower->iphi()
                << " et = " << tower->et()
                << std::endl;
      ***/

    }

  }


  // Loop over the RBX Collection
  int nRBX = 0;
  int nTowers = 0;
  for (int i=0;i<36;i++) {
    if (RBXColl[i].hadEnergy > 3.0) {
      nRBX++;
      nTowers = RBXColl[i].nTowers;
    }
  }
  if ( (nRBX == 1) && (nTowers > 24) ) {
    evtType = 1;
  }

  // Loop over the HPD Collection
  int nHPD = 0;
  for (int i=0;i<144;i++) {
    if (HPDColl[i].hadEnergy > 3.0) {
      nHPD++;
      nTowers = HPDColl[i].nTowers;
    }
  }
  if ( (nHPD == 1) && (nTowers > 6) ) {
    evtType = 2;
    cout << " nHPD = "   << nHPD
         << " Towers = " << nTowers
         << " Type = "   << evtType
         << endl;
  }



  // *********************************************************
  // --- Access Trigger Info
  // *********************************************************

  // **** Get the TriggerResults container
  Handle<TriggerResults> triggerResults;
  evt.getByLabel(theTriggerResultsLabel, triggerResults);

  Int_t JetLoPass = 0;

  if (triggerResults.isValid()) {
    if (DEBUG) std::cout << "trigger valid " << std::endl;
    const edm::TriggerNames & triggerNames = evt.triggerNames(*triggerResults);
    unsigned int n = triggerResults->size();
    for (unsigned int i=0; i!=n; i++) {
      if ( triggerNames.triggerName(i) == "HLT_Jet30" ) {
        JetLoPass =  triggerResults->accept(i);
        if (DEBUG) std::cout << "Found  HLT_Jet30" << std::endl;
      }
    }
  }

  // *********************************************************
  // --- Vertex Selection
  // *********************************************************

  // *********************************************************
  // --- Track Selection
  // *********************************************************
  edm::Handle<reco::TrackCollection> trackCollection;
  //  evt.getByLabel("ctfWithMaterialTracks", trackCollection);
  evt.getByLabel("generalTracks", trackCollection);
  
  const reco::TrackCollection tC = *(trackCollection.product());
  std::cout << "FIL: Reconstructed "<< tC.size() << " tracks" << std::endl ;

  if (tC.size() > 3) filter_NTrks = true;

  //  h_Trk_NTrk->Fill(tC.size());

  //  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){
  //    h_Trk_pt->Fill(track->pt());
  //  }

  /****
    std::cout << "Track number "<< i << std::endl ;
    std::cout << "\tmomentum: " << track->momentum()<< std::endl;
    std::cout << "\tPT: " << track->pt()<< std::endl;
    std::cout << "\tvertex: " << track->vertex()<< std::endl;
    std::cout << "\timpact parameter: " << track->d0()<< std::endl;
    std::cout << "\tcharge: " << track->charge()<< std::endl;
    std::cout << "\tnormalizedChi2: " << track->normalizedChi2()<< std::endl;

    cout<<"\tFrom EXTRA : "<<endl;
    cout<<"\t\touter PT "<< track->outerPt()<<endl;
    std::cout << "\t direction: " << track->seedDirection() << std::endl;
  ****/




  // *********************************************************
  // --- RecHits
  // *********************************************************
  //  Handle<CaloTowerCollection> caloTowers;
  //  evt.getByLabel( "towerMaker", caloTowers );
  edm::Handle<HcalSourcePositionData> spd;
  
  try {
    std::vector<edm::Handle<HBHERecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<HBHERecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HBHERecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
	//	std::cout << *j << std::endl;
	if (j->id().subdet() == HcalBarrel) {
	  //	  std::cout << "Barrel : " << j->id() << std::endl;
	}
	if (j->id().subdet() == HcalEndcap) {
	}

	/***
	std::cout << j->id()     << " "
		  << j->id().subdet() << " "
		  << j->id().ieta()   << " "
		  << j->id().iphi()   << " "
		  << j->id().depth()  << " "
		  << j->energy() << " "
		  << j->time()   << std::endl;
	****/
      }
    }
  } catch (...) {
    cout << "No HB/HE RecHits." << endl;
  }

  try {
    std::vector<edm::Handle<HFRecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<HFRecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HFRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
	//	std::cout << *j << std::endl;
      }
    }
  } catch (...) {
    cout << "No HF RecHits." << endl;
  }
    
  try {
    std::vector<edm::Handle<HORecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<HORecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HORecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
	//	std::cout << *j << std::endl;
      }
    }
  } catch (...) {
    cout << "No HO RecHits." << endl;
  }



  // *********************************************************
  // --- CaloTower Selection
  // *********************************************************
  //  Handle<CaloTowerCollection> caloTowers;
  //  evt.getByLabel( "towerMaker", caloTowers );

  // --- Loop over towers and make a lists of used and unused towers
  int nTow = 0;
  for (CaloTowerCollection::const_iterator tower = caloTowers->begin();
       tower != caloTowers->end(); tower++) {
    //    std::cout << *tower << std::endl;
    if (tower->et() > 0.5) {
      nTow++;
    /****
      std::cout << "Tower Et = " 
		<< tower->et()          << " " 
		<< tower->emEnergy()    << " EmEne = " 
		<< tower->hadEnergy()   << " HadEne = " 
		<< tower->outerEnergy() << " ETime = " 
		<< tower->ecalTime()    << " HTime = " 
		<< tower->hcalTime()    << " ieta = " 
		<< tower->ieta()        << " iphi = " 
		<< tower->iphi()        << " "  
		<< tower->iphi() / 4 
		<< endl;
    ****/
    }

  }
  /****
  std::cout << "Number of caloTowers = " 
	    <<  caloTowers->size() 
	    <<  " / "
	    << nTow 
	    << std::endl;
  ****/

  // *********************************************************
  // --- Jet Selection
  // *********************************************************
  Handle<CaloJetCollection> jets;
  evt.getByLabel( CaloJetAlgorithm, jets );
  int njet = 0;
  for ( CaloJetCollection::const_iterator ijet=jets->begin(); ijet!=jets->end(); ijet++) {

    if ( (ijet->pt() > 100.) && (JetLoPass != 0) ) {
      filter_HLT = true;
    }


    if ((ijet->pt() > 100.) && (evtType == 0)) {
      filter_HighPtTower = true; 
    }

    if (ijet->pt() > 20.) filter_Pt  = true;
    if (ijet->pt() > 10.)  njet++;
    if (ijet->pt() > 10.) {
      if (ijet->emEnergyFraction() > 0.05)  filter_EMF = true;
    }

    //    if (filter_EMF) {
    //      std::cout << "pt = "   << ijet->pt() 
    //		<< " EMF = " << ijet->emEnergyFraction() << std::endl;
    //    }

    //    std::cout << "pt = "   << ijet->pt() << std::endl;

  }
  if (njet > 1) filter_NJets = true;
  //  if (filter_EMF) {
  //    std::cout << "NJets = "   << njet << std::endl;
  //  }


  // *********************************************************
  // --- MET Selection
  // *********************************************************

  Handle<reco::CaloMETCollection> calometcoll;
  evt.getByLabel("met", calometcoll);
  if (calometcoll.isValid()) {
    const CaloMETCollection *calometcol = calometcoll.product();
    const CaloMET *calomet;
    calomet = &(calometcol->front());
    double caloMET    = calomet->pt();
    //double caloMETSig = calomet->mEtSig();
    //double caloSumET  = calomet->sumEt();
    if ((caloMET > 300.) && (evtType = 0)) filter_MET = true;
  }


  if (nRBX > 3) filter_NRBX = true;

  // *********************************************************
  _nEvent++;  

  if ( (filter_HLT) || (filter_NJets) )  {
    result = true;
    //    _acceptedEvt++;
  }

  /***
  if ( (filter_Pt)  || (filter_NTrks) || (filter_EMF) || (filter_NJets) || 
       (filter_MET) || (filter_METSig) || (filter_HighPtTower) ) {
    result = true;
    _acceptedEvt++;
  }
  ***/

  if ( (filter_Pt) || (filter_NJets) ) {
    result = true;
    _acceptedEvt++;
  }  

  if (filter_Pt)           _passPt++;
  if (filter_NJets)        _passNJets++;
  if (filter_NTrks)        _passNTrks++;
  if (filter_EMF)          _passEMF++;
  if (filter_MET)          _passMET++;
  if (filter_METSig)       _passMETSig++;
  if (filter_HighPtTower)  _passHighPtTower++;
  if (filter_NRBX)         _passNRBX++;
  if (filter_HLT)          _passHLT++;

  /****  
  if ((evt.id().run() == 120020) && (evt.id().event() == 453)) {
    result = true;
    _acceptedEvt++;
  } else {
    result = false;
  }
  ****/

  //  result = true;

  }

  if (result) {
    std::cout << "<<<< Event Passed" 
	      << std::endl;
  }
  return result;

}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(myFilter);
