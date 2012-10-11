#include "RecoJets/JetAnalyzers/interface/myFilter.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
// #include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

// #include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// #include "DataFormats/PhotonReco/interface/PhotonFwd.h"
// #include "DataFormats/PhotonReco/interface/Photon.h"


// include files
#include "CommonTools/RecoAlgos/interface/HBHENoiseFilter.h"
#include "DataFormats/METReco/interface/HcalNoiseSummary.h"


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
  CaloJetAlgorithm( cfg.getParameter<string>( "CaloJetAlgorithm" ) ),
  hcalNoiseSummaryTag_(cfg.getParameter<edm::InputTag>("hcalNoiseSummaryTag"))
{
  _nTotal      = 0;
  _nEvent      = 0;
  _acceptedEvt = 0;
  _passPt      = 0;
  _passNJets   = 0;
  _passDiJet   = 0;
  _passNTrks   = 0;
  _passEMF     = 0;
  _passNTowers = 0;
  _passMET     = 0;
  _passMETSig  = 0;
  _passHighPtTower    = 0;
  _passNRBX    = 0;
  _passNHPDNoise    = 0;
  _passHLT     = 0;
  _passNPMTHits     = 0;
  _passNMultiPMTHits     = 0;
  _passPKAM     = 0;
  _passHFMET     = 0;
  _passNoiseSummary     = 0;
  _passNoiseSummaryEMF     = 0;
  _passNoiseSummaryE2E10   = 0;
  _passNoiseSummaryNHITS   = 0;
  _passNoiseSummaryNoOther   = 0;
  _passNoiseSummaryADC0   = 0;
  _passOERatio     = 0;
  _passTime        = 0;
  _passHBHETime    = 0;
  _passHFTime      = 0;
  _passHFFlagged   = 0;
  _passHFHighEnergy   = 0;

  for (int i=0; i<10; i++) _NoiseResult[i] = 0;

  theTriggerResultsLabel = cfg.getParameter<edm::InputTag>("TriggerResultsLabel");

}

myFilter::~myFilter() {
}

void myFilter::beginJob() {
}

void myFilter::endJob() {

  std::cout << "=============================================================" << std::endl;
  std::cout << "myFilter: accepted " 
	    << _acceptedEvt << " / " <<  _nEvent <<  " / " << _nTotal << " events" << std::endl;
  std::cout << "Pt            = " << _passPt          << std::endl;
  std::cout << "NJets         = " << _passNJets       << std::endl;
  std::cout << "DiJets        = " << _passDiJet       << std::endl;
  std::cout << "NTrks         = " << _passNTrks       << std::endl;
  std::cout << "EMF           = " << _passEMF         << std::endl;
  std::cout << "NTowers       = " << _passNTowers     << std::endl;
  std::cout << "MET           = " << _passMET         << std::endl;
  std::cout << "METSig        = " << _passMETSig      << std::endl;
  std::cout << "HighPtTower   = " << _passHighPtTower << std::endl;
  std::cout << "NRBX          = " << _passNRBX        << std::endl;
  std::cout << "NHPDNoise     = " << _passNHPDNoise   << std::endl;
  std::cout << "NPMTHits      = " << _passNPMTHits    << std::endl;
  std::cout << "NMultPMTHits  = " << _passNMultiPMTHits    << std::endl;
  std::cout << "PKAM          = " << _passPKAM    << std::endl;
  std::cout << "HFMET         = " << _passHFMET    << std::endl;
  std::cout << "Noise Summary = " << _passNoiseSummary    << std::endl;
  std::cout << "Noise Summary EMF = " << _passNoiseSummaryEMF    << std::endl;
  std::cout << "Noise Summary E2E10 = " << _passNoiseSummaryE2E10    << std::endl;
  std::cout << "Noise Summary NHITS = " << _passNoiseSummaryNHITS    << std::endl;
  std::cout << "Noise Summary ADC0 = " << _passNoiseSummaryADC0    << std::endl;
  std::cout << "Noise Summary NoOther = " << _passNoiseSummaryNoOther    << std::endl;
  std::cout << "OERatio       = " << _passOERatio    << std::endl;
  std::cout << "Time          = " << _passTime    << std::endl;
  std::cout << "HF Time       = " << _passHFTime    << std::endl;
  std::cout << "HBHE Time     = " << _passHBHETime    << std::endl;
  std::cout << "HF Flagged    = " << _passHFFlagged    << std::endl;
  std::cout << "HF High Energy= " << _passHFHighEnergy    << std::endl;
  std::cout << "=============================================================" << std::endl;


  for (int i=0; i<10; i++) {
    std::cout << "Noise Results = " << _NoiseResult[i] << std::endl;
  }

}

bool
myFilter::filter(edm::Event& evt, edm::EventSetup const& es) {

  double HFRecHit[100][100][2];


  double HFThreshold   = 4.0;
  //double HOThreshold   = 1.0;


  bool result         = false;
  bool filter_Pt      = false;
  bool filter_DiJet   = false;
  bool filter_NTrks   = false;
  bool filter_EMF     = false;
  bool filter_NJets   = false;
  //bool filter_NTowers = false;
  bool filter_MET     = false;
  bool filter_METSig  = false;
  bool filter_HighPtTower  = false;
  bool filter_NRBX         = false;
  bool filter_NHPDNoise    = false;
  bool filter_HLT          = false;
  bool filter_NPMTHits     = false;
  bool filter_NMultiPMTHits       = false;
//  bool filter_PKAM                = false;
  bool filter_HFMET               = false;
  bool filter_NoiseSummary        = false;
  bool filter_NoiseSummaryEMF     = false;
  bool filter_NoiseSummaryE2E10   = false;
  bool filter_NoiseSummaryNHITS   = false;
  bool filter_NoiseSummaryADC0    = false;
  bool filter_NoiseSummaryNoOther = false;
  bool filter_OERatio      = false;
  bool filter_Time         = false;
  bool filter_HFTime       = false;
  bool filter_HBHETime     = false;
  bool filter_HFFlagged    = false;
  bool filter_HFHighEnergy = false;


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
  // get the Noise summary object

  edm::Handle<HcalNoiseSummary> summary_h;
  evt.getByLabel(hcalNoiseSummaryTag_, summary_h);
  if(!summary_h.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound) << " could not find HcalNoiseSummary.\n";
    //    return true;
  }
  const HcalNoiseSummary summary = *summary_h;

  if(summary.minE2Over10TS()<0.7) {
    filter_NoiseSummaryE2E10 = true;
    filter_NoiseSummary = true;
    _NoiseResult[0]++;
  }
  if(summary.maxE2Over10TS()>0.96) {
    filter_NoiseSummaryE2E10 = true;
    filter_NoiseSummary = true;
    _NoiseResult[1]++;
  }
  if(summary.maxHPDHits()>=17) {
    filter_NoiseSummaryNHITS = true;
    filter_NoiseSummary = true;
    _NoiseResult[2]++;
  }
  if(summary.maxRBXHits()>=999) {
    filter_NoiseSummary = true;
    _NoiseResult[3]++;
  }
  if(summary.maxHPDNoOtherHits()>=10) {
    filter_NoiseSummary = true;
    filter_NoiseSummaryNoOther = true;
    _NoiseResult[4]++;
  }
  if(summary.maxZeros()>=10) {
    filter_NoiseSummaryADC0 = true;
    filter_NoiseSummary     = true;
    _NoiseResult[5]++;
  }
  if(summary.min25GeVHitTime()<-9999.0) {
    filter_NoiseSummary = true;
    _NoiseResult[6]++;
  }
  if(summary.max25GeVHitTime()>9999.0) {
    filter_NoiseSummary = true;
    _NoiseResult[7]++;
  }
  if(summary.minRBXEMF()<0.01) {
    filter_NoiseSummaryEMF = true;
    //    filter_NoiseSummary = true;
    _NoiseResult[8]++;
  }

  //  if (filter_NoiseSummary) 
  //    std::cout << ">>> Noise Filter         = " << filter_NoiseSummary    << std::endl;


  //  summary.passLooseNoiseFilter();
  //  summary.passTightNoiseFilter();
  //  summary.passHighLevelNoiseFilter();


  // ***********************
  // ***********************

  for (int i=0; i<100; i++) {
    for (int j=0; j<100; j++) {
      HFRecHit[i][j][0] = -10.;
      HFRecHit[i][j][1] = -10.;
    }
  }


  double HFM_ETime, HFP_ETime;
  double HFM_E, HFP_E;
  double HF_PMM;
  double MaxRecHitEne;

  MaxRecHitEne = 0;
  HFM_ETime = 0.;
  HFM_E = 0.;
  HFP_ETime = 0.;
  HFP_E = 0.;
  int NPMTHits;
  int NHFDigiTimeHits;
  int NHFLongShortHits;
  int nTime = 0;

  NPMTHits          = 0;
  NHFDigiTimeHits   = 0;
  NHFLongShortHits  = 0;

  try {
    std::vector<edm::Handle<HFRecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<HFRecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HFRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
        if (j->id().subdet() == HcalForward) {

	  int myFlag;
	  myFlag= j->flagField(HcalCaloFlagLabels::HFLongShort);
	  if (myFlag==1) {
	    filter_HFFlagged=true;
	    NHFLongShortHits++;
	  }
	  myFlag= j->flagField(HcalCaloFlagLabels::HFDigiTime);
	  if (myFlag==1) {
	    filter_HFFlagged=true;
	    NHFDigiTimeHits++;
	  }

	  if ( ( (j->flagField(HcalCaloFlagLabels::HFLongShort)) == 0) &&
	       ( (j->flagField(HcalCaloFlagLabels::HFDigiTime))  == 0) ) {
	    if (j->energy() > MaxRecHitEne) MaxRecHitEne = j->energy();
	  }


	  //	  if (filter_HFFlagged) {
	  //	    std::cout << "HF Flagged         = " << _passHFFlagged          << std::endl;
	  //	  }


	  // Long:  depth = 1
	  // Short: depth = 2
	  float en = j->energy();
	  float time = j->time();
	  if ( (en > 20.) && (time > 10.)) {
	    nTime++;
	  }
	  int ieta = j->id().ieta();
	  int iphi = j->id().iphi();
	  int depth = j->id().depth();
	  HFRecHit[ieta+41][iphi][depth-1] = en;

	  // Exclude PMTs with crystals 
	  if (  (j->id().iphi() == 67) &&
	       ((j->id().ieta() == 29) || 
		((j->id().ieta() == 30) && (j->id().depth() == 2)) || 
		((j->id().ieta() == 32) && (j->id().depth() == 2)) || 
		((j->id().ieta() == 35) && (j->id().depth() == 1)) || 
		((j->id().ieta() == 36) && (j->id().depth() == 2)) || 
		((j->id().ieta() == 37) && (j->id().depth() == 1)) || 
		((j->id().ieta() == 38) && (j->id().depth() == 2)) ) ) {
	  } else {
	    if ( (j->flagField(0) != 0) && (j->energy() > 1.) ) {
	      NPMTHits++;
	      if (NPMTHits > 1) {
		std::cout << ">>>> MultiHit: Run = "    << evt.id().run()
			  << " Event = " << evt.id().event()
			  << " iEta = " << j->id().ieta()
			  << " iPhi = " << j->id().iphi()
			  << " depth = " << j->id().depth()
			  << " flag = " << j->flagField(0)
			  << " hits = " << NPMTHits
			  << " energy = " << j->energy()
			  <<  std::endl;
	      }

	    }
	  }

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
      break;
    }
  } catch (...) {
    cout << "No HF RecHits." << endl;
  }

  if (MaxRecHitEne > 1000.) filter_HFHighEnergy = true;

  if (nTime > 0) filter_Time = true;
  if (nTime > 0) filter_HFTime = true;


  double OER, OddEne, EvenEne;
  int nOdd, nEven;

  OER = 0.0;
  for (int iphi=0; iphi<100; iphi++) {
    OddEne = EvenEne = 0.;
    nOdd  = 0;
    nEven = 0;
    for (int ieta=0; ieta<100; ieta++) {
      if (HFRecHit[ieta][iphi][0] > 1.0) {
	if (ieta%2 == 0) {
	  EvenEne += HFRecHit[ieta][iphi][0]; 
	  nEven++;
	} else {
	  OddEne  += HFRecHit[ieta][iphi][0];
	  nOdd++;
	}
      }
      if (HFRecHit[ieta][iphi][1] > 1.0) {
	if (ieta%2 == 0) {
	  EvenEne += HFRecHit[ieta][iphi][1]; 
	  nEven++;
	} else {
	  OddEne  += HFRecHit[ieta][iphi][1]; 
	  nOdd++;
	}
      }
    }
    if (((OddEne + EvenEne) > 10.) && (nOdd > 1) && (nEven > 1)) {
      OER = (OddEne - EvenEne) / (OddEne + EvenEne);
    }
  }


  if (NPMTHits > 0) filter_NPMTHits = true;
  if (NPMTHits > 1) filter_NMultiPMTHits = true;
  //  cout << "NPMTHits = " << NPMTHits << endl;


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

  _nTotal++;
  Pass = true;
  if (Pass) {
    /***
  std::cout << ">>>> FIL: Run = "    << evt.id().run()
            << " Event = " << evt.id().event()
            << " Bunch Crossing = " << evt.bunchCrossing()
            << " Orbit Number = "  << evt.orbitNumber()
            <<  std::endl;
    ***/
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

  double HFMET    = 0.0;
  double HFsum_et = 0.0;
  double HFsum_ex = 0.0;
  double HFsum_ey = 0.0;

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

      Double_t et   = tower->et();
      Double_t phix = tower->phi();

      if (fabs(tower->ieta()) > 29) {
	HFsum_et += et;
	HFsum_ex += et*cos(phix);
	HFsum_ey += et*sin(phix);
      }

    }
  }

  HFMET = sqrt( HFsum_ex*HFsum_ex + HFsum_ey*HFsum_ey);
  if ( (HFMET > 40.) && (NPMTHits == 0) )  filter_HFMET = true;

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
    //    cout << " nHPD = "   << nHPD
    //         << " Towers = " << nTowers
    //         << " Type = "   << evtType
    //         << endl;
  }



  // *********************************************************
  // --- Access Trigger Info
  // *********************************************************

  // **** Get the TriggerResults container
  Handle<TriggerResults> triggerResults;
  evt.getByLabel(theTriggerResultsLabel, triggerResults);

  Int_t JetLoPass = 0;

  /****
  if (triggerResults.isValid()) {
    if (DEBUG) std::cout << "trigger valid " << std::endl;
    edm::TriggerNames triggerNames;    // TriggerNames class
    triggerNames.init(*triggerResults);
    unsigned int n = triggerResults->size();
    for (unsigned int i=0; i!=n; i++) {
      if ( triggerNames.triggerName(i) == "HLT_Jet30" ) {
        JetLoPass =  triggerResults->accept(i);
        if (DEBUG) std::cout << "Found  HLT_Jet30" << std::endl;
      }
    }
  }
  *****/

  // *********************************************************
  // --- Vertex Selection
  // *********************************************************

  // *********************************************************
  // --- Pixel Track and Clusters
  // *********************************************************
  /*******
  // -- Tracks
  edm::Handle<std::vector<reco::Track> > hTrackCollection;
  try {
    evt.getByLabel("generalTracks", hTrackCollection);
  } catch (cms::Exception &ex) {
    if (fVerbose > 1) cout << "No Track collection with label " << fTrackCollectionLabel << endl;
  }

  if (hTrackCollection.isValid()) {
    const std::vector<reco::Track> trackColl = *(hTrackCollection.product());
    nTk = trackColl.size();
  }
  *******/

  // *********************************************************
  // --- Pixel Clusters
  // *********************************************************
  // -- Pixel cluster
  /***
  edm::Handle<reco::SiPixelCluster> hClusterColl;
  evt.getByLabel("siPixelClusters", hClusterColl);
  const reco::SiPixelCluster cC = *(hClusterColl.product());
  ***/

  edm::Handle< edmNew::DetSetVector<SiPixelCluster> > hClusterColl;
  evt.getByLabel("siPixelClusters", hClusterColl);
  const edmNew::DetSetVector<SiPixelCluster> clustColl = *(hClusterColl.product());
  //  nCl = clustColl.size();

  /***
  int nCl = 0;
  if (hClusterColl.isValid()) {
    const edmNew::DetSetVector<SiPixelCluster> clustColl = *(hClusterColl.product());
    nCl = clustColl.size();
  }
  ***/

  // *********************************************************
  // --- Track Selection
  // *********************************************************
  edm::Handle<reco::TrackCollection> trackCollection;
  //  evt.getByLabel("ctfWithMaterialTracks", trackCollection);
  evt.getByLabel("generalTracks", trackCollection);
  
  const reco::TrackCollection tC = *(trackCollection.product());
  //  std::cout << "FIL: Reconstructed "<< tC.size() << " tracks" << std::endl ;

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


  if ((tC.size() > 100) && (clustColl.size() > 1000)) {
    _passPKAM++;
//    filter_PKAM = true;
  }
  //  std::cout << "N Tracks =  "  << tC.size() 
  //	    << " N Cluster = " << clustColl.size() << std::endl ;



  // *********************************************************
  // --- RecHits
  // *********************************************************
  //  Handle<CaloTowerCollection> caloTowers;
  //  evt.getByLabel( "towerMaker", caloTowers );
  edm::Handle<HcalSourcePositionData> spd;

  int nHPDNoise = 0;
  nTime = 0;

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

	if (j->flagField(0) != 0) nHPDNoise++;

	float en = j->energy();
	float time = j->time();

	if ( (en > 10.) && (time > 20.)) {
	  nTime++;
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

  if (nHPDNoise > 10) filter_NHPDNoise = true;
  if (nTime > 0) filter_HBHETime = true;


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
  int nDiJet = 0;
  for ( CaloJetCollection::const_iterator ijet=jets->begin(); ijet!=jets->end(); ijet++) {

    if ( (ijet->pt() > 100.) && (JetLoPass != 0) ) {
      filter_HLT = true;
    }


    if ((ijet->pt() > 100.) && (evtType == 0)) {
      filter_HighPtTower = true; 
    }

    if (ijet->pt() > 50.) nDiJet++;
    if (ijet->pt() > 50.) filter_Pt  = true;
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

  if (nDiJet > 1) filter_DiJet = true;
  if (njet > 1) filter_NJets = true;
  //  if (filter_EMF) {
  //    std::cout << "NJets = "   << njet << std::endl;
  //  }


  // *********************************************************
  // --- MET Selection
  // *********************************************************

  Handle<reco::CaloMETCollection> calometcoll;
  evt.getByLabel("met", calometcoll);
  double caloMET = 0;
  if (calometcoll.isValid()) {
    const CaloMETCollection *calometcol = calometcoll.product();
    const CaloMET *calomet;
    calomet = &(calometcol->front());
    caloMET = calomet->pt();
    //double caloMETSig = calomet->mEtSig();
    //double caloSumET  = calomet->sumEt();
    //    if ((caloMET > 50.) && (evtType = 0)) filter_MET = true;
    if (caloMET > 40.) filter_MET = true;
  }
  if ((std::abs(OER) > 0.9) && (caloMET > 20.0)) filter_OERatio = true;
  if (nRBX > 3) filter_NRBX = true;

  // *********************************************************
  _nEvent++;  

  //  if ( (filter_HLT) || (filter_NJets) )  {
  //    result = true;
  //    _acceptedEvt++;
  //  }

  /***
  if ( (filter_Pt)  || (filter_NTrks) || (filter_EMF) || (filter_NJets) || 
       (filter_MET) || (filter_METSig) || (filter_HighPtTower) ) {
    result = true;
    _acceptedEvt++;
  }
  ***/

  //  if ( (filter_Pt) || (filter_NJets) ) {
  //    result = true;
  //    _acceptedEvt++;
  //  }  

  //  if ((filter_PKAM) || (filter_HFMET) ||(filter_NMultiPMTHits) )  {
  //  if ( (filter_DiJet) || (filter_HFMET) ||(filter_NMultiPMTHits) )  {
  //  if ( (filter_NHPDNoise) && ( (filter_Pt) || (filter_MET) ) )  {
  //    result = true;
  //    _acceptedEvt++;
  //  }  

  //  if ( (filter_NoiseSummary) && ( (filter_Pt) || (filter_MET) ) )  {
  //    result = true;
  //    _acceptedEvt++;
  //  }  

  //  if (filter_NoiseSummaryEMF) {
  //    result = true;
  //    _acceptedEvt++;
  //  }  

  //  if (filter_Time) {
  //  if ( (filter_MET) && (!filter_NoiseSummary) )  {

  //  if (filter_NoiseSummary) {
  //    result = true;
  //    _acceptedEvt++;
  //  }  

  //  if (filter_HFTime) {
  //    result = true;
  //    _acceptedEvt++;
  //  }  

  //  if (filter_HBHETime) {
  //    result = true;
  //    _acceptedEvt++;
  //  }  


  //  if (!filter_NoiseSummary && filter_HBHETime) {
  //    result = true;
  //    _acceptedEvt++;
  //  }

  //  if (filter_NoiseSummary && filter_NTrks) {
  //    result = true;
  //    _acceptedEvt++;
  //  }

  //  if (!filter_NoiseSummaryADC0   && !filter_NoiseSummaryNHITS && 
  //      filter_NoiseSummaryE2E10 && !filter_NoiseSummaryNoOther) {
  //    result = true;
  //    _acceptedEvt++;
  //  }  


  //  if ((filter_HFFlagged) && ((NHFLongShortHits > 2) || (NHFDigiTimeHits > 2))) {
  //    result = true;
  //    _acceptedEvt++;
  //  }

  if (filter_HFHighEnergy) {
    result = true;
    _acceptedEvt++;
  }

  if (filter_Pt)            _passPt++;
  if (filter_NJets)         _passNJets++;
  if (filter_DiJet)         _passDiJet++;
  if (filter_NTrks)         _passNTrks++;
  if (filter_EMF)           _passEMF++;
  if (filter_MET)           _passMET++;
  if (filter_METSig)        _passMETSig++;
  if (filter_HighPtTower)   _passHighPtTower++;
  if (filter_NRBX)          _passNRBX++;
  if (filter_NHPDNoise)     _passNHPDNoise++;
  if (filter_HLT)           _passHLT++;
  if (filter_NPMTHits)      _passNPMTHits++;
  if (filter_NMultiPMTHits) _passNMultiPMTHits++;
  if (filter_HFMET)         _passHFMET++;
  if (filter_NoiseSummary)  _passNoiseSummary++;
  if (filter_NoiseSummaryEMF)      _passNoiseSummaryEMF++;
  if (filter_NoiseSummaryE2E10)    _passNoiseSummaryE2E10++;
  if (filter_NoiseSummaryNHITS)    _passNoiseSummaryNHITS++;
  if (filter_NoiseSummaryADC0)     _passNoiseSummaryADC0++;
  if (filter_NoiseSummaryNoOther)  _passNoiseSummaryNoOther++;
  if (filter_OERatio)   _passOERatio++;
  if (filter_Time)      _passTime++;
  if (filter_HFTime)    _passHFTime++;
  if (filter_HBHETime)  _passHBHETime++;
  if (filter_HFFlagged) _passHFFlagged++;
  if (filter_HFHighEnergy) _passHFHighEnergy++;

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
