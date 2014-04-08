// -*- C++ -*-
//
// Package:    WenuPlots
// Class:      WenuPlots
//
/*

 Description:
    this is an analyzer that reads pat::CompositeCandidate WenuCandidates
    and creates some plots
 Implementation:
    The code takes the output of the WenuCandidateFilter and
    * implements on them  a user defined selection
    * implements the selection with one cut (configurable which cut) inverted
    * creates a set of basic plots with the Wenu Candidate distribution
      vs MET, MT etc. These plots are stored in a root file
    If you have several root files from different runs you have to run a macro
    to combine the output and have the final plots

    This analyser is PAT based in the sense that it reads CompositeCandidates,
    which are composed of a pat::MET plus a pat::Electron. You normally
    don't have to change this file when the CMSSW version changes because it
    contains only methods from the stable core of pat Objects. Most
    version dependent changes should be in WenuCandidateFilter.cc
 TO DO LIST:
    * more plots to be added
    * there should be an base Plots class from which WenuPlots and ZeePlots
      inherit. this makes sense since they have so many common methods

  Changes Log:
  12Feb09  First Release of the code for CMSSW_2_2_X
  16Sep09  tested that it works with 3_1_2 as well
  09Sep09  added one extra iso with the name userIso_XX_
  23Feb09  added option to include extra IDs that are in CMSSW, such as
           categorized, likehood etc
           added extra variables TIP and E/P
  27May10  changes to apply the Spring10 selections, relative isolations
           the 3 default ones, pat user isolations added in the end
           change to framework independent variable definitions
	   double->Double_t etc and math.h functions from TMath
  01Jul10  second electron information added
  Contact:
  Nikolaos Rompotis  -  Nikolaos.Rompotis@Cern.ch
  Imperial College London


*/
//
// Original Author:  Nikolaos Rompotis


#include "ElectroWeakAnalysis/WENu/interface/WenuPlots.h"
#include "DataFormats/Math/interface/deltaR.h"
//#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterCrackCorrectionFunctor.h"

WenuPlots::WenuPlots(const edm::ParameterSet& iConfig)

{
////////////////////////////////////////////////////////////////////////////
//                   I N P U T      P A R A M E T E R S
////////////////////////////////////////////////////////////////////////////
//
///////
//  WENU COLLECTION   //////////////////////////////////////////////////////
//

  wenuCollectionToken_ = consumes<pat::CompositeCandidateCollection>(iConfig.getUntrackedParameter<edm::InputTag>("wenuCollectionTag"));
  //
  // code parameters
  //
  std::string outputFile_D = "histos.root";
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", outputFile_D);
  WENU_VBTFselectionFileName_ = iConfig.getUntrackedParameter<std::string>("WENU_VBTFselectionFileName");
  WENU_VBTFpreseleFileName_ = iConfig.getUntrackedParameter<std::string>("WENU_VBTFpreseleFileName");
  DatasetTag_ = iConfig.getUntrackedParameter<Int_t>("DatasetTag");
  //
  // use of precalculatedID
  // if you use it, then no other cuts are applied
  usePrecalcID_ = iConfig.getUntrackedParameter<Bool_t>("usePrecalcID",false);
  if (usePrecalcID_) {
    usePrecalcIDType_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDType");
    usePrecalcIDSign_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDSign","=");
    usePrecalcIDValue_= iConfig.getUntrackedParameter<Double_t>("usePrecalcIDValue");
  }
  useValidFirstPXBHit_ = iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit",false);
  useConversionRejection_ = iConfig.getUntrackedParameter<Bool_t>("useConversionRejection",false);
  useExpectedMissingHits_ = iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits",false);

  maxNumberOfExpectedMissingHits_ = iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits",1);
  if (not usePrecalcID_) {
    if (useValidFirstPXBHit_) std::cout << "WenuPlots: Warning: you have demanded a valid 1st layer PXB hit" << std::endl;
    if (useConversionRejection_) std::cout << "WenuPlots: Warning: you have demanded egamma conversion rejection criteria to be applied" << std::endl;
    if (useExpectedMissingHits_) std::cout << "WenuPlots: Warning: you have demanded at most "
					   <<maxNumberOfExpectedMissingHits_ << " missing inner hits "<< std::endl;
  }
  else {
    std::cout << "WenuPlots: Using Precalculated ID with type " << usePrecalcIDType_
	      << usePrecalcIDSign_ << usePrecalcIDValue_ << std::endl;
  }
  if ((useValidFirstPXBHit_ || useExpectedMissingHits_ || useConversionRejection_) && (not usePrecalcID_)) {
    usePreselection_ = true;
  } else { usePreselection_ = false; }
  includeJetInformationInNtuples_ = iConfig.getUntrackedParameter<Bool_t>("includeJetInformationInNtuples", false);
  if (includeJetInformationInNtuples_) {
    caloJetCollectionTag_ = iConfig.getUntrackedParameter<edm::InputTag>("caloJetCollectionTag");
    caloJetCollectionToken_ = consumes< reco::CaloJetCollection >(caloJetCollectionTag_);
    pfJetCollectionTag_   = iConfig.getUntrackedParameter<edm::InputTag>("pfJetCollectionTag");
    pfJetCollectionToken_ = consumes< reco::PFJetCollection >(pfJetCollectionTag_);
    DRJetFromElectron_    = iConfig.getUntrackedParameter<Double_t>("DRJetFromElectron");
  }
  storeExtraInformation_ = iConfig.getUntrackedParameter<Bool_t>("storeExtraInformation");
  storeAllSecondElectronVariables_ = iConfig.getUntrackedParameter<Bool_t>("storeAllSecondElectronVariables", false);
  // primary vtx collections
  PrimaryVerticesCollectionToken_=consumes< std::vector<reco::Vertex> >(iConfig.getUntrackedParameter<edm::InputTag>("PrimaryVerticesCollection", edm::InputTag("offlinePrimaryVertices")));
  PrimaryVerticesCollectionBSToken_=consumes< std::vector<reco::Vertex> >(iConfig.getUntrackedParameter<edm::InputTag>("PrimaryVerticesCollectionBS",edm::InputTag("offlinePrimaryVerticesWithBS")));
  //
  // the selection cuts:
  trackIso_EB_ = iConfig.getUntrackedParameter<Double_t>("trackIso_EB", 1000.);
  ecalIso_EB_ = iConfig.getUntrackedParameter<Double_t>("ecalIso_EB", 1000.);
  hcalIso_EB_ = iConfig.getUntrackedParameter<Double_t>("hcalIso_EB", 1000.);
  //
  trackIso_EE_ = iConfig.getUntrackedParameter<Double_t>("trackIso_EE", 1000.);
  ecalIso_EE_ = iConfig.getUntrackedParameter<Double_t>("ecalIso_EE", 1000.);
  hcalIso_EE_ = iConfig.getUntrackedParameter<Double_t>("hcalIso_EE", 1000.);
  //
  sihih_EB_ = iConfig.getUntrackedParameter<Double_t>("sihih_EB");
  dphi_EB_ = iConfig.getUntrackedParameter<Double_t>("dphi_EB");
  deta_EB_ = iConfig.getUntrackedParameter<Double_t>("deta_EB");
  hoe_EB_ = iConfig.getUntrackedParameter<Double_t>("hoe_EB");
  cIso_EB_ = iConfig.getUntrackedParameter<Double_t>("cIso_EB", 1000.);
  tip_bspot_EB_=iConfig.getUntrackedParameter<Double_t>("tip_bspot_EB", 1000.);
  eop_EB_=iConfig.getUntrackedParameter<Double_t>("eop_EB", 1000.);
  //
  sihih_EE_ = iConfig.getUntrackedParameter<Double_t>("sihih_EE");
  dphi_EE_ = iConfig.getUntrackedParameter<Double_t>("dphi_EE");
  deta_EE_ = iConfig.getUntrackedParameter<Double_t>("deta_EE");
  hoe_EE_ = iConfig.getUntrackedParameter<Double_t>("hoe_EE");
  cIso_EE_ = iConfig.getUntrackedParameter<Double_t>("cIso_EE", 1000.);
  tip_bspot_EE_=iConfig.getUntrackedParameter<Double_t>("tip_bspot_EE", 1000.);
  eop_EE_=iConfig.getUntrackedParameter<Double_t>("eop_EE", 1000.);
  //
  trackIsoUser_EB_ = iConfig.getUntrackedParameter<Double_t>("trackIsoUser_EB", 1000.);
  ecalIsoUser_EB_ = iConfig.getUntrackedParameter<Double_t>("ecalIsoUser_EB", 1000.);
  hcalIsoUser_EB_ = iConfig.getUntrackedParameter<Double_t>("hcalIsoUser_EB", 1000.);
  trackIsoUser_EE_ = iConfig.getUntrackedParameter<Double_t>("trackIsoUser_EE", 1000.);
  ecalIsoUser_EE_ = iConfig.getUntrackedParameter<Double_t>("ecalIsoUser_EE", 1000.);
  hcalIsoUser_EE_ = iConfig.getUntrackedParameter<Double_t>("hcalIsoUser_EE", 1000.);
  //
  trackIso_EB_inv = iConfig.getUntrackedParameter<Bool_t>("trackIso_EB_inv", false);
  ecalIso_EB_inv = iConfig.getUntrackedParameter<Bool_t>("ecalIso_EB_inv", false);
  hcalIso_EB_inv = iConfig.getUntrackedParameter<Bool_t>("hcalIso_EB_inv", false);
  //
  trackIso_EE_inv = iConfig.getUntrackedParameter<Bool_t>("trackIso_EE_inv", false);
  ecalIso_EE_inv = iConfig.getUntrackedParameter<Bool_t>("ecalIso_EE_inv", false);
  hcalIso_EE_inv = iConfig.getUntrackedParameter<Bool_t>("hcalIso_EE_inv", false);
  //
  sihih_EB_inv = iConfig.getUntrackedParameter<Bool_t>("sihih_EB_inv", false);
  dphi_EB_inv = iConfig.getUntrackedParameter<Bool_t>("dphi_EB_inv", false);
  deta_EB_inv = iConfig.getUntrackedParameter<Bool_t>("deta_EB_inv", false);
  hoe_EB_inv = iConfig.getUntrackedParameter<Bool_t>("hoe_EB_inv", false);
  cIso_EB_inv = iConfig.getUntrackedParameter<Bool_t>("cIso_EB_inv", false);
  tip_bspot_EB_inv=iConfig.getUntrackedParameter<Bool_t>("tip_bspot_EB_inv", false);
  eop_EB_inv=iConfig.getUntrackedParameter<Bool_t>("eop_EB_inv", false);
  //
  sihih_EE_inv = iConfig.getUntrackedParameter<Bool_t>("sihih_EE_inv", false);
  dphi_EE_inv = iConfig.getUntrackedParameter<Bool_t>("dphi_EE_inv", false);
  deta_EE_inv = iConfig.getUntrackedParameter<Bool_t>("deta_EE_inv", false);
  hoe_EE_inv = iConfig.getUntrackedParameter<Bool_t>("hoe_EE_inv", false);
  cIso_EE_inv = iConfig.getUntrackedParameter<Bool_t>("cIso_EE_inv", false);
  tip_bspot_EE_inv=iConfig.getUntrackedParameter<Bool_t>("tip_bspot_EE_inv", false);
  eop_EE_inv=iConfig.getUntrackedParameter<Bool_t>("eop_EE_inv", false);
  //
  trackIsoUser_EB_inv = iConfig.getUntrackedParameter<Bool_t>("trackIsoUser_EB_inv", false);
  ecalIsoUser_EB_inv = iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser_EB_inv", false);
  hcalIsoUser_EB_inv = iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser_EB_inv", false);
  trackIsoUser_EE_inv = iConfig.getUntrackedParameter<Bool_t>("trackIsoUser_EE_inv", false);
  ecalIsoUser_EE_inv = iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser_EE_inv", false);
  hcalIsoUser_EE_inv = iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser_EE_inv", false);

}



WenuPlots::~WenuPlots()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
WenuPlots::analyze(const edm::Event& iEvent, const edm::EventSetup& es)
{
  using namespace std;
  //
  //  Get the collections here
  //
  edm::Handle<pat::CompositeCandidateCollection> WenuCands;
  iEvent.getByToken(wenuCollectionToken_, WenuCands);

  if (not WenuCands.isValid()) {
    cout << "Warning: no wenu candidates in this event..." << endl;
    return;
  }
  const pat::CompositeCandidateCollection *wcands = WenuCands.product();
  const pat::CompositeCandidateCollection::const_iterator
    wenuIter = wcands->begin();
  const pat::CompositeCandidate wenu = *wenuIter;
  //
  // get the parts of the composite candidate:
  const pat::Electron * myElec=
    dynamic_cast<const pat::Electron*> (wenu.daughter("electron"));
  const pat::MET * myMet=
    dynamic_cast<const pat::MET*> (wenu.daughter("met"));
  const pat::MET * myPfMet=
    dynamic_cast<const pat::MET*> (wenu.daughter("pfmet"));
  const pat::MET * myTcMet=
    dynamic_cast<const pat::MET*> (wenu.daughter("tcmet"));
  // _______________________________________________________________________
  //
  // VBTF Root tuple production --------------------------------------------
  // _______________________________________________________________________
  //
  // .......................................................................
  // vbtf  produces 2 root tuples: one that contains the highest pT electron
  //  that  passes a  user  defined selection  and one  other  with only the
  //  preselection criteria applied
  // .......................................................................
  //
  // fill the tree variables
  runNumber   = iEvent.run();
  eventNumber = Long64_t( iEvent.eventAuxiliary().event() );
  lumiSection = (Int_t) iEvent.luminosityBlock();
  //
  ele_sc_eta       = (Float_t)  myElec->superCluster()->eta();
  ele_sc_phi       = (Float_t)  myElec->superCluster()->phi();
  double scx = myElec->superCluster()->x();
  double scy = myElec->superCluster()->y();
  double scz = myElec->superCluster()->z();
  ele_sc_rho       = (Float_t)  sqrt( scx*scx + scy*scy + scz*scz );
  ele_sc_energy    = (Float_t)  myElec->superCluster()->energy();
  ele_sc_gsf_et    = (Float_t)  myElec->superCluster()->energy()/TMath::CosH(myElec->gsfTrack()->eta());
  ele_cand_eta     = (Float_t)  myElec->eta();
  ele_cand_phi     = (Float_t)  myElec->phi();
  ele_cand_et      = (Float_t)  myElec->et();
  //
  ele_iso_track    = (Float_t)  myElec->dr03IsolationVariables().tkSumPt / ele_cand_et;
  ele_iso_ecal     = (Float_t)  myElec->dr03IsolationVariables().ecalRecHitSumEt/ele_cand_et;
  ele_iso_hcal     = (Float_t)  ( myElec->dr03IsolationVariables().hcalDepth1TowerSumEt +
       myElec->dr03IsolationVariables().hcalDepth2TowerSumEt) / ele_cand_et;
  //
  ele_id_sihih     = (Float_t)  myElec->sigmaIetaIeta();
  ele_id_deta      = (Float_t)  myElec->deltaEtaSuperClusterTrackAtVtx();
  ele_id_dphi      = (Float_t)  myElec->deltaPhiSuperClusterTrackAtVtx();
  ele_id_hoe       = (Float_t)  myElec->hadronicOverEm();
  //
  ele_cr_mhitsinner= myElec->gsfTrack()->trackerExpectedHitsInner().numberOfHits();
  ele_cr_dcot      = myElec->convDcot();
  ele_cr_dist      = myElec->convDist();
  //
  ele_vx           = (Float_t) myElec->vx();
  ele_vy           = (Float_t) myElec->vy();
  ele_vz           = (Float_t) myElec->vz();
  // get the primary vtx information
  // no BS
  edm::Handle< std::vector<reco::Vertex> > pVtx;
  iEvent.getByToken(PrimaryVerticesCollectionToken_, pVtx);
  const std::vector<reco::Vertex> Vtx = *(pVtx.product());
  // with BS
  edm::Handle< std::vector<reco::Vertex> > pVtxBS;
  iEvent.getByToken(PrimaryVerticesCollectionBSToken_, pVtxBS);
  const std::vector<reco::Vertex> VtxBS = *(pVtxBS.product());
  if (Vtx.size() > 0) {
    pv_x = Float_t(Vtx[0].position().x());
    pv_y = Float_t(Vtx[0].position().y());
    pv_z = Float_t(Vtx[0].position().z());
    ele_tip_pv = myElec->gsfTrack()->dxy(Vtx[0].position());
  } else {
    pv_x = -999999.;
    pv_y = -999999.;
    pv_z = -999999.;
    ele_tip_pv = -999999.;
  }
  if (VtxBS.size() > 0) {
    pvbs_x = Float_t(VtxBS[0].position().x());
    pvbs_y = Float_t(VtxBS[0].position().y());
    pvbs_z = Float_t(VtxBS[0].position().z());
    ele_tip_pvbs = myElec->gsfTrack()->dxy(VtxBS[0].position());
  } else {
    pvbs_x = -999999.;
    pvbs_y = -999999.;
    pvbs_z = -999999.;
    ele_tip_pvbs = -999999.;
  }

  //
  ele_gsfCharge    = (Int_t) myElec->gsfTrack()->charge();
  // must keep the ctf track collection, i.e. general track collection
  ele_ctfCharge    = (Int_t) myElec->closestCtfTrackRef().isNonnull() ? myElec->closestCtfTrackRef()->charge():-9999;
  ele_scPixCharge  = (Int_t) myElec->chargeInfo().scPixCharge;
  ele_eop          = (Float_t) myElec->eSuperClusterOverP();
  ele_tip_bs       = (Float_t) -myElec->dB();
  //ele_tip_pv       = myElec->userFloat("ele_tip_pv");
  ele_pin          = (Float_t)  myElec->trackMomentumAtVtx().R();
  ele_pout         = (Float_t)  myElec->trackMomentumOut().R();
  //
  event_caloMET    = (Float_t) myMet->et();
  event_pfMET      = (Float_t) myPfMet->et();
  event_tcMET      = (Float_t) myTcMet->et();
  event_caloMET_phi= (Float_t) myMet->phi();
  event_pfMET_phi  = (Float_t) myPfMet->phi();
  event_tcMET_phi  = (Float_t) myTcMet->phi();
  event_caloSumEt  = (Float_t) myMet->sumEt();
  event_pfSumEt  = (Float_t) myPfMet->sumEt();
  event_tcSumEt  = (Float_t) myTcMet->sumEt();
  // transverse mass for the user's convenience
  event_caloMT     = (Float_t) TMath::Sqrt(2.*(ele_sc_gsf_et*event_caloMET -
    (ele_sc_gsf_et*TMath::Cos(ele_sc_phi)*event_caloMET*TMath::Cos(event_caloMET_phi)
     + ele_sc_gsf_et*TMath::Sin(ele_sc_phi)*event_caloMET*TMath::Sin(event_caloMET_phi)
     ) )  );
  event_pfMT       = (Float_t) TMath::Sqrt(2.*(ele_sc_gsf_et*event_pfMET -
    (ele_sc_gsf_et*TMath::Cos(ele_sc_phi)*event_pfMET*TMath::Cos(event_pfMET_phi)
     + ele_sc_gsf_et*TMath::Sin(ele_sc_phi)*event_pfMET*TMath::Sin(event_pfMET_phi)
     ) )  );
  event_tcMT       = (Float_t) TMath::Sqrt(2.*(ele_sc_gsf_et*event_tcMET -
    (ele_sc_gsf_et*TMath::Cos(ele_sc_phi)*event_tcMET*TMath::Cos(event_tcMET_phi)
     + ele_sc_gsf_et*TMath::Sin(ele_sc_phi)*event_tcMET*TMath::Sin(event_tcMET_phi)
     ) )  );
  event_datasetTag = DatasetTag_;
  // jet information - only if the user asks for it
  // keep the 5 highest et jets of the event that are further than DR> DRJetFromElectron_
  if (includeJetInformationInNtuples_) {
    // initialize the array of the jet information
    for (int i=0; i<5; ++i) {
      calojet_et[i] = -999999;  calojet_eta[i] = -999999; calojet_phi[i] = -999999;
      pfjet_et[i] = -999999;    pfjet_eta[i] = -999999;   pfjet_phi[i] = -999999;
    }
    // get hold of the jet collections
    edm::Handle< reco::CaloJetCollection > pCaloJets;
    edm::Handle< reco::PFJetCollection > pPfJets;
    iEvent.getByToken(caloJetCollectionToken_, pCaloJets);
    iEvent.getByToken(pfJetCollectionToken_, pPfJets);
    //
    // calo jets now:
    if (pCaloJets.isValid()) {
      const  reco::CaloJetCollection  *caloJets =  pCaloJets.product();
      int nCaloJets = (int) caloJets->size();
      if (nCaloJets>0) {
	float *nCaloET = new float[nCaloJets];
	float *nCaloEta = new float[nCaloJets];
	float *nCaloPhi = new float[nCaloJets];
	reco::CaloJetCollection::const_iterator cjet = caloJets->begin();
	int counter = 0;
	for (; cjet != caloJets->end(); ++cjet) {
	  // store them only if they are far enough from the electron
	  Double_t DR = reco::deltaR(cjet->eta(), cjet->phi(), myElec->gsfTrack()->eta(), ele_sc_phi);
	  if (DR > DRJetFromElectron_) {
	    nCaloET[counter]  = cjet->et();
	    nCaloEta[counter] = cjet->eta();
	    nCaloPhi[counter] = cjet->phi();
	    ++counter;
	  }
	}
	int *caloJetSorted = new int[nCaloJets];
	TMath::Sort(nCaloJets, nCaloET, caloJetSorted, true);
	for (int i=0; i<nCaloJets; ++i) {
	  if (i>=5) break;
	  calojet_et[i] = nCaloET[ caloJetSorted[i] ];
	  calojet_eta[i] = nCaloEta[ caloJetSorted[i] ];
	  calojet_phi[i] = nCaloPhi[ caloJetSorted[i] ];
	}
	delete [] caloJetSorted;
	delete [] nCaloET;
	delete [] nCaloEta;
	delete [] nCaloPhi;
      }
    } else {
      std::cout << "WenuPlots: Could not get caloJet collection with name "
		<< caloJetCollectionTag_ << std::endl;
    }
    //
    // pf jets now:
    if (pPfJets.isValid()) {
      const  reco::PFJetCollection  *pfJets =  pPfJets.product();
      int nPfJets = (int) pfJets->size();
      if (nPfJets>0) {
	float *nPfET  = new float[nPfJets];
	float *nPfEta = new float[nPfJets];
	float *nPfPhi = new float[nPfJets];
	reco::PFJetCollection::const_iterator pjet = pfJets->begin();
	int counter = 0;
	for (; pjet != pfJets->end(); ++pjet) {
	  // store them only if they are far enough from the electron
	  Double_t DR = reco::deltaR(pjet->eta(), pjet->phi(), myElec->gsfTrack()->eta(), ele_sc_phi);
	  if (DR > DRJetFromElectron_) {
	    nPfET[counter]  = pjet->et();
	    nPfEta[counter] = pjet->eta();
	    nPfPhi[counter] = pjet->phi();
	    ++counter;
	  }
	}
	int *pfJetSorted = new int[nPfJets];
	TMath::Sort(nPfJets, nPfET, pfJetSorted, true);
	for (int i=0; i<nPfJets; ++i) {
	  if (i>=5) break;
	  pfjet_et[i]  = nPfET[ pfJetSorted[i] ];
	  pfjet_eta[i] = nPfEta[ pfJetSorted[i] ];
	  pfjet_phi[i] = nPfPhi[ pfJetSorted[i] ];
	}
	delete [] pfJetSorted;
	delete [] nPfET;
	delete [] nPfEta;
	delete [] nPfPhi;
      }
    } else {
      std::cout << "WenuPlots: Could not get pfJet collection with name "
		<< pfJetCollectionTag_ << std::endl;
    }

  }
  // second electron information - in preselected ntuple only
  ele2nd_sc_gsf_et = -1; // also in sele tree
  ele2nd_sc_eta    = -1;
  ele2nd_sc_phi    = -1;
  ele2nd_sc_rho    =  0;
  ele2nd_cand_eta  =  0;
  ele2nd_cand_phi  =  0;
  ele2nd_cand_et   =  0;
  ele2nd_pin       =  0;
  ele2nd_pout      =  0;
  ele2nd_passes_selection = -1; // also in sele tree
  ele2nd_ecalDriven=  0;
  //
  // second electron selection variables: only if requested by the user
  //
  ele2nd_iso_track = 0;
  ele2nd_iso_ecal  = 0;
  ele2nd_iso_hcal  = 0;
  //
  ele2nd_id_sihih  = 0;
  ele2nd_id_deta   = 0;
  ele2nd_id_dphi   = 0;
  ele2nd_id_hoe    = 0;
  //
  ele2nd_cr_mhitsinner = 0;
  ele2nd_cr_dcot       = 0;
  ele2nd_cr_dist       = 0;
  //
  ele2nd_vx     = 0;
  ele2nd_vy     = 0;
  ele2nd_vz     = 0;
  //
  ele2nd_gsfCharge   = 0;
  // must keep the ctf track collection, i.e. general track collection
  ele2nd_ctfCharge   = 0;
  ele2nd_scPixCharge = 0;
  ele2nd_eop         = 0;
  ele2nd_tip_bs      = 0;
  ele2nd_tip_pv      = 0;
  ele2nd_hltmatched_dr = 0;
  //
  // convention for ele2nd_passes_selection
  // 0 passes no selection
  // 1 passes WP95
  // 2 passes WP90
  // 3 passes WP85
  // 4 passes WP80
  // 5 passes WP70
  // 6 passes WP60
  if (myElec->userInt("hasSecondElectron") == 1 && storeExtraInformation_) {
    const pat::Electron * mySecondElec=
      dynamic_cast<const pat::Electron*> (wenu.daughter("secondElec"));
    ele2nd_sc_gsf_et = (Float_t) mySecondElec->superCluster()->energy()/TMath::CosH(mySecondElec->gsfTrack()->eta());

    ele2nd_sc_eta    = (Float_t) mySecondElec->superCluster()->eta();
    ele2nd_sc_phi    = (Float_t) mySecondElec->superCluster()->phi();
    double sc2x = mySecondElec->superCluster()->x();
    double sc2y = mySecondElec->superCluster()->y();
    double sc2z = mySecondElec->superCluster()->z();
    ele2nd_sc_rho    = (Float_t) sqrt(sc2x*sc2x + sc2y*sc2y + sc2z*sc2z);
    ele2nd_cand_eta  = (Float_t) mySecondElec->eta();
    ele2nd_cand_phi  = (Float_t) mySecondElec->phi();
    ele2nd_cand_et   = (Float_t) mySecondElec->et();
    ele2nd_pin       = (Float_t) mySecondElec->trackMomentumAtVtx().R();;
    ele2nd_pout      = (Float_t) mySecondElec->trackMomentumOut().R();
    ele2nd_ecalDriven= (Int_t)   mySecondElec->ecalDrivenSeed();
    // check the selections
    bool isIDCalc = mySecondElec->isElectronIDAvailable("simpleEleId95relIso") &&
      mySecondElec->isElectronIDAvailable("simpleEleId90relIso") &&
      mySecondElec->isElectronIDAvailable("simpleEleId85relIso") &&
      mySecondElec->isElectronIDAvailable("simpleEleId80relIso") &&
      mySecondElec->isElectronIDAvailable("simpleEleId70relIso") &&
      mySecondElec->isElectronIDAvailable("simpleEleId60relIso");
    if (isIDCalc) {
      ele2nd_passes_selection = 0;
      if (fabs(mySecondElec->electronID("simpleEleId60relIso")-7) < 0.1) ele2nd_passes_selection = 6;
      else if (fabs(mySecondElec->electronID("simpleEleId70relIso")-7) < 0.1) ele2nd_passes_selection = 5;
      else if (fabs(mySecondElec->electronID("simpleEleId80relIso")-7) < 0.1) ele2nd_passes_selection = 4;
      else if (fabs(mySecondElec->electronID("simpleEleId85relIso")-7) < 0.1) ele2nd_passes_selection = 3;
      else if (fabs(mySecondElec->electronID("simpleEleId90relIso")-7) < 0.1) ele2nd_passes_selection = 2;
      else if (fabs(mySecondElec->electronID("simpleEleId95relIso")-7) < 0.1) ele2nd_passes_selection = 1;
    }
    if (storeAllSecondElectronVariables_) {
      ele2nd_iso_track = (Float_t)  mySecondElec->dr03IsolationVariables().tkSumPt / ele2nd_cand_et;
      ele2nd_iso_ecal  = (Float_t)  mySecondElec->dr03IsolationVariables().ecalRecHitSumEt/ele2nd_cand_et;
      ele2nd_iso_hcal  = (Float_t)  ( mySecondElec->dr03IsolationVariables().hcalDepth1TowerSumEt +
				      mySecondElec->dr03IsolationVariables().hcalDepth2TowerSumEt) / ele2nd_cand_et;
      ele2nd_id_sihih  = (Float_t)  mySecondElec->sigmaIetaIeta();
      ele2nd_id_deta   = (Float_t)  mySecondElec->deltaEtaSuperClusterTrackAtVtx();
      ele2nd_id_dphi   = (Float_t)  mySecondElec->deltaPhiSuperClusterTrackAtVtx();
      ele2nd_id_hoe    = (Float_t)  mySecondElec->hadronicOverEm();

      ele2nd_cr_mhitsinner = mySecondElec->gsfTrack()->trackerExpectedHitsInner().numberOfHits();
      ele2nd_cr_dcot       = mySecondElec->convDcot();
      ele2nd_cr_dist       = mySecondElec->convDist();

      ele2nd_vx        = (Float_t) mySecondElec->vx();
      ele2nd_vy        = (Float_t) mySecondElec->vy();
      ele2nd_vz        = (Float_t) mySecondElec->vz();
      ele2nd_gsfCharge   = (Int_t) mySecondElec->gsfTrack()->charge();
      // must keep the ctf track collection, i.e. general track collection
      ele2nd_ctfCharge   = (Int_t) mySecondElec->closestCtfTrackRef().isNonnull() ? mySecondElec->closestCtfTrackRef()->charge():-9999;
      ele2nd_scPixCharge = (Int_t) mySecondElec->chargeInfo().scPixCharge;
      ele2nd_eop         = (Float_t) mySecondElec->eSuperClusterOverP();
      ele2nd_tip_bs      = (Float_t) -mySecondElec->dB();
      if (Vtx.size() > 0) {
	ele2nd_tip_pv      =   mySecondElec->gsfTrack()->dxy(Vtx[0].position());
      }
      if (VtxBS.size() > 0) {
	ele2nd_tip_pvbs      =   mySecondElec->gsfTrack()->dxy(VtxBS[0].position());
      }
      ele2nd_hltmatched_dr = mySecondElec->userFloat("HLTMatchingDR");
    }
  }
  // some extra information
  event_triggerDecision = -1;
  ele_hltmatched_dr = -999.;
  VtxTracksSize.clear();
  VtxNormalizedChi2.clear();
  VtxTracksSizeBS.clear();
  VtxNormalizedChi2BS.clear();
  if (storeExtraInformation_) {
    if (myElec->hasUserFloat("HLTMatchingDR")) {
      ele_hltmatched_dr = myElec->userFloat("HLTMatchingDR");
    }
    if (myElec->hasUserInt("triggerDecision")) {
      event_triggerDecision = myElec->userInt("triggerDecision");
    }
    // extra information related to the primary vtx collection
    for (Int_t i=0; i < (Int_t) Vtx.size(); ++i) {
      VtxTracksSize.push_back(Vtx[i].tracksSize());
      VtxNormalizedChi2.push_back(Vtx[i].normalizedChi2());
    }
    for (Int_t i=0; i < (Int_t) VtxBS.size(); ++i) {
      VtxTracksSizeBS.push_back(VtxBS[i].tracksSize());
      VtxNormalizedChi2BS.push_back(VtxBS[i].normalizedChi2());
    }
  }
  // if the electron passes the selection
  // it is meant to be a precalculated selection here, in order to include
  // conversion rejection too
  if (CheckCuts(myElec) && myElec->userInt("failsSecondElectronCut") == 0) {
    vbtfSele_tree->Fill();
  }
  vbtfPresele_tree->Fill();



  //
  // _______________________________________________________________________
  //
  // histogram production --------------------------------------------------
  // _______________________________________________________________________
  //
  // if you want some preselection: Conv rejection, hit pattern
  if (usePreselection_) {
    if (not PassPreselectionCriteria(myElec)) return;
  }
  //
  // some variables here
  Double_t scEta = myElec->superCluster()->eta();
  Double_t scPhi = myElec->superCluster()->phi();
  Double_t scEt = myElec->superCluster()->energy()/TMath::CosH(scEta);
  Double_t met    = myMet->et();
  Double_t metPhi = myMet->phi();
  Double_t mt  = TMath::Sqrt(2.0*scEt*met*(1.0-(TMath::Cos(scPhi)*TMath::Cos(metPhi)+TMath::Sin(scPhi)*TMath::Sin(metPhi))));

  Double_t trackIso = myElec->userIsolation(pat::TrackIso);
  Double_t ecalIso = myElec->userIsolation(pat::EcalIso);
  Double_t hcalIso = myElec->userIsolation(pat::HcalIso);
  Double_t sihih = myElec->scSigmaIEtaIEta();
  Double_t dphi = myElec->deltaPhiSuperClusterTrackAtVtx();
  Double_t deta = myElec->deltaEtaSuperClusterTrackAtVtx();
  Double_t HoE = myElec->hadronicOverEm();
  //
  //
  //
  // the inverted selection plots:
  // only if not using precalcID
  if (not usePrecalcID_) {
    if (CheckCutsInverse(myElec)){
      //std::cout << "-----------------INVERSION-----------passed" << std::endl;
      h_met_inverse->Fill(met);
      h_mt_inverse->Fill(mt);
      if(TMath::Abs(scEta)<1.479){
	h_met_inverse_EB->Fill(met);
	h_mt_inverse_EB->Fill(mt);
      }
      if(TMath::Abs(scEta)>1.479){
	h_met_inverse_EE->Fill(met);
	h_mt_inverse_EE->Fill(mt);
      }
    }
  }
  //
  ///////////////////////////////////////////////////////////////////////
  //
  // N-1 plots: plot some variable so that all the other cuts are satisfied
  //
  // make these plots only if you have the normal selection, not pre-calced
  if (not usePrecalcID_) {
    if ( TMath::Abs(scEta) < 1.479) { // reminder: the precise fiducial cuts are in
      // in the filter
      if (CheckCutsNminusOne(myElec, 0))
	h_trackIso_eb_NmOne->Fill(trackIso);
    }
    else {
      if (CheckCutsNminusOne(myElec, 0))
	h_trackIso_ee_NmOne->Fill(trackIso);
    }
  }
  //
  // SELECTION APPLICATION
  //
  // from here on you have only events that pass the full selection
  if (not CheckCuts(myElec)) return;
  //////////////////////////////////////////////////////////////////////

  h_met->Fill(met);
  h_mt->Fill(mt);
  if(TMath::Abs(scEta)<1.479){
    h_met_EB->Fill(met);
    h_mt_EB->Fill(mt);

    h_EB_trkiso->Fill( trackIso );
    h_EB_ecaliso->Fill( ecalIso );
    h_EB_hcaliso->Fill( hcalIso );
    h_EB_sIetaIeta->Fill( sihih );
    h_EB_dphi->Fill( dphi );
    h_EB_deta->Fill( deta );
    h_EB_HoE->Fill( HoE );

  }
  if(TMath::Abs(scEta)>1.479){
    h_met_EE->Fill(met);
    h_mt_EE->Fill(mt);

    h_EE_trkiso->Fill( trackIso );
    h_EE_ecaliso->Fill( ecalIso );
    h_EE_hcaliso->Fill( hcalIso );
    h_EE_sIetaIeta->Fill( sihih );
    h_EE_dphi->Fill( dphi );
    h_EE_deta->Fill( deta );
    h_EE_HoE->Fill( HoE );

  }
  // uncomment for debugging purposes
  /*
  std::cout << "tracIso: " <<  trackIso << ", " << myElec->trackIso() << ", ecaliso: " << ecalIso
	    << ", " << myElec->ecalIso() << ", hcaliso: " << hcalIso << ", "  << myElec->hcalIso()
	    << ", mishits: "
	    << myElec->gsfTrack()->trackerExpectedHitsInner().numberOfHits()
	    << std::endl;
  std::cout << "Electron ID: 95relIso=" << myElec->electronID("simpleEleId95relIso")
	    << " 90relIso=" << myElec->electronID("simpleEleId90relIso")
	    << " 85relIso=" << myElec->electronID("simpleEleId85relIso")
	    << " 80relIso=" << myElec->electronID("simpleEleId80relIso")
	    << " 70relIso=" << myElec->electronID("simpleEleId70relIso")
	    << " 60relIso=" << myElec->electronID("simpleEleId60relIso")
	    << " 95cIso=" << myElec->electronID("simpleEleId95cIso")
	    << " 90cIso=" << myElec->electronID("simpleEleId90cIso")
	    << " 85cIso=" << myElec->electronID("simpleEleId85cIso")
	    << " 80cIso=" << myElec->electronID("simpleEleId80cIso")
	    << " 70cIso=" << myElec->electronID("simpleEleId70cIso")
	    << " 60cIso=" << myElec->electronID("simpleEleId60cIso")
	    << std::endl;
  std::cout << "mySelection: " << (CheckCuts(myElec) && PassPreselectionCriteria(myElec)) << endl;
  */
  h_scEt->Fill(scEt);
  h_scEta->Fill(scEta);
  h_scPhi->Fill(scPhi);

}

/***********************************************************************
 *
 *  Checking Cuts and making selections:
 *  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 *  all the available methods take input a pointer to a  pat::Electron
 *
 *  Bool_t  CheckCuts(const pat::Electron *):
 *                            true if the input selection is satisfied
 *  Bool_t  CheckCutsInverse(const pat::Electron *ele):
 *            true if the cuts with inverted the ones specified in the
 *            cfg are satisfied
 *  Bool_t  CheckCutsNminusOne(const pat::Electron *ele, int jj):
 *             true if all the cuts with cut #jj ignored are satisfied
 *
 ***********************************************************************/
Bool_t WenuPlots::CheckCuts( const pat::Electron *ele)
{
  if (usePrecalcID_) {
    if (not ele-> isElectronIDAvailable(usePrecalcIDType_)) {
      std::cout << "Error! not existing ID with name: "
		<< usePrecalcIDType_ << " function will return true!"
		<< std::endl;
      return true;
    }
    Double_t val = ele->electronID(usePrecalcIDType_);
    if (usePrecalcIDSign_ == "<") {
      return val < usePrecalcIDValue_;
    }
    else if (usePrecalcIDSign_ == ">") {
      return val > usePrecalcIDValue_;
    }
    else { // equality: it returns 0,1,2,3 but as float
      return TMath::Abs(val-usePrecalcIDValue_)<0.1;
    }
  }
  else {
    for (int i=0; i<nBarrelVars_; ++i) {
      if (not CheckCut(ele, i)) return false;
    }
    return true;
  }
}
/////////////////////////////////////////////////////////////////////////

Bool_t WenuPlots::CheckCutsInverse(const pat::Electron *ele)
{
  for (int i=0; i<nBarrelVars_; ++i){
    if ( CheckCutInv(ele, i) == false) return false;
  }
  return true;

}
/////////////////////////////////////////////////////////////////////////
Bool_t WenuPlots::CheckCutsNminusOne(const pat::Electron *ele, int jj)
{
  for (int i=0; i<nBarrelVars_; ++i){
    if (i==jj) continue;
    if ( CheckCut(ele, i) == false) return false;
  }
  return true;
}
/////////////////////////////////////////////////////////////////////////
Bool_t WenuPlots::CheckCut(const pat::Electron *ele, int i) {
  Double_t fabseta = TMath::Abs(ele->superCluster()->eta());
  if ( fabseta<1.479) {
    return TMath::Abs(ReturnCandVar(ele, i)) < CutVars_[i];
  }
  return TMath::Abs(ReturnCandVar(ele, i)) < CutVars_[i+nBarrelVars_];
}
/////////////////////////////////////////////////////////////////////////
Bool_t WenuPlots::CheckCutInv(const pat::Electron *ele, int i) {
  Double_t fabseta = TMath::Abs(ele->superCluster()->eta());
  if ( fabseta<1.479) {
    if (InvVars_[i])
    return TMath::Abs(ReturnCandVar(ele, i))>CutVars_[i];
    return TMath::Abs(ReturnCandVar(ele, i)) < CutVars_[i];
  }
  if (InvVars_[i+nBarrelVars_]) {
    if (InvVars_[i])
      return TMath::Abs(ReturnCandVar(ele, i))>CutVars_[i+nBarrelVars_];
  }
  return TMath::Abs(ReturnCandVar(ele, i)) < CutVars_[i+nBarrelVars_];
}
////////////////////////////////////////////////////////////////////////
Double_t WenuPlots::ReturnCandVar(const pat::Electron *ele, int i) {
  if (i==0) return ele->dr03TkSumPt()/ele->p4().Pt();
  else if (i==1) return ele->dr03EcalRecHitSumEt()/ele->p4().Pt();
  else if (i==2) return ele->dr03HcalTowerSumEt()/ele->p4().Pt();
  else if (i==3) return ele->scSigmaIEtaIEta();
  else if (i==4) return ele->deltaPhiSuperClusterTrackAtVtx();
  else if (i==5) return ele->deltaEtaSuperClusterTrackAtVtx();
  else if (i==6) return ele->hadronicOverEm();
  else if (i==7) {
    if (ele->isEB()){
      return ( ele->dr03TkSumPt()+std::max(float(0.),ele->dr03EcalRecHitSumEt()-1)
	       + ele->dr03HcalTowerSumEt())/ele->p4().Pt(); }
    else { // pedestal subtraction is only in barrel
      return ( ele->dr03TkSumPt()+ele->dr03EcalRecHitSumEt()
	       + ele->dr03HcalTowerSumEt())/ele->p4().Pt(); }
  }
  //  else if (i==8) return ele->gsfTrack()->dxy(bspotPosition_);
  else if (i==8) return fabs(ele->dB());
  else if (i==9) return ele->eSuperClusterOverP();
  else if (i==10) return ele->userIsolation(pat::TrackIso);
  else if (i==11) return ele->userIsolation(pat::EcalIso);
  else if (i==12) return ele->userIsolation(pat::HcalIso);
  std::cout << "Error in WenuPlots::ReturnCandVar" << std::endl;
  return -1.;

}
/////////////////////////////////////////////////////////////////////////
Bool_t WenuPlots::PassPreselectionCriteria(const pat::Electron *ele) {
  Bool_t passConvRej = true;
  Bool_t passPXB = true;
  Bool_t passEMH = true;
  if (useConversionRejection_) {
    if (ele->hasUserInt("PassConversionRejection")) {
      //std::cout << "con rej: " << ele->userInt("PassConversionRejection") << std::endl;
      if (not (ele->userInt("PassConversionRejection")==1)) passConvRej = false;
    }
    else {
      std::cout << "WenuPlots: WARNING: Conversion Rejection Request Disregarded: "
		<< "you must calculate it before " << std::endl;
      // return true;
    }
  }
  if (useValidFirstPXBHit_) {
    if (ele->hasUserInt("PassValidFirstPXBHit")) {
      //std::cout << "valid1stPXB: " << ele->userInt("PassValidFirstPXBHit") << std::endl;
      if (not (ele->userInt("PassValidFirstPXBHit")==1)) passPXB = false;
    }
    else {
      std::cout << "WenuPlots: WARNING: Valid First PXB Hit Request Disregarded: "
                << "you must calculate it before " << std::endl;
      // return true;
    }
  }
  if (useExpectedMissingHits_) {
    if (ele->hasUserInt("NumberOfExpectedMissingHits")) {
      //std::cout << "missing hits: " << ele->userInt("NumberOfExpectedMissingHits") << std::endl;
      if (ele->userInt("NumberOfExpectedMissingHits")>maxNumberOfExpectedMissingHits_)
	passEMH = false;
    }
    else {
      std::cout << "WenuPlots: WARNING: Number of Expected Missing Hits Request Disregarded: "
                << "you must calculate it before " << std::endl;
      // return true;
    }
  }
  return passConvRej && passPXB && passEMH;
}
// ------------ method called once each job just before starting event loop  --
void
WenuPlots::beginJob()
{
  //std::cout << "In beginJob()" << std::endl;
  //  Double_t Pi = TMath::Pi();
  //  TString histo_file = outputFile_;
  //  histofile = new TFile( histo_file,"RECREATE");

  h_met         = new TH1F("h_met",         "h_met",         200, 0, 200);
  h_met_inverse = new TH1F("h_met_inverse", "h_met_inverse", 200, 0, 200);

  h_mt         = new TH1F("h_mt",         "h_mt",         200, 0, 200);
  h_mt_inverse = new TH1F("h_mt_inverse", "h_mt_inverse", 200, 0, 200);


  h_met_EB         = new TH1F("h_met_EB",         "h_met_EB",         200, 0, 200);
  h_met_inverse_EB = new TH1F("h_met_inverse_EB", "h_met_inverse_EB", 200, 0, 200);

  h_mt_EB         = new TH1F("h_mt_EB",         "h_mt_EB",         200, 0, 200);
  h_mt_inverse_EB = new TH1F("h_mt_inverse_EB", "h_mt_inverse_EB", 200, 0, 200);


  h_met_EE         = new TH1F("h_met_EE",         "h_met_EE",         200, 0, 200);
  h_met_inverse_EE = new TH1F("h_met_inverse_EE", "h_met_inverse_EE", 200, 0, 200);

  h_mt_EE         = new TH1F("h_mt_EE",         "h_mt_EE",         200, 0, 200);
  h_mt_inverse_EE = new TH1F("h_mt_inverse_EE", "h_mt_inverse_EE", 200, 0, 200);


  h_scEt  = new TH1F("h_scEt",  "h_scEt",  200,  0, 100);
  h_scEta = new TH1F("h_scEta", "h_scEta", 200, -3, 3);
  h_scPhi = new TH1F("h_scPhi", "h_scPhi", 200, -4, 4);


  //VALIDATION PLOTS
  //EB
  h_EB_trkiso = new TH1F("h_EB_trkiso","h_EB_trkiso",200 , 0.0, 9.0);
  h_EB_ecaliso = new TH1F("h_EB_ecaliso","h_EB_ecaliso",200, 0.0 , 9.0);
  h_EB_hcaliso = new TH1F("h_EB_hcaliso","h_EB_hcaliso",200, 0.0 , 9.0);
  h_EB_sIetaIeta = new TH1F("h_EB_sIetaIeta","h_EB_sIetaIeta",200, 0.0 , 0.02 );
  h_EB_dphi = new TH1F("h_EB_dphi","h_EB_dphi",200, -0.03 , 0.03 );
  h_EB_deta = new TH1F("h_EB_deta","h_EB_deta",200, -0.01 , 0.01) ;
  h_EB_HoE = new TH1F("h_EB_HoE","h_EB_HoE",200, 0.0 , 0.2 );
  //EE
  h_EE_trkiso = new TH1F("h_EE_trkiso","h_EE_trkiso",200 , 0.0, 9.0);
  h_EE_ecaliso = new TH1F("h_EE_ecaliso","h_EE_ecaliso",200, 0.0 , 9.0);
  h_EE_hcaliso = new TH1F("h_EE_hcaliso","h_EE_hcaliso",200, 0.0 , 9.0);
  h_EE_sIetaIeta = new TH1F("h_EE_sIetaIeta","h_EE_sIetaIeta",200, 0.0 , 0.1 );
  h_EE_dphi = new TH1F("h_EE_dphi","h_EE_dphi",200, -0.03 , 0.03 );
  h_EE_deta = new TH1F("h_EE_deta","h_EE_deta",200, -0.01 , 0.01) ;
  h_EE_HoE = new TH1F("h_EE_HoE","h_EE_HoE",200, 0.0 , 0.2 );


  //
  //
  h_trackIso_eb_NmOne =
    new TH1F("h_trackIso_eb_NmOne","trackIso EB N-1 plot",80,0,8);
  h_trackIso_ee_NmOne =
    new TH1F("h_trackIso_ee_NmOne","trackIso EE N-1 plot",80,0,8);


  // if you add some new variable change the nBarrelVars_ accordingly
  // reminder: in the current implementation you must have the same number
  //  of vars in both barrel and endcaps
  nBarrelVars_ = 13;
  //
  // Put EB variables together and EE variables together
  // number of barrel variables = number of endcap variable
  // if you don't want to use some variable put a very high cut
  CutVars_.push_back( trackIso_EB_ );//0
  CutVars_.push_back( ecalIso_EB_ ); //1
  CutVars_.push_back( hcalIso_EB_ ); //2
  CutVars_.push_back( sihih_EB_ );   //3
  CutVars_.push_back( dphi_EB_ );    //4
  CutVars_.push_back( deta_EB_ );    //5
  CutVars_.push_back( hoe_EB_ );     //6
  CutVars_.push_back( cIso_EB_ );    //7
  CutVars_.push_back( tip_bspot_EB_);//8
  CutVars_.push_back( eop_EB_ );     //9
  CutVars_.push_back( trackIsoUser_EB_ );//10
  CutVars_.push_back( ecalIsoUser_EB_  );//11
  CutVars_.push_back( hcalIsoUser_EB_  );//12
  //
  CutVars_.push_back( trackIso_EE_);//0
  CutVars_.push_back( ecalIso_EE_); //1
  CutVars_.push_back( hcalIso_EE_); //2
  CutVars_.push_back( sihih_EE_);   //3
  CutVars_.push_back( dphi_EE_);    //4
  CutVars_.push_back( deta_EE_);    //5
  CutVars_.push_back( hoe_EE_ );    //6
  CutVars_.push_back( cIso_EE_ );   //7
  CutVars_.push_back(tip_bspot_EE_);//8
  CutVars_.push_back( eop_EE_ );    //9
  CutVars_.push_back( trackIsoUser_EE_ );//10
  CutVars_.push_back( ecalIsoUser_EE_  );//11
  CutVars_.push_back( hcalIsoUser_EE_  );//12
  //
  InvVars_.push_back( trackIso_EB_inv);//0
  InvVars_.push_back( ecalIso_EB_inv); //1
  InvVars_.push_back( hcalIso_EB_inv); //2
  InvVars_.push_back( sihih_EB_inv);   //3
  InvVars_.push_back( dphi_EB_inv);    //4
  InvVars_.push_back( deta_EB_inv);    //5
  InvVars_.push_back( hoe_EB_inv);     //6
  InvVars_.push_back( cIso_EB_inv);    //7
  InvVars_.push_back(tip_bspot_EB_inv);//8
  InvVars_.push_back( eop_EB_inv);     //9
  InvVars_.push_back( trackIsoUser_EB_inv );//10
  InvVars_.push_back( ecalIsoUser_EB_inv  );//11
  InvVars_.push_back( hcalIsoUser_EB_inv  );//12
  //
  InvVars_.push_back( trackIso_EE_inv);//0
  InvVars_.push_back( ecalIso_EE_inv); //1
  InvVars_.push_back( hcalIso_EE_inv); //2
  InvVars_.push_back( sihih_EE_inv);   //3
  InvVars_.push_back( dphi_EE_inv);    //4
  InvVars_.push_back( deta_EE_inv);    //5
  InvVars_.push_back( hoe_EE_inv);     //6
  InvVars_.push_back( cIso_EE_inv);    //7
  InvVars_.push_back(tip_bspot_EE_inv);//8
  InvVars_.push_back( eop_EE_inv);     //9
  InvVars_.push_back( trackIsoUser_EE_inv );//10
  InvVars_.push_back( ecalIsoUser_EE_inv  );//11
  InvVars_.push_back( hcalIsoUser_EE_inv  );//12
  //
  //
  // ________________________________________________________________________
  //
  // The VBTF Root Tuples ---------------------------------------------------
  // ________________________________________________________________________
  //
  WENU_VBTFselectionFile_ = new TFile(TString(WENU_VBTFselectionFileName_),
				      "RECREATE");

  vbtfSele_tree = new TTree("vbtfSele_tree",
	       "Tree to store the W Candidates that pass the VBTF selection");
  vbtfSele_tree->Branch("runNumber", &runNumber, "runNumber/I");
  vbtfSele_tree->Branch("eventNumber", &eventNumber, "eventNumber/L");
  vbtfSele_tree->Branch("lumiSection", &lumiSection, "lumiSection/I");
  //
  vbtfSele_tree->Branch("ele_sc_gsf_et", &ele_sc_gsf_et,"ele_sc_gsf_et/F");
  vbtfSele_tree->Branch("ele_sc_energy", &ele_sc_energy,"ele_sc_energy/F");
  vbtfSele_tree->Branch("ele_sc_eta", &ele_sc_eta,"ele_sc_eta/F");
  vbtfSele_tree->Branch("ele_sc_phi", &ele_sc_phi,"ele_sc_phi/F");
  vbtfSele_tree->Branch("ele_sc_rho", &ele_sc_rho,"ele_sc_rho/F");
  vbtfSele_tree->Branch("ele_cand_et", &ele_cand_et, "ele_cand_et/F");
  vbtfSele_tree->Branch("ele_cand_eta", &ele_cand_eta,"ele_cand_eta/F");
  vbtfSele_tree->Branch("ele_cand_phi",&ele_cand_phi,"ele_cand_phi/F");
  vbtfSele_tree->Branch("ele_iso_track",&ele_iso_track,"ele_iso_track/F");
  vbtfSele_tree->Branch("ele_iso_ecal",&ele_iso_ecal,"ele_iso_ecal/F");
  vbtfSele_tree->Branch("ele_iso_hcal",&ele_iso_hcal,"ele_iso_hcal/F");
  vbtfSele_tree->Branch("ele_id_sihih",&ele_id_sihih,"ele_id_sihih/F");
  vbtfSele_tree->Branch("ele_id_deta",&ele_id_deta,"ele_id_deta/F");
  vbtfSele_tree->Branch("ele_id_dphi",&ele_id_dphi,"ele_id_dphi/F");
  vbtfSele_tree->Branch("ele_id_hoe",&ele_id_hoe,"ele_id_hoe/F");
  vbtfSele_tree->Branch("ele_cr_mhitsinner",&ele_cr_mhitsinner,"ele_cr_mhitsinner/I");
  vbtfSele_tree->Branch("ele_cr_dcot",&ele_cr_dcot,"ele_cr_dcot/F");
  vbtfSele_tree->Branch("ele_cr_dist",&ele_cr_dist,"ele_cr_dist/F");
  vbtfSele_tree->Branch("ele_vx",&ele_vx,"ele_vx/F");
  vbtfSele_tree->Branch("ele_vy",&ele_vy,"ele_vy/F");
  vbtfSele_tree->Branch("ele_vz",&ele_vz,"ele_vz/F");
  vbtfSele_tree->Branch("pv_x",&pv_x,"pv_x/F");
  vbtfSele_tree->Branch("pv_y",&pv_y,"pv_y/F");
  vbtfSele_tree->Branch("pv_z",&pv_z,"pv_z/F");
  vbtfSele_tree->Branch("ele_gsfCharge",&ele_gsfCharge,"ele_gsfCharge/I");
  vbtfSele_tree->Branch("ele_ctfCharge",&ele_ctfCharge,"ele_ctfCharge/I");
  vbtfSele_tree->Branch("ele_scPixCharge",&ele_scPixCharge,"ele_scPixCharge/I");
  vbtfSele_tree->Branch("ele_eop",&ele_eop,"ele_eop/F");
  vbtfSele_tree->Branch("ele_tip_bs",&ele_tip_bs,"ele_tip_bs/F");
  vbtfSele_tree->Branch("ele_tip_pv",&ele_tip_pv,"ele_tip_pv/F");
  vbtfSele_tree->Branch("ele_pin",&ele_pin,"ele_pin/F");
  vbtfSele_tree->Branch("ele_pout",&ele_pout,"ele_pout/F");
  vbtfSele_tree->Branch("event_caloMET",&event_caloMET,"event_caloMET/F");
  vbtfSele_tree->Branch("event_pfMET",&event_pfMET,"event_pfMET/F");
  vbtfSele_tree->Branch("event_tcMET",&event_tcMET,"event_tcMET/F");
  vbtfSele_tree->Branch("event_caloMT",&event_caloMT,"event_caloMT/F");
  vbtfSele_tree->Branch("event_pfMT",&event_pfMT,"event_pfMT/F");
  vbtfSele_tree->Branch("event_tcMT",&event_tcMT,"event_tcMT/F");
  vbtfSele_tree->Branch("event_caloMET_phi",&event_caloMET_phi,"event_caloMET_phi/F");
  vbtfSele_tree->Branch("event_pfMET_phi",&event_pfMET_phi,"event_pfMET_phi/F");
  vbtfSele_tree->Branch("event_tcMET_phi",&event_tcMET_phi,"event_tcMET_phi/F");
  //
  // the extra jet variables:
  if (includeJetInformationInNtuples_) {
    vbtfSele_tree->Branch("calojet_et",calojet_et,"calojet_et[5]/F");
    vbtfSele_tree->Branch("calojet_eta",calojet_eta,"calojet_eta[5]/F");
    vbtfSele_tree->Branch("calojet_phi",calojet_phi,"calojet_phi[5]/F");
    vbtfSele_tree->Branch("pfjet_et",pfjet_et,"pfjet_et[5]/F");
    vbtfSele_tree->Branch("pfjet_eta",pfjet_eta,"pfjet_eta[5]/F");
    vbtfSele_tree->Branch("pfjet_phi",pfjet_phi,"pfjet_phi[5]/F");
  }
  if (storeExtraInformation_) {
    vbtfSele_tree->Branch("ele2nd_sc_gsf_et", &ele2nd_sc_gsf_et,"ele2nd_sc_gsf_et/F");
    vbtfSele_tree->Branch("ele2nd_passes_selection", &ele2nd_passes_selection,"ele2nd_passes_selection/I");
    vbtfSele_tree->Branch("ele2nd_ecalDriven",&ele2nd_ecalDriven,"ele2nd_ecalDriven/I");
    vbtfSele_tree->Branch("event_caloSumEt",&event_caloSumEt,"event_caloSumEt/F");
    vbtfSele_tree->Branch("event_pfSumEt",&event_pfSumEt,"event_pfSumEt/F");
    vbtfSele_tree->Branch("event_tcSumEt",&event_tcSumEt,"event_tcSumEt/F");
  }
  vbtfSele_tree->Branch("event_datasetTag",&event_datasetTag,"event_dataSetTag/I");
  //
  //
  // everything after preselection
  //
  WENU_VBTFpreseleFile_ = new TFile(TString(WENU_VBTFpreseleFileName_),
				    "RECREATE");

  vbtfPresele_tree = new TTree("vbtfPresele_tree",
	    "Tree to store the W Candidates that pass the VBTF preselection");
  vbtfPresele_tree->Branch("runNumber", &runNumber, "runNumber/I");
  vbtfPresele_tree->Branch("eventNumber", &eventNumber, "eventNumber/L");
  vbtfPresele_tree->Branch("lumiSection", &lumiSection, "lumiSection/I");
  //
  vbtfPresele_tree->Branch("ele_sc_gsf_et", &ele_sc_gsf_et,"ele_sc_gsf_et/F");
  vbtfPresele_tree->Branch("ele_sc_energy", &ele_sc_energy,"ele_sc_energy/F");
  vbtfPresele_tree->Branch("ele_sc_eta", &ele_sc_eta,"ele_sc_eta/F");
  vbtfPresele_tree->Branch("ele_sc_phi", &ele_sc_phi,"ele_sc_phi/F");
  vbtfPresele_tree->Branch("ele_sc_rho", &ele_sc_rho,"ele_sc_rho/F");
  vbtfPresele_tree->Branch("ele_cand_et", &ele_cand_et, "ele_cand_et/F");
  vbtfPresele_tree->Branch("ele_cand_eta", &ele_cand_eta,"ele_cand_eta/F");
  vbtfPresele_tree->Branch("ele_cand_phi",&ele_cand_phi,"ele_cand_phi/F");
  vbtfPresele_tree->Branch("ele_iso_track",&ele_iso_track,"ele_iso_track/F");
  vbtfPresele_tree->Branch("ele_iso_ecal",&ele_iso_ecal,"ele_iso_ecal/F");
  vbtfPresele_tree->Branch("ele_iso_hcal",&ele_iso_hcal,"ele_iso_hcal/F");
  vbtfPresele_tree->Branch("ele_id_sihih",&ele_id_sihih,"ele_id_sihih/F");
  vbtfPresele_tree->Branch("ele_id_deta",&ele_id_deta,"ele_id_deta/F");
  vbtfPresele_tree->Branch("ele_id_dphi",&ele_id_dphi,"ele_id_dphi/F");
  vbtfPresele_tree->Branch("ele_id_hoe",&ele_id_hoe,"ele_id_hoe/F");
  vbtfPresele_tree->Branch("ele_cr_mhitsinner",&ele_cr_mhitsinner,"ele_cr_mhitsinner/I");
  vbtfPresele_tree->Branch("ele_cr_dcot",&ele_cr_dcot,"ele_cr_dcot/F");
  vbtfPresele_tree->Branch("ele_cr_dist",&ele_cr_dist,"ele_cr_dist/F");
  vbtfPresele_tree->Branch("ele_vx",&ele_vx,"ele_vx/F");
  vbtfPresele_tree->Branch("ele_vy",&ele_vy,"ele_vy/F");
  vbtfPresele_tree->Branch("ele_vz",&ele_vz,"ele_vz/F");
  vbtfPresele_tree->Branch("pv_x",&pv_x,"pv_x/F");
  vbtfPresele_tree->Branch("pv_y",&pv_y,"pv_y/F");
  vbtfPresele_tree->Branch("pv_z",&pv_z,"pv_z/F");
  vbtfPresele_tree->Branch("ele_gsfCharge",&ele_gsfCharge,"ele_gsfCharge/I");
  vbtfPresele_tree->Branch("ele_ctfCharge",&ele_ctfCharge,"ele_ctfCharge/I");
  vbtfPresele_tree->Branch("ele_scPixCharge",&ele_scPixCharge,"ele_scPixCharge/I");
  vbtfPresele_tree->Branch("ele_eop",&ele_eop,"ele_eop/F");
  vbtfPresele_tree->Branch("ele_tip_bs",&ele_tip_bs,"ele_tip_bs/F");
  vbtfPresele_tree->Branch("ele_tip_pv",&ele_tip_pv,"ele_tip_pv/F");
  vbtfPresele_tree->Branch("ele_pin",&ele_pin,"ele_pin/F");
  vbtfPresele_tree->Branch("ele_pout",&ele_pout,"ele_pout/F");
  vbtfPresele_tree->Branch("event_caloMET",&event_caloMET,"event_caloMET/F");
  vbtfPresele_tree->Branch("event_pfMET",&event_pfMET,"event_pfMET/F");
  vbtfPresele_tree->Branch("event_tcMET",&event_tcMET,"event_tcMET/F");
  vbtfPresele_tree->Branch("event_caloMT",&event_caloMT,"event_caloMT/F");
  vbtfPresele_tree->Branch("event_pfMT",&event_pfMT,"event_pfMT/F");
  vbtfPresele_tree->Branch("event_tcMT",&event_tcMT,"event_tcMT/F");
  vbtfPresele_tree->Branch("event_caloMET_phi",&event_caloMET_phi,"event_caloMET_phi/F");
  vbtfPresele_tree->Branch("event_pfMET_phi",&event_pfMET_phi,"event_pfMET_phi/F");
  vbtfPresele_tree->Branch("event_tcMET_phi",&event_tcMET_phi,"event_tcMET_phi/F");
  vbtfPresele_tree->Branch("event_caloSumEt",&event_caloSumEt,"event_caloSumEt/F");
  vbtfPresele_tree->Branch("event_pfSumEt",&event_pfSumEt,"event_pfSumEt/F");
  vbtfPresele_tree->Branch("event_tcSumEt",&event_tcSumEt,"event_tcSumEt/F");
  // the extra jet variables:
  if (includeJetInformationInNtuples_) {
    vbtfPresele_tree->Branch("calojet_et",calojet_et,"calojet_et[5]/F");
    vbtfPresele_tree->Branch("calojet_eta",calojet_eta,"calojet_eta[5]/F");
    vbtfPresele_tree->Branch("calojet_phi",calojet_phi,"calojet_phi[5]/F");
    vbtfPresele_tree->Branch("pfjet_et",pfjet_et,"pfjet_et[5]/F");
    vbtfPresele_tree->Branch("pfjet_eta",pfjet_eta,"pfjet_eta[5]/F");
    vbtfPresele_tree->Branch("pfjet_phi",pfjet_phi,"pfjet_phi[5]/F");
  }
  if (storeExtraInformation_) {
    vbtfPresele_tree->Branch("ele2nd_sc_gsf_et",&ele2nd_sc_gsf_et,"ele2nd_sc_gsf_et/F");
    vbtfPresele_tree->Branch("ele2nd_sc_eta",&ele2nd_sc_eta,"ele2nd_sc_eta/F");
    vbtfPresele_tree->Branch("ele2nd_sc_phi",&ele2nd_sc_phi,"ele2nd_sc_phi/F");
    vbtfPresele_tree->Branch("ele2nd_sc_rho",&ele2nd_sc_rho,"ele2nd_sc_rho/F");
    vbtfPresele_tree->Branch("ele2nd_cand_eta",&ele2nd_cand_eta,"ele2nd_cand_eta/F");
    vbtfPresele_tree->Branch("ele2nd_cand_phi",&ele2nd_cand_phi,"ele2nd_cand_phi/F");
    vbtfPresele_tree->Branch("ele2nd_pin",&ele2nd_pin,"ele2nd_pin/F");
    vbtfPresele_tree->Branch("ele2nd_pout",&ele2nd_pout,"ele2nd_pout/F");
    vbtfPresele_tree->Branch("ele2nd_ecalDriven",&ele2nd_ecalDriven,"ele2nd_ecalDriven/I");
    vbtfPresele_tree->Branch("ele2nd_passes_selection",&ele2nd_passes_selection,"ele2nd_passes_selection/I");
    vbtfPresele_tree->Branch("ele_hltmatched_dr",&ele_hltmatched_dr,"ele_hltmatched_dr/F");
    vbtfPresele_tree->Branch("event_triggerDecision",&event_triggerDecision,"event_triggerDecision/I");
    vbtfPresele_tree->Branch("VtxTracksSize",&VtxTracksSize);
    vbtfPresele_tree->Branch("VtxNormalizedChi2",&VtxNormalizedChi2);
    vbtfPresele_tree->Branch("VtxTracksSizeBS",&VtxTracksSizeBS);
    vbtfPresele_tree->Branch("VtxNormalizedChi2BS",&VtxNormalizedChi2BS);
  }
  if (storeAllSecondElectronVariables_) {
    vbtfPresele_tree->Branch("ele2nd_cand_et",&ele2nd_cand_et,"ele2nd_cand_et/F");
    vbtfPresele_tree->Branch("ele2nd_iso_track",&ele2nd_iso_track ,"ele2nd_iso_track /F");
    vbtfPresele_tree->Branch("ele2nd_iso_ecal",&ele2nd_iso_ecal,"ele2nd_iso_ecal/F");
    vbtfPresele_tree->Branch("ele2nd_iso_hcal",&ele2nd_iso_hcal,"ele2nd_iso_hcal/F");
    vbtfPresele_tree->Branch("ele2nd_id_sihih",&ele2nd_id_sihih,"ele2nd_id_sihih/F");
    vbtfPresele_tree->Branch("ele2nd_id_deta",&ele2nd_id_deta,"ele2nd_id_deta/F");
    vbtfPresele_tree->Branch("ele2nd_id_dphi",&ele2nd_id_dphi,"ele2nd_id_dphi/F");
    vbtfPresele_tree->Branch("ele2nd_id_hoe",&ele2nd_id_hoe,"ele2nd_id_hoe/F");
    vbtfPresele_tree->Branch("ele2nd_cr_mhitsinner",&ele2nd_cr_mhitsinner,"ele2nd_cr_mhitsinner/I");
    vbtfPresele_tree->Branch("ele2nd_cr_dcot",&ele2nd_cr_dcot,"ele2nd_cr_dcot/F");
    vbtfPresele_tree->Branch("ele2nd_cr_dist",&ele2nd_cr_dist ,"ele2nd_cr_dist/F");
    vbtfPresele_tree->Branch("ele2nd_vx",&ele2nd_vx,"ele2nd_vx/F");
    vbtfPresele_tree->Branch("ele2nd_vy",&ele2nd_vy,"ele2nd_vy/F");
    vbtfPresele_tree->Branch("ele2nd_vz",&ele2nd_vz,"ele2nd_vz/F");

    vbtfPresele_tree->Branch("ele2nd_gsfCharge",&ele2nd_gsfCharge,"ele2nd_gsfCharge/I");
    vbtfPresele_tree->Branch("ele2nd_ctfCharge",&ele2nd_ctfCharge,"ele2nd_ctfCharge/I");
    vbtfPresele_tree->Branch("ele2nd_scPixCharge",&ele2nd_scPixCharge,"ele2nd_scPixCharge/I");
    vbtfPresele_tree->Branch("ele2nd_eop",&ele2nd_eop,"ele2nd_eop/F");
    vbtfPresele_tree->Branch("ele2nd_tip_bs",&ele2nd_tip_bs,"ele2nd_tip_bs/F");
    vbtfPresele_tree->Branch("ele2nd_tip_pv",&ele2nd_tip_pv,"ele2nd_tip_pv/F");
    vbtfPresele_tree->Branch("ele2nd_hltmatched_dr",&ele2nd_hltmatched_dr,"ele2nd_hltmatched_dr/F");
  }
  vbtfPresele_tree->Branch("event_datasetTag",&event_datasetTag,"event_dataSetTag/I");

  //
  // _________________________________________________________________________
  //
  //
  //


}

// ------------ method called once each job just after ending the event loop  -
void
WenuPlots::endJob() {
  TFile * newfile = new TFile(TString(outputFile_),"RECREATE");
  //
  // for consistency all the plots are in the root file
  // even though they may be empty (in the case when
  // usePrecalcID_== true inverted and N-1 are empty)
  h_met->Write();
  h_met_inverse->Write();
  h_mt->Write();
  h_mt_inverse->Write();

  h_met_EB->Write();
  h_met_inverse_EB->Write();
  h_mt_EB->Write();
  h_mt_inverse_EB->Write();

  h_met_EE->Write();
  h_met_inverse_EE->Write();
  h_mt_EE->Write();
  h_mt_inverse_EE->Write();

  h_scEt->Write();
  h_scEta->Write();
  h_scPhi->Write();

  h_EB_trkiso->Write();
  h_EB_ecaliso->Write();
  h_EB_hcaliso->Write();
  h_EB_sIetaIeta->Write();
  h_EB_dphi->Write();
  h_EB_deta->Write();
  h_EB_HoE->Write();

  h_EE_trkiso->Write();
  h_EE_ecaliso->Write();
  h_EE_hcaliso->Write();
  h_EE_sIetaIeta->Write();
  h_EE_dphi->Write();
  h_EE_deta->Write();
  h_EE_HoE->Write();

  //
  h_trackIso_eb_NmOne->Write();
  h_trackIso_ee_NmOne->Write();
  //
  newfile->Close();
  //
  // write the VBTF trees
  //
  WENU_VBTFpreseleFile_->Write();
  WENU_VBTFpreseleFile_->Close();
  WENU_VBTFselectionFile_->Write();
  WENU_VBTFselectionFile_->Close();

}


//define this as a plug-in
DEFINE_FWK_MODULE(WenuPlots);
