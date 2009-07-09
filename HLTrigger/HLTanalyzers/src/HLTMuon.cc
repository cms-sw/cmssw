#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>
#include <stdlib.h>
#include <string.h>

#include "HLTrigger/HLTanalyzers/interface/HLTMuon.h"

HLTMuon::HLTMuon() {
  evtCounter=0;

  //set parameter defaults 
  _Monte=false;
  _Debug=false;
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTMuon::setup(const edm::ParameterSet& pSet, TTree* HltTree) {

  edm::ParameterSet myEmParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  vector<std::string> parameterNames = myEmParams.getParameterNames() ;
  
  for ( vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myEmParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "Debug" ) _Debug =  myEmParams.getParameter<bool>( *iParam );
  }

  const int kMaxMuon = 10000;
  muonpt = new float[kMaxMuon];
  muonphi = new float[kMaxMuon];
  muoneta = new float[kMaxMuon];
  muonet = new float[kMaxMuon];
  muone = new float[kMaxMuon];
  const int kMaxMuonL2 = 500;
  muonl2pt = new float[kMaxMuonL2];
  muonl2phi = new float[kMaxMuonL2];
  muonl2eta = new float[kMaxMuonL2];
  muonl2dr = new float[kMaxMuonL2];
  muonl2dz = new float[kMaxMuonL2];
  muonl2chg = new int[kMaxMuonL2];
  muonl2pterr = new float[kMaxMuonL2];
  muonl2iso = new int[kMaxMuonL2];
  const int kMaxMuonL3 = 500;
  muonl3pt = new float[kMaxMuonL3];
  muonl3phi = new float[kMaxMuonL3];
  muonl3eta = new float[kMaxMuonL3];
  muonl3dr = new float[kMaxMuonL3];
  muonl3dz = new float[kMaxMuonL3];
  muonl3chg = new int[kMaxMuonL3];
  muonl3pterr = new float[kMaxMuonL3];
  muonl3iso = new int[kMaxMuonL3];
  muonl32idx = new int[kMaxMuonL3];

  // Muon-specific branches of the tree 
  HltTree->Branch("NrecoMuon",&nmuon,"NrecoMuon/I");
  HltTree->Branch("recoMuonPt",muonpt,"recoMuonPt[NrecoMuon]/F");
  HltTree->Branch("recoMuonPhi",muonphi,"recoMuonPhi[NrecoMuon]/F");
  HltTree->Branch("recoMuonEta",muoneta,"recoMuonEta[NrecoMuon]/F");
  HltTree->Branch("recoMuonEt",muonet,"recoMuonEt[NrecoMuon]/F");
  HltTree->Branch("recoMuonE",muone,"recoMuonE[NrecoMuon]/F");
  HltTree->Branch("NohMuL2",&nmu2cand,"NohMuL2/I");
  HltTree->Branch("ohMuL2Pt",muonl2pt,"ohMuL2Pt[NohMuL2]/F");
  HltTree->Branch("ohMuL2Phi",muonl2phi,"ohMuL2Phi[NohMuL2]/F");
  HltTree->Branch("ohMuL2Eta",muonl2eta,"ohMuL2Eta[NohMuL2]/F");
  HltTree->Branch("ohMuL2Chg",muonl2chg,"ohMuL2Chg[NohMuL2]/I");
  HltTree->Branch("ohMuL2PtErr",muonl2pterr,"ohMuL2PtErr[NohMuL2]/F");
  HltTree->Branch("ohMuL2Iso",muonl2iso,"ohMuL2Iso[NohMuL2]/I");
  HltTree->Branch("ohMuL2Dr",muonl2dr,"ohMuL2Dr[NohMuL2]/F");
  HltTree->Branch("ohMuL2Dz",muonl2dz,"ohMuL2Dz[NohMuL2]/F");
  HltTree->Branch("NohMuL3",&nmu3cand,"NohMuL3/I");
  HltTree->Branch("ohMuL3Pt",muonl3pt,"ohMuL3Pt[NohMuL3]/F");
  HltTree->Branch("ohMuL3Phi",muonl3phi,"ohMuL3Phi[NohMuL3]/F");
  HltTree->Branch("ohMuL3Eta",muonl3eta,"ohMuL3Eta[NohMuL3]/F");
  HltTree->Branch("ohMuL3Chg",muonl3chg,"ohMuL3Chg[NohMuL3]/I");
  HltTree->Branch("ohMuL3PtErr",muonl3pterr,"ohMuL3PtErr[NohMuL3]/F");
  HltTree->Branch("ohMuL3Iso",muonl3iso,"ohMuL3Iso[NohMuL3]/I");
  HltTree->Branch("ohMuL3Dr",muonl3dr,"ohMuL3Dr[NohMuL3]/F");
  HltTree->Branch("ohMuL3Dz",muonl3dz,"ohMuL3Dz[NohMuL3]/F");
  HltTree->Branch("ohMuL3L2idx",muonl32idx,"ohMuL3L2idx[NohMuL3]/I");

}

/* **Analyze the event** */
void HLTMuon::analyze(const edm::Handle<MuonCollection>                 & Muon,
		      const edm::Handle<RecoChargedCandidateCollection> & MuCands2,
		      const edm::Handle<edm::ValueMap<bool> >           & isoMap2,
		      const edm::Handle<RecoChargedCandidateCollection> & MuCands3,
		      const edm::Handle<edm::ValueMap<bool> >           & isoMap3,
		      TTree* HltTree) {

  //std::cout << " Beginning HLTMuon " << std::endl;

  if (Muon.isValid()) {
    MuonCollection mymuons;
    mymuons = * Muon;
    std::sort(mymuons.begin(),mymuons.end(),PtGreater());
    nmuon = mymuons.size();
    typedef MuonCollection::const_iterator muiter;
    int imu=0;
    for (muiter i=mymuons.begin(); i!=mymuons.end(); i++) {
      muonpt[imu] = i->pt();
      muonphi[imu] = i->phi();
      muoneta[imu] = i->eta();
      muonet[imu] = i->et();
      muone[imu] = i->energy();
      imu++;
    }
  }
  else {nmuon = 0;}

  /////////////////////////////// Open-HLT muons ///////////////////////////////

  // Dealing with L2 muons
  RecoChargedCandidateCollection myMucands2;
  if (MuCands2.isValid()) {
//     RecoChargedCandidateCollection myMucands2;
    myMucands2 = * MuCands2;
    std::sort(myMucands2.begin(),myMucands2.end(),PtGreater());
    nmu2cand = myMucands2.size();
    typedef RecoChargedCandidateCollection::const_iterator cand;
    int imu2c=0;
    for (cand i=myMucands2.begin(); i!=myMucands2.end(); i++) {
      TrackRef tk = i->get<TrackRef>();

      muonl2pt[imu2c] = tk->pt();
      // eta (we require |eta|<2.5 in all filters
      muonl2eta[imu2c] = tk->eta();
      muonl2phi[imu2c] = tk->phi();

      // Dr (transverse distance to (0,0,0))
      // For baseline triggers, we do no cut at L2 (|dr|<9999 cm)
      // However, we use |dr|<200 microns at L3, which it probably too tough for LHC startup
      muonl2dr[imu2c] = fabs(tk->d0());

      // Dz (longitudinal distance to z=0 when at minimum transverse distance)
      // For baseline triggers, we do no cut (|dz|<9999 cm), neither at L2 nor at L3
      muonl2dz[imu2c] = fabs(tk->dz());

      // At present we do not cut on this, but on a 90% CL value "ptLx" defined here below
      // We should change this in the future and cut directly on "pt", to avoid unnecessary complications and risks
      // Baseline cuts (HLT exercise):
      //                Relaxed Single muon:  ptLx>16 GeV
      //                Isolated Single muon: ptLx>11 GeV
      //                Relaxed Double muon: ptLx>3 GeV
       double l2_err0 = tk->error(0); // error on q/p
       double l2_abspar0 = fabs(tk->parameter(0)); // |q/p|
//       double ptLx = tk->pt();
      // convert 50% efficiency threshold to 90% efficiency threshold
      // For L2 muons: nsigma_Pt_ = 3.9
//       double nsigma_Pt_ = 3.9;
      // For L3 muons: nsigma_Pt_ = 2.2
      // these are the old TDR values for nsigma_Pt_
      // We know that these values are slightly smaller for CMSSW
      // But as quoted above, we want to get rid of this gymnastics in the future
//       if (abspar0>0) ptLx += nsigma_Pt_*err0/abspar0*tk->pt();

      // Charge
      // We use the charge in some dimuon paths
			muonl2pterr[imu2c] = l2_err0/l2_abspar0;
      muonl2chg[imu2c] = tk->charge();

      if (isoMap2.isValid()){
	// Isolation flag (this is a bool value: true => isolated)
	edm::ValueMap<bool> ::value_type muon1IsIsolated = (*isoMap2)[tk];
	muonl2iso[imu2c] = muon1IsIsolated;
      }
      else {muonl2iso[imu2c] = -999;}

      imu2c++;
    }
  }
  else {nmu2cand = 0;}

  // Dealing with L3 muons
  RecoChargedCandidateCollection myMucands3;
  if (MuCands3.isValid()) {
    myMucands3 = * MuCands3;
    std::sort(myMucands3.begin(),myMucands3.end(),PtGreater());
    nmu3cand = myMucands3.size();
    typedef RecoChargedCandidateCollection::const_iterator cand;
    int imu3c=0;
    for (cand i=myMucands3.begin(); i!=myMucands3.end(); i++) {
      TrackRef tk = i->get<TrackRef>();

      TrackRef staTrack;
      typedef MuonTrackLinksCollection::const_iterator l3muon;
      int il3 = 0;
      //find the corresponding L2 track
      staTrack = tk->seedRef().castTo<edm::Ref< L3MuonTrajectorySeedCollection> >()->l2Track();
      il3++;
      int imu2idx = 0;
      if (MuCands2.isValid()) {
	typedef RecoChargedCandidateCollection::const_iterator candl2;
	for (candl2 i=myMucands2.begin(); i!=myMucands2.end(); i++) {
	  TrackRef tkl2 = i->get<TrackRef>();
	  if ( tkl2 == staTrack ) {break;}
	  imu2idx++;
	}
      }
      else {imu2idx = -999;}
      muonl32idx[imu3c] = imu2idx; // Index of the L2 muon having matched with the L3 muon with index imu3c
      
      muonl3pt[imu3c] = tk->pt();
      // eta (we require |eta|<2.5 in all filters
      muonl3eta[imu3c] = tk->eta();
      muonl3phi[imu3c] = tk->phi();

//       // Dr (transverse distance to (0,0,0))
//       // For baseline triggers, we do no cut at L2 (|dr|<9999 cm)
//       // However, we use |dr|<300 microns at L3, which it probably too tough for LHC startup
      muonl3dr[imu3c] = fabs(tk->d0());

//       // Dz (longitudinal distance to z=0 when at minimum transverse distance)
//       // For baseline triggers, we do no cut (|dz|<9999 cm), neither at L2 nor at L3
      muonl3dz[imu3c] = fabs(tk->dz());

//       // At present we do not cut on this, but on a 90% CL value "ptLx" defined here below
//       // We should change this in the future and cut directly on "pt", to avoid unnecessary complications and risks
//       // Baseline cuts (HLT exercise):
//       //                Relaxed Single muon:  ptLx>16 GeV
//       //                Isolated Single muon: ptLx>11 GeV
//       //                Relaxed Double muon: ptLx>3 GeV
        double l3_err0 = tk->error(0); // error on q/p
        double l3_abspar0 = fabs(tk->parameter(0)); // |q/p|
// //       double ptLx = tk->pt();
//       // convert 50% efficiency threshold to 90% efficiency threshold
//       // For L2 muons: nsigma_Pt_ = 3.9
//       // For L3 muons: nsigma_Pt_ = 2.2
// //       double nsigma_Pt_ = 2.2;
//       // these are the old TDR values for nsigma_Pt_
//       // We know that these values are slightly smaller for CMSSW
//       // But as quoted above, we want to get rid of this gymnastics in the future
// //       if (abspar0>0) ptLx += nsigma_Pt_*err0/abspar0*tk->pt();

      // Charge
      // We use the charge in some dimuon paths
      muonl3pterr[imu3c] = l3_err0/l3_abspar0;
      muonl3chg[imu3c] = tk->charge();

      if (isoMap3.isValid()){
	// Isolation flag (this is a bool value: true => isolated)
	edm::ValueMap<bool> ::value_type muon1IsIsolated = (*isoMap3)[tk];
	muonl3iso[imu3c] = muon1IsIsolated;
      }
      else {muonl3iso[imu3c] = -999;}

      imu3c++;
    }
  }
  else {nmu3cand = 0;}

  //////////////////////////////////////////////////////////////////////////////



}
