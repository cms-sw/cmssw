#ifndef HLTMUON_H
#define HLTMUON_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
//#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h" 
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h" 
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h" 
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h" 

#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "HLTrigger/HLTanalyzers/interface/CaloTowerBoundries.h"

#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"

typedef std::vector<std::string> MyStrings;

/** \class HLTMuon
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */
class HLTMuon {
public:
  HLTMuon(); 

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  /** Analyze the Data */
  void analyze(const edm::Handle<reco::MuonCollection>                 & muon,
               const edm::Handle<l1extra::L1MuonParticleCollection>   & mucands1, 
	       const edm::Handle<reco::RecoChargedCandidateCollection> & mucands2,
	       const edm::Handle<edm::ValueMap<bool> >           & isoMap2,
	       const edm::Handle<reco::RecoChargedCandidateCollection> & mucands3,
	       const edm::Handle<edm::ValueMap<bool> >           & isoMap3,
               const edm::Handle<reco::RecoChargedCandidateCollection> & oniaPixelCands,
               const edm::Handle<reco::RecoChargedCandidateCollection> & oniaTrackCands,
	       const edm::Handle<reco::RecoChargedCandidateCollection> & munovtxcands2, 
               const reco::BeamSpot::Point & BSPosition,
	       TTree* tree);


private:

  // Tree variables
  float *muonpt, *muonphi, *muoneta, *muonet, *muone, *muonchi2NDF, *muoncharge,
  *muonTrkIsoR03, *muonECalIsoR03, *muonHCalIsoR03, *muonD0;
  int *muontype, *muonNValidTrkHits, *muonNValidMuonHits;
  float *muonl2pt, *muonl2eta, *muonl2phi, *muonl2dr, *muonl2dz;
  float *muonl3pt, *muonl3eta, *muonl3phi, *muonl3dr, *muonl3dz;
  float *muonl2novtxpt, *muonl2novtxeta, *muonl2novtxphi, *muonl2novtxdr, *muonl2novtxdz; 
  float *muonl2pterr, *muonl3pterr, *muonl2novtxpterr;
  int nmuon, nmu2cand, nmu3cand, nmu2novtxcand;
  int *muonl2chg, *muonl2iso, *muonl3chg, *muonl3iso, *muonl32idx, *muonl21idx, *muonl2novtxchg, *muonl2novtxiso, *muonl2novtx1idx;
  int nOniaPixelCand, nOniaTrackCand;
  float *oniaPixelpt, *oniaPixeleta, *oniaPixelphi, *oniaPixeldr, *oniaPixeldz, *oniaPixelNormChi2;
  float *oniaTrackpt, *oniaTracketa, *oniaTrackphi, *oniaTrackdr, *oniaTrackdz, *oniaTrackNormChi2;
  int *oniaPixelchg, *oniaTrackchg, *oniaPixelHits, *oniaTrackHits;
	

  // input variables
  bool _Monte,_Debug;

  int evtCounter;

  const float etaBarrel() {return 1.4;}

};

#endif
