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

#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "HLTrigger/HLTanalyzers/interface/CaloTowerBoundries.h"

#include "DataFormats/METReco/interface/CaloMETCollection.h"

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
  void analyze(const edm::Handle<MuonCollection>                 & muon,
	       const edm::Handle<RecoChargedCandidateCollection> & mucands2,
	       const edm::Handle<edm::ValueMap<bool> >           & isoMap2,
	       const edm::Handle<RecoChargedCandidateCollection> & mucands3,
	       const edm::Handle<edm::ValueMap<bool> >           & isoMap3,
	       TTree* tree);


private:

  // Tree variables
  float *muonpt, *muonphi, *muoneta, *muonet, *muone; 
  float *muonl2pt, *muonl2eta, *muonl2phi, *muonl2dr, *muonl2dz;
  float *muonl3pt, *muonl3eta, *muonl3phi, *muonl3dr, *muonl3dz;
  float *muonl2pterr, *muonl3pterr;
  int nmuon, nmu2cand, nmu3cand;
  int *muonl2chg, *muonl2iso, *muonl3chg, *muonl3iso, *muonl32idx;
	

  // input variables
  bool _Monte,_Debug;

  int evtCounter;

  const float etaBarrel() {return 1.4;}

};

#endif
