#ifndef HLTTRACK_H
#define HLTTRACK_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h" 
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

typedef std::vector<std::string> MyStrings;

/** \class HLTTRack
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */
class HLTTrack {
public:
  HLTTrack(); 

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  /** Analyze the Data */
  void analyze(const edm::Handle<reco::IsolatedPixelTrackCandidateCollection>                 & IsoPixelTrackL3,
	       TTree* tree);

  
private:

  // Tree variables
  float *isopixeltrackL3pt, *isopixeltrackL3eta, *isopixeltrackL3phi, *isopixeltrackL3maxptpxl, *isopixeltrackL3energy;
  int nisopixeltrackL3;

  // input variables
  bool _Monte,_Debug;

  int evtCounter;

};

#endif
