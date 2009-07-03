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

#include "HLTrigger/HLTanalyzers/interface/HLTTrack.h"

HLTTrack::HLTTrack() {
  evtCounter=0;

  //set parameter defaults 
  _Monte=false;
  _Debug=false;
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTTrack::setup(const edm::ParameterSet& pSet, TTree* HltTree) {

  edm::ParameterSet myEmParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  std::vector<std::string> parameterNames = myEmParams.getParameterNames() ;
  
  for ( std::vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myEmParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "Debug" ) _Debug =  myEmParams.getParameter<bool>( *iParam );
  }

  const int kMaxTrackL3 = 10000;
  isopixeltrackL3pt = new float[kMaxTrackL3];
  isopixeltrackL3eta = new float[kMaxTrackL3];
  isopixeltrackL3phi = new float[kMaxTrackL3]; 
  isopixeltrackL3maxptpxl = new float[kMaxTrackL3];
  isopixeltrackL3energy = new float[kMaxTrackL3]; 

  // Track-specific branches of the tree 
  HltTree->Branch("NohIsoPixelTrackL3",&nisopixeltrackL3,"NohIsoPixelTrackL3/I");
  HltTree->Branch("ohIsoPixelTrackL3Pt",isopixeltrackL3pt,"ohIsoPixelTrackL3Pt[NohIsoPixelTrackL3]/F");
  HltTree->Branch("ohIsoPixelTrackL3Eta",isopixeltrackL3eta,"ohIsoPixelTrackL3Eta[NohIsoPixelTrackL3]/F");
  HltTree->Branch("ohIsoPixelTrackL3Phi",isopixeltrackL3phi,"ohIsoPixelTrackL3Phi[NohIsoPixelTrackL3]/F"); 
  HltTree->Branch("ohIsoPixelTrackL3MaxPtPxl",isopixeltrackL3maxptpxl,"ohIsoPixelTrackL3MaxPtPxl[NohIsoPixelTrackL3]/F");
  HltTree->Branch("ohIsoPixelTrackL3Energy",isopixeltrackL3energy,"ohIsoPixelTrackL3Energy[NohIsoPixelTrackL3]/F");
}

/* **Analyze the event** */
void HLTTrack::analyze(
		       const edm::Handle<reco::IsolatedPixelTrackCandidateCollection>                 & IsoPixelTrackL3,
		       TTree* HltTree) {

  //std::cout << " Beginning HLTTrack " << std::endl;
  
  if (IsoPixelTrackL3.isValid()) { 
    // Ref to Candidate object to be recorded in filter object 
    edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref; 
    
    nisopixeltrackL3 = IsoPixelTrackL3->size();
    
    for (unsigned int i=0; i<IsoPixelTrackL3->size(); i++) 
      { 
	candref = edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(IsoPixelTrackL3, i); 
	
	isopixeltrackL3maxptpxl[i] = candref->maxPtPxl();
	isopixeltrackL3pt[i] = candref->pt();
	isopixeltrackL3eta[i] = candref->track()->eta();
        isopixeltrackL3phi[i] = candref->phi();   
	isopixeltrackL3energy[i] = (candref->pt())*cosh(candref->track()->eta());
      } 
  }
  else {nisopixeltrackL3 = 0;} 

  //////////////////////////////////////////////////////////////////////////////
  
}
