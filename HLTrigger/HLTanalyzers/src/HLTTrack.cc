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
#include "DataFormats/Math/interface/deltaR.h"

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

   //common
  const int kMaxTrackL3 = 10000;
   //isoPixel
  isopixeltrackL3pt = new float[kMaxTrackL3];
  isopixeltrackL3eta = new float[kMaxTrackL3];
  isopixeltrackL3phi = new float[kMaxTrackL3]; 
  isopixeltrackL3maxptpxl = new float[kMaxTrackL3];
  isopixeltrackL3energy = new float[kMaxTrackL3]; 
  isopixeltrackL2pt = new float[kMaxTrackL3];
  isopixeltrackL2eta = new float[kMaxTrackL3];
  isopixeltrackL2dXY = new float[kMaxTrackL3];

   //minBiasPixel
  pixeltracksL3pt = new float[kMaxTrackL3];
  pixeltracksL3eta = new float[kMaxTrackL3];
  pixeltracksL3phi = new float[kMaxTrackL3]; 
  pixeltracksL3vz = new float[kMaxTrackL3];

  nisopixeltrackHBL2=0;
  nisopixeltrackHBL3=0;
  isopixeltrackHBL2P        = new float[kMaxTrackL3]; 
  isopixeltrackHBL2Eta      = new float[kMaxTrackL3]; 
  isopixeltrackHBL2Phi      = new float[kMaxTrackL3]; 
  isopixeltrackHBL2MaxNearP = new float[kMaxTrackL3];
  isopixeltrackHBL3P        = new float[kMaxTrackL3]; 
  isopixeltrackHBL3Eta      = new float[kMaxTrackL3]; 
  isopixeltrackHBL3Phi      = new float[kMaxTrackL3];
  isopixeltrackHBL3MaxNearP = new float[kMaxTrackL3];

  nisopixeltrackHEL2=0;
  nisopixeltrackHEL3=0;
  isopixeltrackHEL2P        = new float[kMaxTrackL3]; 
  isopixeltrackHEL2Eta      = new float[kMaxTrackL3]; 
  isopixeltrackHEL2Phi      = new float[kMaxTrackL3]; 
  isopixeltrackHEL2MaxNearP = new float[kMaxTrackL3];
  isopixeltrackHEL3P        = new float[kMaxTrackL3]; 
  isopixeltrackHEL3Eta      = new float[kMaxTrackL3]; 
  isopixeltrackHEL3Phi      = new float[kMaxTrackL3];
  isopixeltrackHEL3MaxNearP = new float[kMaxTrackL3];


  // Track-specific branches of the tree 
   //isoPixel
  HltTree->Branch("NohIsoPixelTrackL3",&nisopixeltrackL3,"NohIsoPixelTrackL3/I");
  HltTree->Branch("ohIsoPixelTrackL3Pt",isopixeltrackL3pt,"ohIsoPixelTrackL3Pt[NohIsoPixelTrackL3]/F");
  HltTree->Branch("ohIsoPixelTrackL3Eta",isopixeltrackL3eta,"ohIsoPixelTrackL3Eta[NohIsoPixelTrackL3]/F");
  HltTree->Branch("ohIsoPixelTrackL3Phi",isopixeltrackL3phi,"ohIsoPixelTrackL3Phi[NohIsoPixelTrackL3]/F"); 
  HltTree->Branch("ohIsoPixelTrackL3MaxPtPxl",isopixeltrackL3maxptpxl,"ohIsoPixelTrackL3MaxPtPxl[NohIsoPixelTrackL3]/F");
  HltTree->Branch("ohIsoPixelTrackL3Energy",isopixeltrackL3energy,"ohIsoPixelTrackL3Energy[NohIsoPixelTrackL3]/F");
  HltTree->Branch("ohIsoPixelTrackL2pt",isopixeltrackL2pt,"ohIsoPixelTrackL2pt[NohIsoPixelTrackL3]/F");
  HltTree->Branch("ohIsoPixelTrackL2eta",isopixeltrackL2eta,"ohIsoPixelTrackL2eta[NohIsoPixelTrackL3]/F");
  HltTree->Branch("ohIsoPixelTrackL2dXY",isopixeltrackL2dXY,"ohIsoPixelTrackL2dXY[NohIsoPixelTrackL3]/F");

   //minBiasPixel
  HltTree->Branch("NohPixelTracksL3",&npixeltracksL3,"NohPixelTracksL3/I");
  HltTree->Branch("ohPixelTracksL3Pt",pixeltracksL3pt,"ohPixelTracksL3Pt[NohPixelTracksL3]/F");
  HltTree->Branch("ohPixelTracksL3Eta",pixeltracksL3eta,"ohPixelTracksL3Eta[NohPixelTracksL3]/F");
  HltTree->Branch("ohPixelTracksL3Phi",pixeltracksL3phi,"ohPixelTracksL3Phi[NohPixelTracksL3]/F"); 
  HltTree->Branch("ohPixelTracksL3Vz",pixeltracksL3vz,"ohPixelTracksL3Vz[NohPixelTracksL3]/F");

  //== IsoTrack Trigger
  HltTree->Branch("ohIsoPixelTrackHBL2N",       &nisopixeltrackHBL2,        "ohIsoPixelTrackHBL2N/I");
  HltTree->Branch("ohIsoPixelTrackHBL2P",        isopixeltrackHBL2P,        "ohIsoPixelTrackHBL2P[ohIsoPixelTrackHBL2N]/F");
  HltTree->Branch("ohIsoPixelTrackHBL2Eta",      isopixeltrackHBL2Eta,      "ohIsoPixelTrackHBL2Eta[ohIsoPixelTrackHBL2N]/F");
  HltTree->Branch("ohIsoPixelTrackHBL2Phi",      isopixeltrackHBL2Phi,      "ohIsoPixelTrackHBL2Phi[ohIsoPixelTrackHBL2N]/F");
  HltTree->Branch("ohIsoPixelTrackHBL2MaxNearP", isopixeltrackHBL2MaxNearP, "ohIsoPixelTrackHBL2MaxNearP[ohIsoPixelTrackHBL2N]/F");

  HltTree->Branch("ohIsoPixelTrackHBL3N",       &nisopixeltrackHBL3,        "ohIsoPixelTrackHBL3N/I");
  HltTree->Branch("ohIsoPixelTrackHBL3P",        isopixeltrackHBL3P,        "ohIsoPixelTrackHBL3P[ohIsoPixelTrackHBL3N]/F");
  HltTree->Branch("ohIsoPixelTrackHBL3Eta",      isopixeltrackHBL3Eta,      "ohIsoPixelTrackHBL3Eta[ohIsoPixelTrackHBL3N]/F");
  HltTree->Branch("ohIsoPixelTrackHBL3Phi",      isopixeltrackHBL3Phi,      "ohIsoPixelTrackHBL3Phi[ohIsoPixelTrackHBL3N]/F");
  HltTree->Branch("ohIsoPixelTrackHBL3MaxNearP", isopixeltrackHBL3MaxNearP, "ohIsoPixelTrackHBL3MaxNearP[ohIsoPixelTrackHBL3N]/F");

  HltTree->Branch("ohIsoPixelTrackHEL2N",       &nisopixeltrackHEL2,        "ohIsoPixelTrackHEL2N/I");
  HltTree->Branch("ohIsoPixelTrackHEL2P",        isopixeltrackHEL2P,        "ohIsoPixelTrackHEL2P[ohIsoPixelTrackHEL2N]/F");
  HltTree->Branch("ohIsoPixelTrackHEL2Eta",      isopixeltrackHEL2Eta,      "ohIsoPixelTrackHEL2Eta[ohIsoPixelTrackHEL2N]/F");
  HltTree->Branch("ohIsoPixelTrackHEL2Phi",      isopixeltrackHEL2Phi,      "ohIsoPixelTrackHEL2Phi[ohIsoPixelTrackHEL2N]/F");
  HltTree->Branch("ohIsoPixelTrackHEL2MaxNearP", isopixeltrackHEL2MaxNearP, "ohIsoPixelTrackHEL2MaxNearP[ohIsoPixelTrackHEL2N]/F");

  HltTree->Branch("ohIsoPixelTrackHEL3N",       &nisopixeltrackHEL3,        "ohIsoPixelTrackHEL3N/I");
  HltTree->Branch("ohIsoPixelTrackHEL3P",        isopixeltrackHEL3P,        "ohIsoPixelTrackHEL3P[ohIsoPixelTrackHEL3N]/F");
  HltTree->Branch("ohIsoPixelTrackHEL3Eta",      isopixeltrackHEL3Eta,      "ohIsoPixelTrackHEL3Eta[ohIsoPixelTrackHEL3N]/F");
  HltTree->Branch("ohIsoPixelTrackHEL3Phi",      isopixeltrackHEL3Phi,      "ohIsoPixelTrackHEL3Phi[ohIsoPixelTrackHEL3N]/F");
  HltTree->Branch("ohIsoPixelTrackHEL3MaxNearP", isopixeltrackHEL3MaxNearP, "ohIsoPixelTrackHEL3MaxNearP[ohIsoPixelTrackHEL3N]/F");
  //==
}

/* **Analyze the event** */
void HLTTrack::analyze(const edm::Handle<reco::IsolatedPixelTrackCandidateCollection> & IsoPixelTrackHBL2,
		       const edm::Handle<reco::IsolatedPixelTrackCandidateCollection> & IsoPixelTrackHBL3,  
		       const edm::Handle<reco::IsolatedPixelTrackCandidateCollection> & IsoPixelTrackHEL2,
		       const edm::Handle<reco::IsolatedPixelTrackCandidateCollection> & IsoPixelTrackHEL3,		       
		       const edm::Handle<reco::IsolatedPixelTrackCandidateCollection> & IsoPixelTrackL3,
		       const edm::Handle<reco::IsolatedPixelTrackCandidateCollection> & IsoPixelTrackL2,
		       const edm::Handle<reco::VertexCollection> & pixelVertices,
		       const edm::Handle<reco::RecoChargedCandidateCollection> & PixelTracksL3,
		       TTree* HltTree) {

  //============== HLT HCAL Isolated Track Trigger variables ================
  if(IsoPixelTrackHBL2.isValid()) {
    nisopixeltrackHBL2 = IsoPixelTrackHBL2->size();
    for (unsigned int itrk=0; itrk<IsoPixelTrackHBL2->size(); itrk++) {
      edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref = 
	edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(IsoPixelTrackHBL2, itrk);
      
      isopixeltrackHBL2P       [itrk] = candref->p();
      isopixeltrackHBL2Eta     [itrk] = candref->eta();
      isopixeltrackHBL2Phi     [itrk] = candref->phi();
      isopixeltrackHBL2MaxNearP[itrk] = candref->maxPtPxl();
      if(_Debug) {
	std::cout<<itrk<<" IsoPixelTrackHBL2 (pt,eta,phi) "<<candref->pt()<<" "
		 <<candref->eta()<<" "<<candref->phi()<<" p,E "
		 <<candref->p()  <<" "<<(candref->pt())*cosh(candref->track()->eta())
		 <<" maxNearPt "<<candref->maxPtPxl()
		 <<std::endl;
      }
    }
  }

  if(IsoPixelTrackHBL3.isValid()) {
    nisopixeltrackHBL3 = IsoPixelTrackHBL3->size();
    for (unsigned int itrk=0; itrk<IsoPixelTrackHBL3->size(); itrk++) {
      edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref = 
	edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(IsoPixelTrackHBL3, itrk);
      
      isopixeltrackHBL3P       [itrk] = candref->p();
      isopixeltrackHBL3Eta     [itrk] = candref->eta();
      isopixeltrackHBL3Phi     [itrk] = candref->phi();
      isopixeltrackHBL3MaxNearP[itrk] = candref->maxPtPxl();
      
      if(_Debug) {
	std::cout<<itrk<<" IsoPixelTrackHBL3 (pt,eta,phi) "<<candref->pt()<<" "
		 <<candref->eta()<<" "<<candref->phi()<<" p,E "
		 <<candref->p()  <<" "<<(candref->pt())*cosh(candref->track()->eta())
		 <<" maxNearPt "<<candref->maxPtPxl()
		 <<std::endl;
      }
    }
  }

  if(IsoPixelTrackHEL2.isValid()) {
    nisopixeltrackHEL2 = IsoPixelTrackHEL2->size();
    for (unsigned int itrk=0; itrk<IsoPixelTrackHEL2->size(); itrk++) {
      edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref = 
	edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(IsoPixelTrackHEL2, itrk);
      
      isopixeltrackHEL2P       [itrk] = candref->p();
      isopixeltrackHEL2Eta     [itrk] = candref->eta();
      isopixeltrackHEL2Phi     [itrk] = candref->phi();
      isopixeltrackHEL2MaxNearP[itrk] = candref->maxPtPxl();
      if(_Debug) {
	std::cout<<itrk<<" IsoPixelTrackHEL2 (pt,eta,phi) "<<candref->pt()<<" "
		 <<candref->eta()<<" "<<candref->phi()<<" p,E "
		 <<candref->p()  <<" "<<(candref->pt())*cosh(candref->track()->eta())
		 <<" maxNearPt "<<candref->maxPtPxl()
		 <<std::endl;
      }
    }
  }

  if(IsoPixelTrackHEL3.isValid()) {
    nisopixeltrackHEL3 = IsoPixelTrackHEL3->size();
    for (unsigned int itrk=0; itrk<IsoPixelTrackHEL3->size(); itrk++) {
      edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref = 
	edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(IsoPixelTrackHEL3, itrk);
      
      isopixeltrackHEL3P       [itrk] = candref->p();
      isopixeltrackHEL3Eta     [itrk] = candref->eta();
      isopixeltrackHEL3Phi     [itrk] = candref->phi();
      isopixeltrackHEL3MaxNearP[itrk] = candref->maxPtPxl();
      
      if(_Debug) {
	std::cout<<itrk<<" IsoPixelTrackHEL3 (pt,eta,phi) "<<candref->pt()<<" "
		 <<candref->eta()<<" "<<candref->phi()<<" p,E "
		 <<candref->p()  <<" "<<(candref->pt())*cosh(candref->track()->eta())
		 <<" maxNearPt "<<candref->maxPtPxl()
		 <<std::endl;
      }
    }
  }

  //=========================================================================

  //isoPixel
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

  //minBiasPixel  
  if (PixelTracksL3.isValid()) { 
    // Ref to Candidate object to be recorded in filter object 
    edm::Ref<reco::RecoChargedCandidateCollection> candref; 
    
    npixeltracksL3 = PixelTracksL3->size();
    
    for (unsigned int i=0; i<PixelTracksL3->size(); i++) 
      { 
	candref = edm::Ref<reco::RecoChargedCandidateCollection>(PixelTracksL3, i); 
	
	pixeltracksL3pt[i] = candref->pt();
	pixeltracksL3eta[i] = candref->eta();
        pixeltracksL3phi[i] = candref->phi();   
	pixeltracksL3vz[i] = candref->vz();
	
	if (IsoPixelTrackL2.isValid()) {
	  double minDR=100;
	  edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candrefl2;
	  edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candrefl2matched;
	  for (unsigned int j=0; j<IsoPixelTrackL2->size(); j++) 
	    { 
	      candrefl2 = edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(IsoPixelTrackL2, j);
	      double drL3L2 = deltaR(candrefl2->eta(), candrefl2->phi(),candref->eta(), candref->phi());
	      if (drL3L2<minDR)
		{
		  candrefl2matched=candrefl2;
		  minDR=drL3L2;
		}
	    }
	  if (candrefl2matched.isNonnull())
	    {
	      isopixeltrackL2pt[i]=candrefl2matched->pt();
	      isopixeltrackL2eta[i]=candrefl2matched->eta();
	      if (pixelVertices.isValid()) 
		{
		  double minDZ=100;
		  edm::Ref<reco::VertexCollection> vertref; 
		  edm::Ref<reco::VertexCollection> vertrefMatched;
		  for (unsigned int k=0; k<pixelVertices->size(); k++)
		    {
		      vertref=edm::Ref<reco::VertexCollection>(pixelVertices, k);
		      double dz=fabs(candrefl2matched->track()->dz(vertref->position()));
		      if (dz<minDZ)
			{
			  minDZ=dz;
			  vertrefMatched=vertref;
			}
		    }
		  if (vertrefMatched.isNonnull()) isopixeltrackL2dXY[i] = candrefl2matched->track()->dxy(vertref->position());
		  else isopixeltrackL2dXY[i]=0;
		}
	    }
	}
      } 
  }
  else {npixeltracksL3 = 0;} 


  //////////////////////////////////////////////////////////////////////////////
  
}
