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

  HltTree->Branch("ohPixelFEDSize",&pixelfedsize,"ohPixelFEDSize/I");
  HltTree->Branch("NohPixelClusters",&npixelclusters,"NohPixelClusters/I");
}

/* **Analyze the event** */
void HLTTrack::analyze(
                       const edm::Handle<reco::IsolatedPixelTrackCandidateCollection> & IsoPixelTrackL3,
		       const edm::Handle<reco::IsolatedPixelTrackCandidateCollection> & IsoPixelTrackL2,
		       const edm::Handle<reco::VertexCollection> & pixelVertices,
		       const edm::Handle<reco::RecoChargedCandidateCollection> & PixelTracksL3,
		       const edm::Handle<FEDRawDataCollection> hfedraw,
		       const edm::Handle<edmNew::DetSetVector<SiPixelCluster> > & pixelClusters,
		       TTree* HltTree) {

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

  //Pixel FED activity
  const FEDRawDataCollection theRaw = * hfedraw;
  int sumoffeds = 0; 

  for (unsigned int i = 0; i <= 39; i++)
    {
      sumoffeds = sumoffeds + (theRaw.FEDData(i).size());
    }

  pixelfedsize = sumoffeds;

  npixelclusters = pixelClusters->dataSize();

  //////////////////////////////////////////////////////////////////////////////
  
}
