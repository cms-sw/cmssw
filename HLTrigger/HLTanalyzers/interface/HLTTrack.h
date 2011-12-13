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
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

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
  void analyze(const edm::Handle<reco::IsolatedPixelTrackCandidateCollection> & IsoPixelTrackL3,
               const edm::Handle<reco::IsolatedPixelTrackCandidateCollection> & IsoPixelTrackL2,  
	       const edm::Handle<reco::VertexCollection> & pixelVertices,
               const edm::Handle<reco::RecoChargedCandidateCollection> & PixelTracksL3,
	       const edm::Handle<FEDRawDataCollection> hfedraw,
	       const edm::Handle<edmNew::DetSetVector<SiPixelCluster> > & pixelClusters,
	       TTree* tree);

  
private:

  // Tree variables
    //isoPixel
  float *isopixeltrackL3pt, *isopixeltrackL3eta, *isopixeltrackL3phi, *isopixeltrackL3maxptpxl, *isopixeltrackL3energy, *isopixeltrackL2pt, *isopixeltrackL2eta, *isopixeltrackL2dXY ;
  int nisopixeltrackL3;
    //minBiasPixel
  float *pixeltracksL3pt, *pixeltracksL3eta, *pixeltracksL3phi, *pixeltracksL3vz;
  int npixeltracksL3;
  int pixelfedsize;
  int npixelclusters;

  // input variables
  bool _Monte,_Debug;

  int evtCounter;

};

#endif
