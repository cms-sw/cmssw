#ifndef CaloRecoTauTagInfoAlgorithm_H
#define CaloRecoTauTagInfoAlgorithm_H

#include "DataFormats/TauReco/interface/TauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace std;
using namespace reco;
using namespace edm;

class  CaloRecoTauTagInfoAlgorithm  {
 public:
  CaloRecoTauTagInfoAlgorithm(){}  
  CaloRecoTauTagInfoAlgorithm(const ParameterSet& parameters){
    // parameters of the considered rec. tk's (catched through a JetTracksAssociator object) :
    tkminPt_                            = parameters.getParameter<double>("tkminPt");
    tkminPixelHitsn_                    = parameters.getParameter<int>("tkminPixelHitsn");
    tkminTrackerHitsn_                  = parameters.getParameter<int>("tkminTrackerHitsn");
    tkmaxipt_                           = parameters.getParameter<double>("tkmaxipt");
    tkmaxChi2_                          = parameters.getParameter<double>("tkmaxChi2");
    tktorefpointDZ_                     = parameters.getParameter<double>("tktorefpointDZ");
    // 
    UsePVconstraint_                    = parameters.getParameter<bool>("UsePVconstraint");
  }
  ~CaloRecoTauTagInfoAlgorithm(){}
  TauTagInfo tag(const CaloJetRef&,const TrackRefVector&,const Vertex&); 
 private:  
  TrackRefVector filteredTracks(TrackRefVector theTracks,double tkminPt,int tkminPixelHitsn,int tkminTrackerHitsn,double tkmaxipt,double tkmaxChi2,double tktorefpointDZ,bool UsePVconstraint,double PVtx_Z);
  //
  double tkminPt_;
  int tkminPixelHitsn_;
  int tkminTrackerHitsn_;
  double tkmaxipt_;
  double tkmaxChi2_;
  double tktorefpointDZ_;
  // 
  bool UsePVconstraint_;
};
#endif 

