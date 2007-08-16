#ifndef CaloRecoTauAlgorithm_H
#define CaloRecoTauAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BTauReco/interface/CombinedTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TauReco/interface/Tau.h"

#include "RecoTauTag/TauTagTools/interface/CaloTauElementsOperators.h"

#include "RecoBTag/BTagTools/interface/SignedTransverseImpactParameter.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

using namespace std;
using namespace reco;
using namespace edm;

const double chargedpi_mass=0.13957018;      //PDG Particle Physics Booklet, 2004

class  CaloRecoTauAlgorithm  {
 public:
  CaloRecoTauAlgorithm() : TransientTrackBuilder_(0){}  
  CaloRecoTauAlgorithm(const ParameterSet& iConfig) : TransientTrackBuilder_(0){
    LeadTrack_minPt_                    = iConfig.getParameter<double>("LeadTrack_minPt");
    Track_minPt_                        = iConfig.getParameter<double>("Track_minPt");
    
    MatchingConeMetric_                 = iConfig.getParameter<string>("MatchingConeMetric");
    MatchingConeSizeFormula_            = iConfig.getParameter<string>("MatchingConeSizeFormula");
    MatchingConeVariableSize_min_       = iConfig.getParameter<double>("MatchingConeVariableSize_min");
    MatchingConeVariableSize_max_       = iConfig.getParameter<double>("MatchingConeVariableSize_max");
    TrackerSignalConeMetric_            = iConfig.getParameter<string>("TrackerSignalConeMetric");
    TrackerSignalConeSizeFormula_       = iConfig.getParameter<string>("TrackerSignalConeSizeFormula");
    TrackerSignalConeVariableSize_min_  = iConfig.getParameter<double>("TrackerSignalConeVariableSize_min");
    TrackerSignalConeVariableSize_max_  = iConfig.getParameter<double>("TrackerSignalConeVariableSize_max");
    TrackerIsolConeMetric_              = iConfig.getParameter<string>("TrackerIsolConeMetric"); 
    TrackerIsolConeSizeFormula_         = iConfig.getParameter<string>("TrackerIsolConeSizeFormula"); 
    TrackerIsolConeVariableSize_min_    = iConfig.getParameter<double>("TrackerIsolConeVariableSize_min");
    TrackerIsolConeVariableSize_max_    = iConfig.getParameter<double>("TrackerIsolConeVariableSize_max");
    
    AreaMetric_recoElements_maxabsEta_  = iConfig.getParameter<double>("AreaMetric_recoElements_maxabsEta");
  }
  ~CaloRecoTauAlgorithm(){}
  void setTransientTrackBuilder(const TransientTrackBuilder* x){TransientTrackBuilder_=x;}
  Tau tag(const CombinedTauTagInfo&,const Vertex&); 
 private:
  const TransientTrackBuilder* TransientTrackBuilder_;
  double LeadTrack_minPt_;
  double Track_minPt_;
  string MatchingConeMetric_;
  string MatchingConeSizeFormula_;
  double MatchingConeVariableSize_min_;
  double MatchingConeVariableSize_max_;
  string TrackerSignalConeMetric_;
  string TrackerSignalConeSizeFormula_;
  double TrackerSignalConeVariableSize_min_;
  double TrackerSignalConeVariableSize_max_;
  string TrackerIsolConeMetric_;
  string TrackerIsolConeSizeFormula_;
  double TrackerIsolConeVariableSize_min_;
  double TrackerIsolConeVariableSize_max_;
  double AreaMetric_recoElements_maxabsEta_;
};
#endif 

