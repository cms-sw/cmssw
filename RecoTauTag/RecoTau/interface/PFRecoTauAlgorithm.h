#ifndef PFRecoTauAlgorithm_H
#define PFRecoTauAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BTauReco/interface/PFIsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TauReco/interface/Tau.h"

#include "RecoTauTag/TauTagTools/interface/PFTauElementsOperators.h"
#include "RecoTauTag/TauTagTools/interface/CaloTauElementsOperators.h"

#include "RecoBTag/BTagTools/interface/SignedTransverseImpactParameter.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

using namespace std;
using namespace reco;
using namespace edm;

class  PFRecoTauAlgorithm  {
 public:
  PFRecoTauAlgorithm() : TransientTrackBuilder_(0){}  
  PFRecoTauAlgorithm(const ParameterSet& iConfig) : TransientTrackBuilder_(0){
    LeadChargedHadrCand_minPt_             = iConfig.getParameter<double>("LeadChargedHadrCand_minPt"); 
    ChargedHadrCand_minPt_                 = iConfig.getParameter<double>("ChargedHadrCand_minPt");
    NeutrHadrCand_minPt_                   = iConfig.getParameter<double>("NeutrHadrCand_minPt");
    GammaCand_minPt_                       = iConfig.getParameter<double>("GammaCand_minPt");       
    LeadTrack_minPt_                       = iConfig.getParameter<double>("LeadTrack_minPt");
    Track_minPt_                           = iConfig.getParameter<double>("Track_minPt");
     
    MatchingConeMetric_                    = iConfig.getParameter<string>("MatchingConeMetric");
    MatchingConeSizeFormula_               = iConfig.getParameter<string>("MatchingConeSizeFormula");
    MatchingConeVariableSize_min_          = iConfig.getParameter<double>("MatchingConeVariableSize_min");
    MatchingConeVariableSize_max_          = iConfig.getParameter<double>("MatchingConeVariableSize_max");
    TrackerSignalConeMetric_               = iConfig.getParameter<string>("TrackerSignalConeMetric");
    TrackerSignalConeSizeFormula_          = iConfig.getParameter<string>("TrackerSignalConeSizeFormula");
    TrackerSignalConeVariableSize_min_     = iConfig.getParameter<double>("TrackerSignalConeVariableSize_min");
    TrackerSignalConeVariableSize_max_     = iConfig.getParameter<double>("TrackerSignalConeVariableSize_max");
    TrackerIsolConeMetric_                 = iConfig.getParameter<string>("TrackerIsolConeMetric"); 
    TrackerIsolConeSizeFormula_            = iConfig.getParameter<string>("TrackerIsolConeSizeFormula"); 
    TrackerIsolConeVariableSize_min_       = iConfig.getParameter<double>("TrackerIsolConeVariableSize_min");
    TrackerIsolConeVariableSize_max_       = iConfig.getParameter<double>("TrackerIsolConeVariableSize_max");
    ECALSignalConeMetric_               = iConfig.getParameter<string>("ECALSignalConeMetric");
    ECALSignalConeSizeFormula_          = iConfig.getParameter<string>("ECALSignalConeSizeFormula");    
    ECALSignalConeVariableSize_min_     = iConfig.getParameter<double>("ECALSignalConeVariableSize_min");
    ECALSignalConeVariableSize_max_     = iConfig.getParameter<double>("ECALSignalConeVariableSize_max");
    ECALIsolConeMetric_                 = iConfig.getParameter<string>("ECALIsolConeMetric");
    ECALIsolConeSizeFormula_            = iConfig.getParameter<string>("ECALIsolConeSizeFormula");      
    ECALIsolConeVariableSize_min_       = iConfig.getParameter<double>("ECALIsolConeVariableSize_min");
    ECALIsolConeVariableSize_max_       = iConfig.getParameter<double>("ECALIsolConeVariableSize_max");
    HCALSignalConeMetric_               = iConfig.getParameter<string>("HCALSignalConeMetric");
    HCALSignalConeSizeFormula_          = iConfig.getParameter<string>("HCALSignalConeSizeFormula");    
    HCALSignalConeVariableSize_min_     = iConfig.getParameter<double>("HCALSignalConeVariableSize_min");
    HCALSignalConeVariableSize_max_     = iConfig.getParameter<double>("HCALSignalConeVariableSize_max");
    HCALIsolConeMetric_                 = iConfig.getParameter<string>("HCALIsolConeMetric");
    HCALIsolConeSizeFormula_            = iConfig.getParameter<string>("HCALIsolConeSizeFormula");      
    HCALIsolConeVariableSize_min_       = iConfig.getParameter<double>("HCALIsolConeVariableSize_min");
    HCALIsolConeVariableSize_max_       = iConfig.getParameter<double>("HCALIsolConeVariableSize_max");
      
    AreaMetric_recoElements_maxabsEta_  = iConfig.getParameter<double>("AreaMetric_recoElements_maxabsEta");
  }
  ~PFRecoTauAlgorithm(){}
  void setTransientTrackBuilder(const TransientTrackBuilder* x){TransientTrackBuilder_=x;}
  Tau tag(const PFIsolatedTauTagInfo&,const Vertex&); 
 private:
  const TransientTrackBuilder* TransientTrackBuilder_;
  double LeadChargedHadrCand_minPt_;
  double ChargedHadrCand_minPt_;
  double NeutrHadrCand_minPt_;
  double GammaCand_minPt_;
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
  string ECALSignalConeMetric_;
  string ECALSignalConeSizeFormula_;
  double ECALSignalConeVariableSize_min_;
  double ECALSignalConeVariableSize_max_;
  string ECALIsolConeMetric_;
  string ECALIsolConeSizeFormula_;
  double ECALIsolConeVariableSize_min_;
  double ECALIsolConeVariableSize_max_;
  string HCALSignalConeMetric_;
  string HCALSignalConeSizeFormula_;
  double HCALSignalConeVariableSize_min_;
  double HCALSignalConeVariableSize_max_;
  string HCALIsolConeMetric_;
  string HCALIsolConeSizeFormula_;
  double HCALIsolConeVariableSize_min_;
  double HCALIsolConeVariableSize_max_;
  double AreaMetric_recoElements_maxabsEta_;
};
#endif 

