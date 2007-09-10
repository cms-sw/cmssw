#ifndef RecoTauTag_RecoTau_PFRecoTauAlgorithm_H
#define RecoTauTag_RecoTau_PFRecoTauAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoTauTag/TauTagTools/interface/PFTauElementsOperators.h"
#include "RecoTauTag/TauTagTools/interface/CaloTauElementsOperators.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

using namespace std;
using namespace reco;
using namespace edm;

class  PFRecoTauAlgorithm  {
 public:
  PFRecoTauAlgorithm();
  PFRecoTauAlgorithm(const ParameterSet&);
  ~PFRecoTauAlgorithm(){}
  void setTransientTrackBuilder(const TransientTrackBuilder*);
  PFTau buildPFTau(const PFTauTagInfoRef&,const Vertex&); 
 private:
  const TransientTrackBuilder* TransientTrackBuilder_;
  double LeadChargedHadrCand_minPt_;
  double ChargedHadrCand_minPt_;
  bool UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint_;
  double ChargedHadrCandLeadChargedHadrCand_tksmaxDZ_;
  double NeutrHadrCand_minPt_;
  double GammaCand_minPt_;
  double LeadTrack_minPt_;
  double Track_minPt_;
  bool UseTrackLeadTrackDZconstraint_;
  double TrackLeadTrack_maxDZ_;
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

