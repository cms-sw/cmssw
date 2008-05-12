#ifndef RecoTauTag_RecoTau_PFRecoTauAlgorithm_H
#define RecoTauTag_RecoTau_PFRecoTauAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoTauTag/TauTagTools/interface/PFTauElementsOperators.h"
#include "RecoTauTag/TauTagTools/interface/CaloTauElementsOperators.h"
#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

using namespace std;
using namespace reco;
using namespace edm;

class  PFRecoTauAlgorithm  {
 public:
  PFRecoTauAlgorithm();
  PFRecoTauAlgorithm(const ParameterSet&);
  ~PFRecoTauAlgorithm(){}
  void setTransientTrackBuilder(const TransientTrackBuilder*);
  void setMagneticField(const MagneticField*); 
  
  // PFRecTrackCollection: Temporary until integrated to PFCandidate
  PFTau buildPFTau(const PFTauTagInfoRef&,const Vertex&,const PFRecTrackCollection&); 
 private:
  const TransientTrackBuilder* TransientTrackBuilder_;
  bool checkPos(std::vector<math::XYZPoint>,math::XYZPoint) const;

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
  double MatchingConeSize_min_;
  double MatchingConeSize_max_;
  string TrackerSignalConeMetric_;
  string TrackerSignalConeSizeFormula_;
  double TrackerSignalConeSize_min_;
  double TrackerSignalConeSize_max_;
  string TrackerIsolConeMetric_;
  string TrackerIsolConeSizeFormula_;
  double TrackerIsolConeSize_min_;
  double TrackerIsolConeSize_max_;
  string ECALSignalConeMetric_;
  string ECALSignalConeSizeFormula_;
  double ECALSignalConeSize_min_;
  double ECALSignalConeSize_max_;
  string ECALIsolConeMetric_;
  string ECALIsolConeSizeFormula_;
  double ECALIsolConeSize_min_;
  double ECALIsolConeSize_max_;
  string HCALSignalConeMetric_;
  string HCALSignalConeSizeFormula_;
  double HCALSignalConeSize_min_;
  double HCALSignalConeSize_max_;
  string HCALIsolConeMetric_;
  string HCALIsolConeSizeFormula_;
  double HCALIsolConeSize_min_;
  double HCALIsolConeSize_max_;
  double AreaMetric_recoElements_maxabsEta_;

  uint32_t ChargedHadrCand_IsolAnnulus_minNhits_;
  uint32_t Track_IsolAnnulus_minNhits_;

  const MagneticField* MagneticField_;
  double ElecPreIDLeadTkMatch_maxDR_;
  double EcalStripSumE_minClusEnergy_;
  double EcalStripSumE_deltaEta_;
  double EcalStripSumE_deltaPhiOverQ_minValue_;
  double EcalStripSumE_deltaPhiOverQ_maxValue_;
  

};
#endif 

