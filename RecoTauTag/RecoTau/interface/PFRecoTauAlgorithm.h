#ifndef RecoTauTag_RecoTau_PFRecoTauAlgorithm_H
#define RecoTauTag_RecoTau_PFRecoTauAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithmBase.h"

class  PFRecoTauAlgorithm  : public PFRecoTauAlgorithmBase {
 public:
  PFRecoTauAlgorithm();
  PFRecoTauAlgorithm(const edm::ParameterSet&);
  ~PFRecoTauAlgorithm(){}

  // PFRecTrackCollection: Temporary until integrated to PFCandidate
  reco::PFTau buildPFTau(const reco::PFTauTagInfoRef&,const reco::Vertex&);
 private:
  bool checkPos(const std::vector<math::XYZPoint>&,const math::XYZPoint&) const;

  double   LeadPFCand_minPt_;
  double   LeadTrack_minPt_;
  bool     UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint_;
  double   ChargedHadrCandLeadChargedHadrCand_tksmaxDZ_;

  bool     UseTrackLeadTrackDZconstraint_;
  double   TrackLeadTrack_maxDZ_;
  std::string   MatchingConeMetric_;
  std::string   MatchingConeSizeFormula_;
  double   MatchingConeSize_min_;
  double   MatchingConeSize_max_;
  std::string   TrackerSignalConeMetric_;
  std::string   TrackerSignalConeSizeFormula_;
  double   TrackerSignalConeSize_min_;
  double   TrackerSignalConeSize_max_;
  std::string   TrackerIsolConeMetric_;
  std::string   TrackerIsolConeSizeFormula_;
  double   TrackerIsolConeSize_min_;
  double   TrackerIsolConeSize_max_;
  std::string   ECALSignalConeMetric_;
  std::string   ECALSignalConeSizeFormula_;
  double   ECALSignalConeSize_min_;
  double   ECALSignalConeSize_max_;
  std::string   ECALIsolConeMetric_;
  std::string   ECALIsolConeSizeFormula_;
  double   ECALIsolConeSize_min_;
  double   ECALIsolConeSize_max_;
  std::string   HCALSignalConeMetric_;
  std::string   HCALSignalConeSizeFormula_;
  double   HCALSignalConeSize_min_;
  double   HCALSignalConeSize_max_;
  std::string   HCALIsolConeMetric_;
  std::string   HCALIsolConeSizeFormula_;
  double   HCALIsolConeSize_min_;
  double   HCALIsolConeSize_max_;
  double   AreaMetric_recoElements_maxabsEta_;
  // parameters for Ellipse ... EELL
  double Rphi_;
  double MaxEtInEllipse_;
  bool AddEllipseGammas_;
  // EELL

  uint32_t ChargedHadrCand_IsolAnnulus_minNhits_;
  uint32_t Track_IsolAnnulus_minNhits_;

  // Whether or not to include the neutral hadrons in the P4
  bool putNeutralHadronsInP4_;

  std::string   DataType_;

  double   ElecPreIDLeadTkMatch_maxDR_;
  double   EcalStripSumE_minClusEnergy_;
  double   EcalStripSumE_deltaEta_;
  double   EcalStripSumE_deltaPhiOverQ_minValue_;
  double   EcalStripSumE_deltaPhiOverQ_maxValue_;
  double   maximumForElectrionPreIDOutput_;

  TFormula myMatchingConeSizeTFormula,
           myTrackerSignalConeSizeTFormula,
           myTrackerIsolConeSizeTFormula,
           myECALSignalConeSizeTFormula,
           myECALIsolConeSizeTFormula,
           myHCALSignalConeSizeTFormula,
           myHCALIsolConeSizeTFormula;

};
#endif

