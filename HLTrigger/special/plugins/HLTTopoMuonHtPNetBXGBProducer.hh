/** \class HLTTopoMuonHtPNetBXGBProducer
 *
 *  This class is an EDProducer that produces a single float value corresponding to the output score of an XGBoost model
 *  of a "topological trigger" (TOPO) for events with at least one muon + HT and b-tag. 
 *  The model takes as input the PFHT, 
 *  the maximum PNetB score among jets in the event, 
 *  and the pt and isolation variables of up to N muons (configurable).
 *
 *  \author Artur Lobanov – University of Hamburg
 */

#ifndef HLTrigger_Muon_HLTTopoMuonHtPNetBXGBProducer_h
#define HLTrigger_Muon_HLTTopoMuonHtPNetBXGBProducer_h

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToValue.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "xgboost/c_api.h"

class HLTTopoMuonHtPNetBXGBProducer : public edm::stream::EDProducer<> {
 public:
  using RecoChargedCandMap = edm::AssociationMap<edm::OneToValue<
      std::vector<reco::RecoChargedCandidate>, float, unsigned int>>;

  // 2 global features (PFHT, MaxPNetB) + 4 features per muon
  static constexpr unsigned int kGlobalFeatures = 2;
  static constexpr unsigned int kFeaturesPerMuon = 4;

  explicit HLTTopoMuonHtPNetBXGBProducer(edm::ParameterSet const&);
  ~HLTTopoMuonHtPNetBXGBProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

 private:
  void produce(edm::Event&, edm::EventSetup const&) override;

  /* Tokens */
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection>
      chargedCandidatesToken_;
  edm::EDGetTokenT<RecoChargedCandMap> ecalIsoMapToken_;
  edm::EDGetTokenT<RecoChargedCandMap> hcalIsoMapToken_;
  edm::EDGetTokenT<edm::ValueMap<double>> trackIsoMapToken_;
  edm::EDGetTokenT<reco::METCollection> pfhtToken_;
  edm::EDGetTokenT<reco::JetTagCollection> pnetToken_;

  /* Cuts */
  double muonPtCut_;
  double muonEtaCut_;

  /* Config */
  unsigned int nMuons_;     // number of muons used as input features
  unsigned int nFeatures_;  // kGlobalFeatures + kFeaturesPerMuon * nMuons_
  bool muonSortByTkIso_;    // if true: ascending tkiso; if false: descending pt

  /* XGBoost */
  BoosterHandle booster_ = nullptr;
  DMatrixHandle dmat_ = nullptr;
  std::vector<float> buffer_;
  std::string xgbConfig_;

  bool debug_;
};

#endif