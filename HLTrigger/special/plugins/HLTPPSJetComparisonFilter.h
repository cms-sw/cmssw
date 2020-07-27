#ifndef HLTPPSJetComparisonFilter_h
#define HLTPPSJetComparisonFilter_h

// Author: Mariana Araujo
// Created: 2019-12-10
/*
Description:
HLT filter module to select events according to matching of central (jets) and PPS (RP tracks) kinematics

Implementation:
Matching can be done on the xi and/or mass+rapidity variables, using the do_xi and do_my booleans. If both are set to true, both matching conditions must be met
*/

// include files
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h"

// class declaration
//
class PPSKinFilter : public edm::global::EDFilter<> {
public:
  explicit PPSKinFilter(const edm::ParameterSet&);
  ~PPSKinFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------
  edm::ParameterSet param_;

  edm::InputTag jetInputTag_;  // Input tag identifying the jet track
  edm::EDGetTokenT<reco::PFJetCollection> jet_token_;

  edm::InputTag forwardProtonInputTag_;  // Input tag identifying the forward proton collection
  edm::EDGetTokenT<std::vector<reco::ForwardProton>> recoProtonSingleRPToken_;

  std::string lhcInfoLabel_;

  double maxDiffxi_;
  double maxDiffm_;
  double maxDiffy_;

  unsigned int n_jets_;

  bool do_xi_;
  bool do_my_;
};

#endif
