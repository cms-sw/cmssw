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
