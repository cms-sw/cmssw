#include "JetMETCorrections/Type1MET/interface/PFJetMETcorrInputProducerT.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "PhysicsTools/PatUtils/interface/PATJetCorrExtractor.h"

namespace PFJetMETcorrInputProducer_namespace {
  template <>
  class InputTypeCheckerT<pat::Jet, PATJetCorrExtractor> {
  public:
    void operator()(const pat::Jet& jet) const {
      // check that pat::Jet is of PF-type
      if (!jet.isPFJet())
        throw cms::Exception("InvalidInput") << "Input pat::Jet is not of PF-type !!\n";
    }
    bool isPatJet(const pat::Jet& jet) const { return true; }
  };

  // template specialization for pat::Jets
  // remove JER correction if JER factor was saved as userFloat previously
  template <>
  class RawJetExtractorT<pat::Jet> {
  public:
    RawJetExtractorT() {}
    reco::Candidate::LorentzVector operator()(const pat::Jet& jet) const {
      reco::Candidate::LorentzVector uncorrected_jet;
      if (jet.jecSetsAvailable())
        uncorrected_jet = jet.correctedP4("Uncorrected");
      else
        uncorrected_jet = jet.p4();
      // remove JER correction factor from pat::Jets
      if (jet.isJerFactorValid()) {
        uncorrected_jet *= (1.0 / jet.loadJerFactor());
      }
      return uncorrected_jet;
    }
  };

  // template specialization for pat::Jets
  // retrieve JER factor if it was saved previously
  // otherwise just return 1
  template <>
  class RetrieveJerT<pat::Jet> {
  public:
    RetrieveJerT() {}
    float operator()(const pat::Jet& jet) const {
      if (jet.isJerFactorValid()) {
        return jet.loadJerFactor();
      } else
        return 1.0;
    }
  };

}  // namespace PFJetMETcorrInputProducer_namespace

typedef PFJetMETcorrInputProducerT<pat::Jet, PATJetCorrExtractor> PATPFJetMETcorrInputProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATPFJetMETcorrInputProducer);
