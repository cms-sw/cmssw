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

  template <>
  class RawJetExtractorT<pat::Jet> {
  public:
    RawJetExtractorT() {}
    reco::Candidate::LorentzVector operator()(const pat::Jet& jet) const {
      if (jet.jecSetsAvailable())
        return jet.correctedP4(0);
      else
        return jet.p4();
    }
  };

  // template specialization for pat::Jets
  // retrieve combined factor of additional scales applied to the jets
  // otherwise just return 1
  template <>
  class AdditionalScalesT<pat::Jet> {
  public:
    AdditionalScalesT() {}
    float operator()(const pat::Jet& jet) const {
      if (jet.jecSetsAvailable()) {
        return jet.jecFactor("Uncorrected") / jet.jecFactor(0);
      } else
        return 1.0;
    }
  };
}  // namespace PFJetMETcorrInputProducer_namespace

typedef PFJetMETcorrInputProducerT<pat::Jet, PATJetCorrExtractor> PATPFJetMETcorrInputProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATPFJetMETcorrInputProducer);
