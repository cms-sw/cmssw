#ifndef PhysicsTools_PatUtils_PATJetCorrExtractor_h
#define PhysicsTools_PatUtils_PATJetCorrExtractor_h

/** \class PATJetCorrExtractor
 *
 * Retrieve jet energy correction factor for pat::Jets (of either PF-type or Calo-type)
 *
 * NOTE: this specialization of the "generic" template defined in
 *         JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h
 *       is to be used for pat::Jets only
 *
 * \author Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>
#include <vector>

class PATJetCorrExtractor {
public:
  reco::Candidate::LorentzVector operator()(
      const pat::Jet rawJet,
      const reco::JetCorrector* jetCorr,
      double jetCorrEtaMax = 9.9,
      const reco::Candidate::LorentzVector* const rawJetP4_specified = nullptr) const {
    JetCorrExtractorT<pat::Jet> jetCorrExtractor;
    return jetCorrExtractor(rawJet, jetCorr, jetCorrEtaMax, rawJetP4_specified);
  }

  reco::Candidate::LorentzVector operator()(
      const pat::Jet& jet,
      const std::string& jetCorrLabel,
      double jetCorrEtaMax = 9.9,
      const reco::Candidate::LorentzVector* const rawJetP4_specified = nullptr) const {
    reco::Candidate::LorentzVector corrJetP4;

    try {
      corrJetP4 = jet.correctedP4(jetCorrLabel);
      if (rawJetP4_specified != nullptr) {
        //MM: compensate for potential removal of constituents (as muons)
        //similar effect in JetMETCorrection/Type1MET/interface/JetCorrExtractor.h
        reco::Candidate::LorentzVector rawJetP4 = jet.correctedP4("Uncorrected");
        double corrFactor = corrJetP4.pt() / rawJetP4.pt();
        corrJetP4 = (*rawJetP4_specified);
        corrJetP4 *= corrFactor;
        if (corrFactor < 0) {
          edm::LogWarning("PATJetCorrExtractor") << "Negative jet energy scale correction noticed"
                                                 << ".\n";
        }
      }
    } catch (cms::Exception const&) {
      throw cms::Exception("InvalidRequest")
          << "The JEC level " << jetCorrLabel << " does not exist !!\n"
          << "Available levels = { " << format_vstring(jet.availableJECLevels()) << " }.\n";
    }

    return corrJetP4;
  }

private:
  static std::string format_vstring(const std::vector<std::string>& v) {
    std::string retVal;
    auto ss = std::accumulate(v.begin(), v.end(), 0, [](int a, std::string const& s) { return a + s.length() + 2; });
    retVal.reserve(ss);
    for_each(v.begin(), v.end(), [&](std::string const& s) { retVal += (retVal.empty() ? "" : ", ") + s; });
    return retVal;
  }
};

#endif
