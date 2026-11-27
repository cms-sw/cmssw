#ifndef PhysicsTools_PatAlgos_plugins_PATGenJetPluginHelpers_h
#define PhysicsTools_PatAlgos_plugins_PATGenJetPluginHelpers_h
//
// Small helpers shared by the PATGenJet plugins (header-only so that the unit tests can include them).
//
#include "DataFormats/PatCandidates/interface/GenJet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>

namespace pat::genjethelpers {

  /// user-data label of an InputTag after stripping postfixes, as PATUserDataMerger.h:110-117
  inline std::string strippedLabel(edm::InputTag tag, std::vector<std::string> const& postfixes) {
    for (std::string const& stripme : postfixes) {
      auto match = tag.label().rfind(stripme);
      if (match != std::string::npos && match == (tag.label().length() - stripme.length())) {
        tag = edm::InputTag(tag.label().substr(0, match), tag.instance(), tag.process());
      }
    }
    return tag.encode();
  }

}  // namespace pat::genjethelpers

#endif
