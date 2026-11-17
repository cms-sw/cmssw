#ifndef PhysicsTools_PatAlgos_PATGenJetLinker_h
#define PhysicsTools_PatAlgos_PATGenJetLinker_h

/**
  \class    pat::PATGenJetLinker PATGenJetLinker.h "PhysicsTools/PatAlgos/interface/PATGenJetLinker.h"
  \brief    Sets pat::Jet::patGenJetRef: the pat::GenJet twin of the jet's matched gen jet

  Used by PATJetSlimmer and PATJetSelector with the optional parameter
      patGenJets = cms.VInputTag(...)   # pat::GenJet collections; absent or empty = off
  A pat::GenJet's forward link (genJetFwdRef().ref()) points to the slimmed gen jet it replaces; a jet's matched gen
  jet is its genJetFwdRef().backRef(). The twin is the pat::GenJet whose forward link is the jet's matched gen jet.
  A jet with a matched gen jet but no twin keeps no link (reported, first few times per module).
*/

#include "DataFormats/PatCandidates/interface/GenJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace pat {

  class PATGenJetLinker {
  public:
    PATGenJetLinker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC, std::string module)
        : module_(std::move(module)) {
      if (iConfig.existsAs<std::vector<edm::InputTag>>("patGenJets"))
        for (auto const& tag : iConfig.getParameter<std::vector<edm::InputTag>>("patGenJets"))
          tokens_.push_back(iC.consumes<pat::GenJetCollection>(tag));
    }

    bool enabled() const { return !tokens_.empty(); }

    /// (ProductID, key) of each slimmed gen jet -> its pat::GenJet twin, for this event
    void newEvent(const edm::Event& iEvent) {
      twin_.clear();
      for (auto const& token : tokens_) {
        edm::Handle<pat::GenJetCollection> h;
        iEvent.getByToken(token, h);
        for (unsigned int i = 0, n = h->size(); i < n; ++i) {
          auto const& fwd = (*h)[i].genJetFwdRef().ref();
          if (fwd.isNonnull())
            twin_.emplace(std::make_pair(fwd.id(), static_cast<unsigned int>(fwd.key())), pat::GenJetRef(h, i));
        }
      }
    }

    void link(pat::Jet& jet) {
      auto const& back = jet.genJetFwdRef().backRef();
      if (back.isNull())
        return;
      auto const found = twin_.find(std::make_pair(back.id(), static_cast<unsigned int>(back.key())));
      if (found != twin_.end()) {
        jet.setPatGenJetRef(found->second);
      } else if (nUnlinked_++ < 5) {
        edm::LogWarning(module_) << "jet with pt " << jet.pt() << " has a matched gen jet (product " << back.id()
                                 << ", key " << back.key()
                                 << ") without a pat::GenJet twin in the patGenJets collections";
      }
    }

    static void fillDescription(edm::ParameterSetDescription& desc) {
      desc.addOptional<std::vector<edm::InputTag>>("patGenJets")
          ->setComment("pat::GenJet collections: link each jet to the pat::GenJet twin of its matched gen jet");
    }

  private:
    const std::string module_;
    std::vector<edm::EDGetTokenT<pat::GenJetCollection>> tokens_;
    std::map<std::pair<edm::ProductID, unsigned int>, pat::GenJetRef> twin_;
    unsigned int nUnlinked_ = 0;
  };

}  // namespace pat

#endif
