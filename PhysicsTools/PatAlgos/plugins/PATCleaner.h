#ifndef PhysicsTools_PatAlgos_plugins_PATCleaner_h
#define PhysicsTools_PatAlgos_plugins_PATCleaner_h
//
//

/**
  \class    pat::PATCleaner PATCleaner.h "PhysicsTools/PatAlgos/interface/PATCleaner.h"
  \brief    PAT Cleaner module for PAT Objects

            The same module is used for all collections.

  \author   Giovanni Petrucciani
  \version  $Id: PATCleaner.h,v 1.3 2010/10/20 23:08:30 wmtan Exp $
*/


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/PFParticle.h"

#include "PhysicsTools/PatAlgos/interface/OverlapTest.h"
#include <vector>
#include <memory>

namespace pat {

  template<class PATObjType>
  class PATCleaner : public edm::stream::EDProducer<> {
    public:
      explicit PATCleaner(const edm::ParameterSet & iConfig);
      virtual ~PATCleaner() {}

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override final;

    private:
      typedef StringCutObjectSelector<PATObjType> Selector;

      const edm::InputTag src_;
      const edm::EDGetTokenT<edm::View<PATObjType> > srcToken_;
      const Selector preselectionCut_;
      const Selector finalCut_;

      typedef pat::helper::OverlapTest OverlapTest;
      std::vector<std::unique_ptr<OverlapTest> > overlapTests_;
  };

} // namespace

template <class PATObjType>
pat::PATCleaner<PATObjType>::PATCleaner(const edm::ParameterSet & iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    srcToken_(consumes<edm::View<PATObjType> >(src_)),
    preselectionCut_(iConfig.getParameter<std::string>("preselection")),
    finalCut_(iConfig.getParameter<std::string>("finalCut"))
{
    // pick parameter set for overlaps
    edm::ParameterSet overlapPSet = iConfig.getParameter<edm::ParameterSet>("checkOverlaps");
    // get all the names of the tests (all nested PSets in this PSet)
    std::vector<std::string> overlapNames = overlapPSet.getParameterNamesForType<edm::ParameterSet>();
    // loop on them
    for (std::vector<std::string>::const_iterator itn = overlapNames.begin(); itn != overlapNames.end(); ++itn) {
        // retrieve configuration
        edm::ParameterSet cfg = overlapPSet.getParameter<edm::ParameterSet>(*itn);
        // skip empty parameter sets
        if (cfg.empty()) continue;
        // get the name of the algorithm to use
        std::string algorithm = cfg.getParameter<std::string>("algorithm");
        // create the appropriate OverlapTest
        if (algorithm == "byDeltaR") {
            overlapTests_.emplace_back(new pat::helper::BasicOverlapTest(*itn, cfg, consumesCollector()));
        } else if (algorithm == "bySuperClusterSeed") {
            overlapTests_.emplace_back(new pat::helper::OverlapBySuperClusterSeed(*itn, cfg, consumesCollector()));
        } else {
            throw cms::Exception("Configuration") << "PATCleaner for " << src_ << ": unsupported algorithm '" << algorithm << "'\n";
        }
    }


    produces<std::vector<PATObjType> >();
}

template <class PATObjType>
void
pat::PATCleaner<PATObjType>::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // Read the input. We use edm::View<> in case the input happes to be something different than a std::vector<>
  edm::Handle<edm::View<PATObjType> > candidates;
  iEvent.getByToken(srcToken_, candidates);

  // Prepare a collection for the output
  auto output = std::make_unique<std::vector<PATObjType>>();

  // initialize the overlap tests
  for (auto& itov : overlapTests_) {
    itov->readInput(iEvent,iSetup);
  }

  for (typename edm::View<PATObjType>::const_iterator it = candidates->begin(), ed = candidates->end(); it != ed; ++it) {
      // Apply a preselection to the inputs and copy them in the output
      if (!preselectionCut_(*it)) continue;

      // Add it to the list and take a reference to it, so it can be modified (e.g. to set the overlaps)
      // If at some point I'll decide to drop this item, I'll use pop_back to remove it
      output->push_back(*it);
      PATObjType &obj = output->back();

      // Look for overlaps
      bool badForOverlap = false;
      for (auto& itov : overlapTests_) {
        reco::CandidatePtrVector overlaps;
        bool hasOverlap = itov->fillOverlapsForItem(obj, overlaps);
        if (hasOverlap && itov->requireNoOverlaps()) {
            badForOverlap = true; // mark for discarding
            break; // no point in checking the others, as this item will be discarded
        }
        obj.setOverlaps(itov->name(), overlaps);
      }
      if (badForOverlap) { output->pop_back(); continue; }

      // Apply one final selection cut
      if (!finalCut_(obj)) output->pop_back();
  }

  iEvent.put(std::move(output));
}


#endif
