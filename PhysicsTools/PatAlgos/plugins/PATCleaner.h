#ifndef PhysicsTools_PatAlgos_plugins_PATCleaner_h
#define PhysicsTools_PatAlgos_plugins_PATCleaner_h
//
// $Id: PATCleaner.h,v 1.4 2013/02/27 23:26:56 wmtan Exp $
//

/**
  \class    pat::PATCleaner PATCleaner.h "PhysicsTools/PatAlgos/interface/PATCleaner.h"
  \brief    PAT Cleaner module for PAT Objects
            
            The same module is used for all collections.

  \author   Giovanni Petrucciani
  \version  $Id: PATCleaner.h,v 1.4 2013/02/27 23:26:56 wmtan Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
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
#include <boost/ptr_container/ptr_vector.hpp>

namespace pat {

  template<class PATObjType>
  class PATCleaner : public edm::EDProducer {
    public:
      explicit PATCleaner(const edm::ParameterSet & iConfig);
      virtual ~PATCleaner() {}

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

    private:
      typedef StringCutObjectSelector<PATObjType> Selector;

      edm::InputTag src_;
      bool doPreselection_, doFinalCut_;  
      Selector preselectionCut_;
      Selector finalCut_;

      typedef pat::helper::OverlapTest OverlapTest;
      // make a list of overlap tests (ptr_vector instead of std::vector because they're polymorphic)
      boost::ptr_vector<OverlapTest> overlapTests_;
  };

} // namespace

template <class PATObjType>
pat::PATCleaner<PATObjType>::PATCleaner(const edm::ParameterSet & iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
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
            overlapTests_.push_back(new pat::helper::BasicOverlapTest(*itn, cfg));
        } else if (algorithm == "bySuperClusterSeed") {
            overlapTests_.push_back(new pat::helper::OverlapBySuperClusterSeed(*itn, cfg));
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
  iEvent.getByLabel(src_, candidates);

  // Prepare a collection for the output
  std::auto_ptr< std::vector<PATObjType> > output(new std::vector<PATObjType>());

  // initialize the overlap tests
  for (boost::ptr_vector<OverlapTest>::iterator itov = overlapTests_.begin(), edov = overlapTests_.end(); itov != edov; ++itov) {
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
      for (boost::ptr_vector<OverlapTest>::iterator itov = overlapTests_.begin(), edov = overlapTests_.end(); itov != edov; ++itov) {
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

  iEvent.put(output);
}


#endif
