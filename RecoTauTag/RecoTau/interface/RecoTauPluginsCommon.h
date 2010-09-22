#ifndef RecoTauTag_RecoTau_RecoTauPluginsCommon_h
#define RecoTauTag_RecoTau_RecoTauPluginsCommon_h

/*
 * Common base classes for the plugins used in the
 * the PFTau production classes.
 *
 * Author: Evan K. Friis, UC Davis
 *
 * $Id $
 */


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

namespace reco {

   namespace tau {

      class RecoTauNamedPlugin
      {
         /* Base class for all named RecoTau plugins */
         public:
            explicit RecoTauNamedPlugin(const edm::ParameterSet& pset);
            virtual ~RecoTauNamedPlugin() {}
            const std::string& name() const;
         private:
            std::string name_;
      };

      class RecoTauEventHolderPlugin : public RecoTauNamedPlugin
      {
         /* Base class for all plugins that cache the edm::Event and edm::EventSetup
          * as internal data members */
         public:
            explicit RecoTauEventHolderPlugin(const edm::ParameterSet& pset);
            virtual ~RecoTauEventHolderPlugin() {}

            // Get the internal cached copy of the event
            const edm::Event* evt() const;
            const edm::EventSetup* evtSetup() const;
            void setup(const edm::Event&, const edm::EventSetup&);
            // Should be at the beginning of each event (after setup(...))
            virtual void beginEvent() {}
         private:
            const edm::Event* evt_;
            const edm::EventSetup* es_;
      };
   }
}

#endif
