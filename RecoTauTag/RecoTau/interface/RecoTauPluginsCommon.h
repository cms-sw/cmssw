#ifndef RecoTauTag_RecoTau_RecoTauPluginsCommon_h
#define RecoTauTag_RecoTau_RecoTauPluginsCommon_h

/*
 * RecoTauPluginsCommon
 *
 * Common base classes for the plugins used in the
 * the PFTau production classes.  The named plugin simply
 * assigns a name to each plugin, that is retrieved by the
 * ParameterSet passed to the constructor.  The EventHolder
 * plugin manages plugins that might need access to event data.
 *
 * The setup(...) method should be called for each event to cache
 * pointers to the edm::Event and edm::EventSetup.  Derived classes
 * can access this information through the evt() and evtSetup() methods.
 * The virtual function beginEvent() is provided as a hook for derived
 * classes to do setup tasks and is called by RecoTauEventHolderPlugin
 * after setup is called.
 *
 * Author: Evan K. Friis, UC Davis
 *
 * $Id $
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

namespace reco { namespace tau {

class RecoTauNamedPlugin {
   /* Base class for all named RecoTau plugins */
   public:
      explicit RecoTauNamedPlugin(const edm::ParameterSet& pset);
      virtual ~RecoTauNamedPlugin() {}
      const std::string& name() const;
   private:
      std::string name_;
};

class RecoTauEventHolderPlugin : public RecoTauNamedPlugin {
   /* Base class for all plugins that cache the edm::Event and edm::EventSetup
    * as internal data members */
   public:
      explicit RecoTauEventHolderPlugin(const edm::ParameterSet& pset);
      virtual ~RecoTauEventHolderPlugin() {}
      // Get the internal cached copy of the event
      const edm::Event* evt() const;
      edm::Event* evt();
      const edm::EventSetup* evtSetup() const;
      // Cache a local pointer to the event and event setup
      void setup(edm::Event&, const edm::EventSetup&);
      // Called after setup(...)
      virtual void beginEvent() {}
   private:
      edm::Event* evt_;
      const edm::EventSetup* es_;
};
}} // end namespace reco::tau
#endif
