#ifndef Workspace_EventSelector_h_
#define Workspace_EventSelector_h_
/** Base class for event selection modules for SUSY analysis.
 */
// Original author: W. Adam, 10/4/08

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EventSelector {
public:
  EventSelector () {}
  EventSelector (const edm::ParameterSet& iConfig, edm::ConsumesCollector && iC) :
    EventSelector(iConfig, iC) {}
  EventSelector (const edm::ParameterSet& iConfig, edm::ConsumesCollector & iC) {
    std::string selector = iConfig.getParameter<std::string>("selector");
    name_ = iConfig.getUntrackedParameter<std::string>("name",selector);
  }
  virtual ~EventSelector () {}
  /// name of the module (from configuration)
  const std::string& name () const {return name_;}
  const std::vector<std::string> & description() { return description_;}
  /// decision of the selector module
  virtual bool select (const edm::Event&) const = 0;

 protected:
  std::string name_;
  std::vector<std::string> description_;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory< EventSelector* (const edm::ParameterSet&, edm::ConsumesCollector &&) > EventSelectorFactory;
typedef edmplugin::PluginFactory< EventSelector* (const edm::ParameterSet&, edm::ConsumesCollector &) > EventSelectorFactoryFromHelper;

#endif
