#ifndef PhysicsTools_UtilAlgos_EventSelectorBase_h
#define PhysicsTools_UtilAlgos_EventSelectorBase_h

/** \class EventSelectorBase
 *
 * Base-class for event selections in physics analyses
 * 
 * \author Christian Veelken, UC Davis
 *
 * \version $Revision: 1.25 $
 *
 * $Id: EventSelectorBase.h,v 1.25 2008/02/04 10:44:27 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EventSelectorBase
{
 public:
  // constructor 
  explicit EventSelectorBase() {}
  
  // destructor
  virtual ~EventSelectorBase() {}
  
  // function implementing actual cut
  // ( return value = true  : event passes cut
  //                  false : event fails cut ) 
  virtual bool operator()(edm::Event&, const edm::EventSetup&) = 0;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<EventSelectorBase* (const edm::ParameterSet&)> EventSelectorPluginFactory;

#endif
