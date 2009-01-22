#ifndef PhysicsTools_UtilAlgos_EventSelectorAdapter_h
#define PhysicsTools_UtilAlgos_EventSelectorAdapter_h

/** \class EventSelectorAdapter
 *
 * Provide classes derrived from EventSelectorBase with an EDFilter interface 
 * 
 * \author Christian Veelken, UC Davis
 *
 * \version $Revision: 1.25 $
 *
 * $Id: Cut.h,v 1.25 2008/02/04 10:44:27 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

template<typename T>
class EventSelectorAdapter : public edm::EDFilter 
{
 public:
  // constructor 
  explicit EventSelectorAdapter(const edm::ParameterSet& cfg) : 
    eventSelector_( cfg ) {
  }
    
  // destructor
  virtual ~EventSelectorAdapter() {}
    
 private:
  bool filter(edm::Event& evt, const edm::EventSetup& es) { return eventSelector_(evt, es); } 
  
  T eventSelector_;
};

#endif
