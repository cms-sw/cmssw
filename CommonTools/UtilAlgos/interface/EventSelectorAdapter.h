#ifndef CommonTools_UtilAlgos_EventSelectorAdapter_h
#define CommonTools_UtilAlgos_EventSelectorAdapter_h

/** \class EventSelectorAdapter
 *
 * Provide classes derrived from EventSelectorBase with an EDFilter interface 
 * 
 * \author Christian Veelken, UC Davis
 *
 * \version $Revision: 1.1 $
 *
 * $Id: EventSelectorAdapter.h,v 1.1 2009/01/22 14:02:08 veelken Exp $
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
