#ifndef EventFilter_FilterByLTC_h
#define EventFilter_FilterByLTC_h

/** \class FilterByLTC
 *
 *  Class to select events depending on the trigger source
 *  (DT,CSC,RPC,DT+CSC,DT+RPC,CSC+RPC,DT+CSC+RPC,NoDT,NoCSC,NoRPC)
 *
 *  $Date: 2012/09/13 14:53:56 $
 *  $Revision: 1.3 $
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class FilterByLTC : public edm::EDFilter {
 public:
  /// Constructor
  FilterByLTC(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~FilterByLTC();
  
  virtual bool filter(edm::Event & event, const edm::EventSetup& eventSetup);

 private:
  // counters
  int nEventsProcessed;
  int nEventsSelected;
  //trigger source 1,...,13 = (
  // only DT, CSC, RPC, 
  // both DT&&CSC,DT&&RPC,CSC&&RPC,CSC&&RPC&&DT, 
  // NoDT,NoCSC,NoRPC
  // DT+anything else, CSC+anything else, RPC+anything else)
  int theTriggerSource;
  edm::InputTag ltcTag_;
};
#endif
