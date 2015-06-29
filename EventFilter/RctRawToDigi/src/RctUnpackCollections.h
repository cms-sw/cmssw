#ifndef RctUnpackCollections_h
#define RctUnpackCollections_h

/*!
* \class RctUnpackCollections
*
*/ 

// CMSSW headers
#include "FWCore/Framework/interface/Event.h"

// DataFormat headers
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1Trigger/interface/L1TriggerError.h"
#include "TList.h"

class RctUnpackCollections
{
public:
  /// Construct with an event. The collections get put into the event when the object instance goes out of scope (i.e. in the destructor).
  RctUnpackCollections(edm::Event& event);

  /// Destructor - the last action of this object is to put the rct collections into the event provided on construction.
  ~RctUnpackCollections();

  // Collections for storing RCT input data.  
  L1CaloEmCollection * const rctEm() const { return m_rctEm.get(); } ///< Input electrons from the RCT to the RCT.
  L1CaloRegionCollection * const rctCalo() const { return m_rctCalo.get(); } ///< Input calo regions from the RCT to the RCT.

private:

  RctUnpackCollections(const RctUnpackCollections&); ///< Copy ctor - deliberately not implemented!
  RctUnpackCollections& operator=(const RctUnpackCollections&); ///< Assignment op - deliberately not implemented!  

  edm::Event& m_event;  ///< The event the collections will be put into on destruction of the RctUnpackCollections instance.

  // Collections for storing RCT input data.  
  std::auto_ptr<L1CaloEmCollection> m_rctEm; ///< Input electrons.
  std::auto_ptr<L1CaloRegionCollection> m_rctCalo; ///< Input calo regions.

};

// Pretty print for the RctUnpackCollections sub-class
std::ostream& operator<<(std::ostream& os, const RctUnpackCollections& rhs);

#endif /* RctUnpackCollections_h */
