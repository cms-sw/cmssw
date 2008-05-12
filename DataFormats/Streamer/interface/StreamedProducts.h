#ifndef Streamer_StreamedProducts_h
#define Streamer_StreamedProducts_h

/*
  Simple packaging of all the event data that is needed to be serialized
  for transfer.

  The "other stuff in the SendEvent still needs to be
  populated.

  The product is paired with its provenance, and the entire event
  is captured in the SendEvent structure.
 */

#include <vector>

#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/EventID.h"

namespace edm {

  // ------------------------------------------

  class BranchDescription;
  class EventEntryInfo;
  class EDProduct;
  struct ProdPair
  {
    ProdPair():
      prod_(),prov_(),desc_() { }
    explicit ProdPair(const EDProduct* p):
      prod_(p),prov_(),desc_() { }
    ProdPair(const EDProduct* prod,
	     const Provenance* prov):
      prod_(prod),
      prov_(&prov->branchEntryInfo()),
      desc_(&prov->product()) { }

    const EventEntryInfo* prov() const { return prov_; }
    const EDProduct* prod() const { return prod_; }
    const BranchDescription* desc() const { return desc_; }

    void clear() { prod_=0; prov_=0; desc_=0; }

    const EDProduct* prod_;
    const EventEntryInfo* prov_;
    const BranchDescription* desc_;
  };

  // ------------------------------------------

  typedef std::vector<ProdPair> SendProds;

  // ------------------------------------------

  struct SendEvent
  {
    SendEvent() { }
    SendEvent(const EventID& id, const Timestamp& t):id_(id),time_(t) { }

    EventID id_;
    Timestamp time_;
    SendProds prods_;

    // other tables necessary for provenance lookup
  };

  typedef std::vector<BranchDescription> SendDescs;

  struct SendJobHeader
  {
    SendJobHeader() { }

    SendDescs descs_;
    // trigger bit descriptions will be added here and permanent
    //  provenance values
  };


}
#endif

