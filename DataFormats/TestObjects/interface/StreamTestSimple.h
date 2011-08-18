#ifndef DataFormats_TestObjects_StreamTestSimple_h
#define DataFormats_TestObjects_StreamTestSimple_h

#include "DataFormats/Common/interface/SortedCollection.h"
#include "FWCore/Utilities/interface/typedefs.h"

namespace edmtestprod {
  struct Simple {
    typedef int key_type;
    cms_int32_t key_;
    double data_;

    key_type  id() const { return key_; }
    bool operator==(Simple const&) const { return true; }
    bool operator<(Simple const&) const { return true; }
  };

  typedef edm::SortedCollection<Simple> StreamTestSimple;

  struct X0123456789012345678901234567890123456789012345678901234567890123456789012345678901 {
    int blob_;
  };

  typedef X0123456789012345678901234567890123456789012345678901234567890123456789012345678901 Pig;
}

#endif
