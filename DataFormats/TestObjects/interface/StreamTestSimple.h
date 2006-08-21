#ifndef TestObjects_StreamTestSimple_h
#define TestObjects_StreamTestSimple_h

#include "DataFormats/Common/interface/SortedCollection.h"
//INCLUDECHECKER: Removed this line: #include <vector>

namespace edmtestprod
{
  struct Simple
  {
    typedef int key_type;
    int key_;
    double data_;

    key_type  id() const { return key_; }
    bool operator==(const Simple& s) const { return true; }
    bool operator<(const Simple& s) const { return true; }
  };

  typedef edm::SortedCollection<Simple> StreamTestSimple;

  struct X0123456789012345678901234567890123456789012345678901234567890123456789012345678901
  {
    int blob_;
  };

  typedef X0123456789012345678901234567890123456789012345678901234567890123456789012345678901 Pig;
}

#endif
