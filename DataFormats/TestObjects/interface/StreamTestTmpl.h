#ifndef TestObjects_StreamTestTmpl_h
#define TestObjects_StreamTestTmpl_h

#include <vector>
#include "DataFormats/TestObjects/interface/StreamTestSimple.h"

namespace edmtestprod
{
  template <class T>
    struct Ord
    {
      T data_;
    };

  template <class T> //, class U = Ord<T> >
    struct StreamTestTmpl
    {
      T data_;
      //U more_;
    };

  typedef Ord<Simple> OSimple;
  //typedef Simple OSimple;
}

#endif
