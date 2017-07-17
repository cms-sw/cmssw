#ifndef FWCore_Framework_ProductResolverIndexAndSkipBit_h
#define FWCore_Framework_ProductResolverIndexAndSkipBit_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::ProductResolverIndexAndSkipBit
// 
/**\class edm::ProductResolverIndexAndSkipBit

 Description:  This class holds a ProductIndexHolder and a boolean value
    to indicate whether the skipCurrentProcess option was set. Internally
    it bit packs them in an unsigned int. There is a little extra complexity
    in the implementation to hide the fact that some special values of
    ProductResolverIndex use the same bit we use for skipCurrentProcess.

 Usage:
    EDConsumerBase use this and pass a container of them to the Workers
    to let them know what data will be consumed.
*/
//
// Original Author:  W. David Dagenhart
//         Created:  3 October 2013

#include "FWCore/Utilities/interface/ProductResolverIndex.h"

namespace edm {

  class ProductResolverIndexAndSkipBit {
  public:
    ProductResolverIndexAndSkipBit(ProductResolverIndex productResolverIndex, bool skipCurrentProcess) :
      value_(skipCurrentProcess ? s_skipMask | productResolverIndex :
                                  ~s_skipMask & productResolverIndex) { }
      ProductResolverIndex productResolverIndex() const {
        bool specialIndexValue = (value_ & ProductResolverIndexValuesBit) != 0;
        return specialIndexValue ? value_ | s_skipMask :
                                   value_ & ~s_skipMask;
      }
      bool skipCurrentProcess() const { return (value_ & s_skipMask) != 0; }

      bool operator==(ProductResolverIndexAndSkipBit const& r) {
        return value_ == r.value_;
      }

  private:
      static const unsigned int s_skipMask = 1U << 31;

      unsigned int value_;
  };
}
#endif
