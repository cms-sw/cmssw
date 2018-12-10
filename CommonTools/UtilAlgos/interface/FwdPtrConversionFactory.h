#ifndef CommonTools_UtilAlgos_FwdPtrConversionFactory_h
#define CommonTools_UtilAlgos_FwdPtrConversionFactory_h


/**
  \class    "CommonTools/UtilAlgos/interface/FwdPtrConversionFactory.h"
  \brief    Converts back and forth from FwdPtr to instances. 


  \author   Salvatore Rappoccio
*/

#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"


namespace edm {
  /// Factory class template for how to produce products
  /// from a FwdPtr. This particular example is for copy
  /// construction, but the same signature can be used elsewhere. 
  template<class T>
  struct ProductFromFwdPtrFactory {
    T operator() (edm::FwdPtr<T> const &r)  const  { return T(*r); }
  };




  /// Factory class template for how to produce FwdPtrs
  /// from a View. 
  template<class T>
  struct FwdPtrFromProductFactory {
    edm::FwdPtr<T> operator() (edm::View<T> const & view, unsigned int i)  const  { return edm::FwdPtr<T>(view.ptrAt(i),view.ptrAt(i)); }
  };



}

#endif
