#ifndef Fireworks_Core_FWItemTVirtualCollectionProxyAccessor_h
#define Fireworks_Core_FWItemTVirtualCollectionProxyAccessor_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemTVirtualCollectionProxyAccessor
//
/**\class FWItemTVirtualCollectionProxyAccessor FWItemTVirtualCollectionProxyAccessor.h Fireworks/Core/interface/FWItemTVirtualCollectionProxyAccessor.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 18 08:43:45 EDT 2008
//

// system include files
#include <memory>

// user include files
#include "Fireworks/Core/interface/FWItemAccessorBase.h"

// forward declarations
class TVirtualCollectionProxy;

class FWItemTVirtualCollectionProxyAccessor : public FWItemAccessorBase {
public:
  FWItemTVirtualCollectionProxyAccessor(const TClass* iType,
                                        std::shared_ptr<TVirtualCollectionProxy> iProxy,
                                        size_t iOffset = 0);
  ~FWItemTVirtualCollectionProxyAccessor() override;

  // ---------- const member functions ---------------------
  const void* modelData(int iIndex) const override;
  const void* data() const override;
  unsigned int size() const override;
  const TClass* modelType() const override;
  const TClass* type() const override;

  bool isCollection() const override;

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void setData(const edm::ObjectWithDict&) override;
  void reset() override;

  FWItemTVirtualCollectionProxyAccessor(const FWItemTVirtualCollectionProxyAccessor&) = delete;  // stop default

  const FWItemTVirtualCollectionProxyAccessor& operator=(const FWItemTVirtualCollectionProxyAccessor&) =
      delete;  // stop default

private:
  // ---------- member data --------------------------------
  const TClass* m_type;
  std::shared_ptr<TVirtualCollectionProxy> m_colProxy;  //should be something other than shared_ptr
  mutable const void* m_data;
  size_t m_offset;
};

#endif
