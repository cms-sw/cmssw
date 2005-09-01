#ifndef StreamerClassFiller_h
#define StreamerClassFiller_h

#include "FWCore/Framework/interface/ProductRegistry.h"
#include "StorageSvc/IClassLoader.h"
#include "TClass.h"
#include <typeinfo>

namespace edm
{
  pool::IClassLoader* getClassLoader();
  void fillStreamers(ProductRegistry const& reg);
  TClass* getTClass(const std::type_info& ti);
  TClass* loadClass(pool::IClassLoader* cl, const std::type_info& ti);
  void loadExtraClasses();
}

#endif

