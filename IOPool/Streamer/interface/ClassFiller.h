#ifndef IOPool_Streamer_ClassFiller_h
#define IOPool_Streamer_ClassFiller_h

// -*- C++ -*-

#include "Rtypes.h"

#include <typeinfo>
#include <string>
#include <set>
#include <vector>

namespace edm::streamer {
  void loadExtraClasses();
  TClass* getTClass(const std::type_info& ti);
  bool loadCap(const std::string& name, std::vector<std::string>& missingDictionaries);
  void doBuildRealData(const std::string& name);
}  // namespace edm::streamer

#endif
