#ifndef FWCore_Reflection_DataProductReflectionInfoRegistry_h
#define FWCore_Reflection_DataProductReflectionInfoRegistry_h
// -*- C++ -*-
//
// Package:     FWCore/Reflection
// Class  :     DataProductReflectionInfoRegistry
//
/**\class DataProductReflectionInfoRegistry DataProductReflectionInfoRegistry.h "FWCore/Reflection/interface/DataProductReflectionInfoRegistry.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 27 Jul 2022 21:00:33 GMT
//

// system include files
#include <typeinfo>
#include <typeindex>
#include "oneapi/tbb/concurrent_unordered_map.h"

// user include files
#include "FWCore/Reflection/interface/DataProductReflectionInfo.h"

// forward declarations
namespace edm {
  class DataProductReflectionInfoRegistry {
  public:
    ~DataProductReflectionInfoRegistry();

    DataProductReflectionInfoRegistry(const DataProductReflectionInfoRegistry&) = delete;  // stop default
    const DataProductReflectionInfoRegistry& operator=(const DataProductReflectionInfoRegistry&) =
        delete;  // stop default

    // ---------- const member functions ---------------------
    DataProductReflectionInfo const* findType(std::type_index) const;

    // ---------- static member functions --------------------
    static DataProductReflectionInfoRegistry& instance();

    // ---------- member functions ---------------------------
    void registerDataProduct(std::type_index, DataProductReflectionInfo);

  private:
    // ---------- member data --------------------------------
    DataProductReflectionInfoRegistry();

    oneapi::tbb::concurrent_unordered_map<std::type_index, DataProductReflectionInfo const> m_registry;
  };

  template <typename T>
  struct RegisterDataProductReflectionInfo {
    RegisterDataProductReflectionInfo() {
      DataProductReflectionInfoRegistry::instance().registerDataProduct(std::type_index(typeid(T)),
                                                                        makeDataProductReflectionInfo<T>());
    }
  };

}  // namespace edm

#define EDM_DATAPRODUCTINFO_SYM(x, y) EDM_DATAPRODUCTINFO_SYM2(x, y)
#define EDM_DATAPRODUCTINFO_SYM2(x, y) x##y

#define DEFINE_DATA_PRODUCT_INFO(type, ...)                                                             \
  namespace edm {                                                                                       \
    template <>                                                                                         \
    struct StaticDataProductReflection<type>                                                            \
        : public StaticDataProductReflectionBase<type __VA_OPT__(, ) __VA_ARGS__> {};                   \
    static const RegisterDataProductReflectionInfo<type> EDM_DATAPRODUCTINFO_SYM(s_registry, __LINE__); \
  }                                                                                                     \
  using require_semicolon = int

#endif
