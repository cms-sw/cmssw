#ifndef FWCore_Framework_test_TestTypeResolvers_h
#define FWCore_Framework_test_TestTypeResolvers_h

#include "FWCore/Framework/interface/ModuleTypeResolverBase.h"

#include <string>
#include <utility>

namespace edm::test {
  class SimpleTestTypeResolver : public edm::ModuleTypeResolverBase {
  public:
    SimpleTestTypeResolver() = default;
    std::pair<std::string, int> resolveType(std::string basename, int index) const final {
      return {basename, kLastIndex};
    }
  };

  class ComplexTestTypeResolver : public edm::ModuleTypeResolverBase {
  public:
    ComplexTestTypeResolver() = default;
    std::pair<std::string, int> resolveType(std::string basename, int index) const final {
      constexpr auto kGeneric = "generic::";
      constexpr auto kOther = "edm::test::other::";
      constexpr auto kCPU = "edm::test::cpu::";
      if (index != kInitialIndex and index != kLastIndex) {
        basename.replace(basename.find(kOther), strlen(kOther), kCPU);
        return {basename, kLastIndex};
      }
      if (index == kInitialIndex and basename.find(kGeneric) != std::string::npos) {
        basename.replace(basename.find(kGeneric), strlen(kGeneric), kOther);
        return {basename, kInitialIndex + 1};
      }
      return {basename, kLastIndex};
    }
  };
}  // namespace edm::test

#endif
