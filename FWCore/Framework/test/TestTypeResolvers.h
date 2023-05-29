#ifndef FWCore_Framework_test_TestTypeResolvers_h
#define FWCore_Framework_test_TestTypeResolvers_h

#include "FWCore/Framework/interface/ModuleTypeResolverBase.h"
#include "FWCore/Framework/interface/ModuleTypeResolverMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace edm::test {
  class SimpleTestTypeResolver : public edm::ModuleTypeResolverBase {
  public:
    SimpleTestTypeResolver() = default;
    std::pair<std::string, int> resolveType(std::string basename, int index) const final {
      return {basename, kLastIndex};
    }
  };
  class SimpleTestTypeResolverMaker : public edm::ModuleTypeResolverMaker {
  public:
    SimpleTestTypeResolverMaker() = default;
    std::shared_ptr<ModuleTypeResolverBase const> makeResolver(edm::ParameterSet const&) const final {
      return std::make_shared<SimpleTestTypeResolver>();
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
  class ComplexTestTypeResolverMaker : public edm::ModuleTypeResolverMaker {
  public:
    ComplexTestTypeResolverMaker() = default;
    std::shared_ptr<ModuleTypeResolverBase const> makeResolver(edm::ParameterSet const&) const final {
      return std::make_shared<ComplexTestTypeResolver>();
    }
  };

  class ConfigurableTestTypeResolver : public edm::ModuleTypeResolverBase {
  public:
    ConfigurableTestTypeResolver(std::string variant) : variant_(std::move(variant)) {}
    std::pair<std::string, int> resolveType(std::string basename, int index) const final {
      constexpr auto kGeneric = "generic::";
      constexpr auto kOther = "edm::test::other::";
      constexpr auto kCPU = "edm::test::cpu::";
      if (index != kInitialIndex and index != kLastIndex) {
        basename.replace(basename.find(kOther), strlen(kOther), kCPU);
        return {basename, kLastIndex};
      }
      if (index == kInitialIndex and basename.find(kGeneric) != std::string::npos) {
        if (not variant_.empty()) {
          if (variant_ == "other") {
            basename.replace(basename.find(kGeneric), strlen(kGeneric), kOther);
          } else if (variant_ == "cpu") {
            basename.replace(basename.find(kGeneric), strlen(kGeneric), kCPU);
          }
          return {basename, kLastIndex};
        }
        basename.replace(basename.find(kGeneric), strlen(kGeneric), kOther);
        return {basename, kInitialIndex + 1};
      }
      return {basename, kLastIndex};
    }

  private:
    std::string const variant_;
  };
  class ConfigurableTestTypeResolverMaker : public edm::ModuleTypeResolverMaker {
  public:
    ConfigurableTestTypeResolverMaker() = default;
    std::shared_ptr<ModuleTypeResolverBase const> makeResolver(edm::ParameterSet const& pset) const final {
      std::string variant;
      if (pset.existsAs<std::string>("variant", false)) {
        variant = pset.getUntrackedParameter<std::string>("variant");
        if (variant != "other" and variant != "cpu") {
          throw edm::Exception(edm::errors::Configuration) << "Variant must be 'other' or 'cpu'. Got " << variant;
        }
      }
      auto found = cache_.find(variant);
      if (found == cache_.end()) {
        bool inserted;
        std::tie(found, inserted) = cache_.emplace(variant, std::make_shared<ConfigurableTestTypeResolver>(variant));
      }
      return found->second;
    }

  private:
    // no protection needed because this object is used only in single-thread context
    CMS_SA_ALLOW mutable std::unordered_map<std::string, std::shared_ptr<ConfigurableTestTypeResolver>> cache_;
  };
}  // namespace edm::test

#endif
