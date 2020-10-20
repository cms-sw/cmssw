#ifndef FWCore_ParameterSet_ParameterSetDescriptionFiller_h
#define FWCore_ParameterSet_ParameterSetDescriptionFiller_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterSetDescriptionFiller
//
/**\class ParameterSetDescriptionFiller ParameterSetDescriptionFiller.h FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h

 Description: A concrete ParameterSetDescription filler which calls a static function of the template argument

 Usage:
    This is an ParameterSetDescription filler adapter class which calls the

void fillDescription(edm::ParameterSetDescription&)

method of the templated argument.  This allows the ParameterSetDescriptionFillerPluginFactory to communicate with existing plugins.

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Aug  1 16:46:56 EDT 2007
//

#include <type_traits>
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

namespace edm {
  template <typename T>
  class ParameterSetDescriptionFiller : public ParameterSetDescriptionFillerBase {
  public:
    ParameterSetDescriptionFiller() {}
    ParameterSetDescriptionFiller(const ParameterSetDescriptionFiller&) = delete;                   // stop default
    const ParameterSetDescriptionFiller& operator=(const ParameterSetDescriptionFiller&) = delete;  // stop default

    void fill(ConfigurationDescriptions& descriptions) const override {
      T::fillDescriptions(descriptions);
      T::prevalidate(descriptions);
    }

    const std::string& baseType() const override { return T::baseType(); }

    const std::string& extendedBaseType() const override {
      const T* type = nullptr;
      return ParameterSetDescriptionFillerBase::extendedBaseType(type);
    }
  };

  // We need a special version of this class for Services because there is
  // no common base class for all Service classes.  This means we cannot define
  // the baseType and fillDescriptions functions for all Service classes without
  // great difficulty.

  // First, some template metaprogramming to determining if the class T has
  // a fillDescriptions function.

  namespace fillDetails {

    using no_tag = std::false_type;  // type indicating FALSE
    using yes_tag = std::true_type;  // type indicating TRUE

    template <typename T, void (*)(ConfigurationDescriptions&)>
    struct fillDescriptions_function;
    template <typename T>
    no_tag has_fillDescriptions_helper(...);
    template <typename T>
    yes_tag has_fillDescriptions_helper(fillDescriptions_function<T, &T::fillDescriptions>* dummy);

    template <typename T>
    struct has_fillDescriptions_function {
      static constexpr bool value = std::is_same<decltype(has_fillDescriptions_helper<T>(nullptr)), yes_tag>::value;
    };

    template <typename T>
    struct DoFillDescriptions {
      void operator()(ConfigurationDescriptions& descriptions) { T::fillDescriptions(descriptions); }
    };

    template <typename T>
    struct DoFillAsUnknown {
      void operator()(ConfigurationDescriptions& descriptions) {
        ParameterSetDescription desc;
        desc.setUnknown();
        descriptions.addDefault(desc);
      }
    };

    template <typename T, void (*)(ConfigurationDescriptions&)>
    struct prevalidate_function;
    template <typename T>
    no_tag has_prevalidate_helper(...);
    template <typename T>
    yes_tag has_prevalidate_helper(fillDescriptions_function<T, &T::prevalidate>* dummy);

    template <typename T>
    struct has_prevalidate_function {
      static constexpr bool value = std::is_same<decltype(has_prevalidate_helper<T>(nullptr)), yes_tag>::value;
    };

    template <typename T>
    struct DoPrevalidate {
      void operator()(ConfigurationDescriptions& descriptions) { T::prevalidate(descriptions); }
    };

    template <typename T>
    struct DoNothing {
      void operator()(ConfigurationDescriptions& descriptions) {}
    };

  }  // namespace fillDetails

  // Not needed at the moment
  //void prevalidateService(ConfigurationDescriptions &);

  template <typename T>
  class DescriptionFillerForServices : public ParameterSetDescriptionFillerBase {
  public:
    DescriptionFillerForServices() {}

    // If T has a fillDescriptions function then just call that, otherwise
    // put in an "unknown description" as a default.
    void fill(ConfigurationDescriptions& descriptions) const override {
      std::conditional_t<edm::fillDetails::has_fillDescriptions_function<T>::value,
                         edm::fillDetails::DoFillDescriptions<T>,
                         edm::fillDetails::DoFillAsUnknown<T>>
          fill_descriptions;
      fill_descriptions(descriptions);
      //we don't have a need for prevalidation of services at the moment, so this is a placeholder
      // Probably the best package to declare this in would be FWCore/ServiceRegistry
      //prevalidateService(descriptions);
    }

    const std::string& baseType() const override { return kBaseForService; }

    const std::string& extendedBaseType() const override { return kEmpty; }

  private:
    void prevalidate(ConfigurationDescriptions& descriptions);
    DescriptionFillerForServices(const DescriptionFillerForServices&);                   // stop default
    const DescriptionFillerForServices& operator=(const DescriptionFillerForServices&);  // stop default
  };

  template <typename T>
  class DescriptionFillerForESSources : public ParameterSetDescriptionFillerBase {
  public:
    DescriptionFillerForESSources() {}
    DescriptionFillerForESSources(const DescriptionFillerForESSources&) = delete;                   // stop default
    const DescriptionFillerForESSources& operator=(const DescriptionFillerForESSources&) = delete;  // stop default

    // If T has a fillDescriptions function then just call that, otherwise
    // put in an "unknown description" as a default.
    void fill(ConfigurationDescriptions& descriptions) const override {
      std::conditional_t<edm::fillDetails::has_fillDescriptions_function<T>::value,
                         edm::fillDetails::DoFillDescriptions<T>,
                         edm::fillDetails::DoFillAsUnknown<T>>
          fill_descriptions;
      fill_descriptions(descriptions);

      std::conditional_t<edm::fillDetails::has_prevalidate_function<T>::value,
                         edm::fillDetails::DoPrevalidate<T>,
                         edm::fillDetails::DoNothing<T>>
          prevalidate;
      prevalidate(descriptions);
    }

    const std::string& baseType() const override { return kBaseForESSource; }

    const std::string& extendedBaseType() const override { return kEmpty; }
  };

  template <typename T>
  class DescriptionFillerForESProducers : public ParameterSetDescriptionFillerBase {
  public:
    DescriptionFillerForESProducers() {}
    DescriptionFillerForESProducers(const DescriptionFillerForESProducers&) = delete;                   // stop default
    const DescriptionFillerForESProducers& operator=(const DescriptionFillerForESProducers&) = delete;  // stop default

    // If T has a fillDescriptions function then just call that, otherwise
    // put in an "unknown description" as a default.
    void fill(ConfigurationDescriptions& descriptions) const override {
      std::conditional_t<edm::fillDetails::has_fillDescriptions_function<T>::value,
                         edm::fillDetails::DoFillDescriptions<T>,
                         edm::fillDetails::DoFillAsUnknown<T>>
          fill_descriptions;
      fill_descriptions(descriptions);

      std::conditional_t<edm::fillDetails::has_prevalidate_function<T>::value,
                         edm::fillDetails::DoPrevalidate<T>,
                         edm::fillDetails::DoNothing<T>>
          prevalidate;
      prevalidate(descriptions);
    }

    const std::string& baseType() const override { return kBaseForESProducer; }

    const std::string& extendedBaseType() const override { return kEmpty; }
  };
}  // namespace edm
#endif
