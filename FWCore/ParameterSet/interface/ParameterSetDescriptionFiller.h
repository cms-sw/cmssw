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
#include <boost/mpl/if.hpp>
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/global/OutputModule.h"

namespace edm {
  template< typename T>
  class ParameterSetDescriptionFiller : public ParameterSetDescriptionFillerBase
  {
  public:
    ParameterSetDescriptionFiller() {}

    virtual void fill(ConfigurationDescriptions & descriptions) const {
      T::fillDescriptions(descriptions);
      T::prevalidate(descriptions);
    }

    virtual const std::string& baseType() const {
      return T::baseType();
    }

    virtual const std::string& extendedBaseType() const {
      if (std::is_base_of<edm::EDAnalyzer, T>::value)
        return kExtendedBaseForEDAnalyzer;
      if (std::is_base_of<edm::EDProducer, T>::value)
        return kExtendedBaseForEDProducer;
      if (std::is_base_of<edm::EDFilter, T>::value)
        return kExtendedBaseForEDFilter;
      if (std::is_base_of<edm::OutputModule, T>::value)
        return kExtendedBaseForOutputModule;
      if (std::is_base_of<edm::one::EDAnalyzerBase, T>::value)
        return kExtendedBaseForOneEDAnalyzer;
      if (std::is_base_of<edm::one::EDProducerBase, T>::value)
        return kExtendedBaseForOneEDProducer;
      if (std::is_base_of<edm::one::EDFilterBase, T>::value)
        return kExtendedBaseForOneEDFilter;
      if (std::is_base_of<edm::one::OutputModuleBase, T>::value)
        return kExtendedBaseForOneOutputModule;
      if (std::is_base_of<edm::stream::EDAnalyzerBase, T>::value)
        return kExtendedBaseForStreamEDAnalyzer;
      if (std::is_base_of<edm::stream::EDProducerBase, T>::value)
        return kExtendedBaseForStreamEDProducer;
      if (std::is_base_of<edm::stream::EDFilterBase, T>::value)
        return kExtendedBaseForStreamEDFilter;
      if (std::is_base_of<edm::global::EDAnalyzerBase, T>::value)
        return kExtendedBaseForGlobalEDAnalyzer;
      if (std::is_base_of<edm::global::EDProducerBase, T>::value)
        return kExtendedBaseForGlobalEDProducer;
      if (std::is_base_of<edm::global::EDFilterBase, T>::value)
        return kExtendedBaseForGlobalEDFilter;
      if (std::is_base_of<edm::global::OutputModuleBase, T>::value)
        return kExtendedBaseForGlobalOutputModule;

      return kEmpty;
    }

  private:
    ParameterSetDescriptionFiller(const ParameterSetDescriptionFiller&); // stop default
    const ParameterSetDescriptionFiller& operator=(const ParameterSetDescriptionFiller&); // stop default
  };

  // We need a special version of this class for Services because there is
  // no common base class for all Service classes.  This means we cannot define
  // the baseType and fillDescriptions functions for all Service classes without
  // great difficulty.

  // First, some template metaprogramming to determining if the class T has
  // a fillDescriptions function.

  namespace fillDetails {

    typedef char (& no_tag)[1]; // type indicating FALSE
    typedef char (& yes_tag)[2]; // type indicating TRUE

    template <typename T, void (*)(ConfigurationDescriptions &)>  struct fillDescriptions_function;
    template <typename T> no_tag  has_fillDescriptions_helper(...);
    template <typename T> yes_tag has_fillDescriptions_helper(fillDescriptions_function<T, &T::fillDescriptions> * dummy);

    template<typename T>
    struct has_fillDescriptions_function {
      static bool const value =
        sizeof(has_fillDescriptions_helper<T>(0)) == sizeof(yes_tag);
    };

    template <typename T>
    struct DoFillDescriptions {
      void operator()(ConfigurationDescriptions & descriptions) {
        T::fillDescriptions(descriptions);
      }
    };

    template <typename T>
    struct DoFillAsUnknown {
      void operator()(ConfigurationDescriptions & descriptions) {
        ParameterSetDescription desc;
        desc.setUnknown();
        descriptions.addDefault(desc);
      }
    };
    
    template <typename T, void (*)(ConfigurationDescriptions &)>  struct prevalidate_function;
    template <typename T> no_tag  has_prevalidate_helper(...);
    template <typename T> yes_tag has_prevalidate_helper(fillDescriptions_function<T, &T::prevalidate> * dummy);
    
    template<typename T>
    struct has_prevalidate_function {
      static bool const value =
      sizeof(has_prevalidate_helper<T>(0)) == sizeof(yes_tag);
    };
    
    template <typename T>
    struct DoPrevalidate {
      void operator()(ConfigurationDescriptions & descriptions) {
        T::prevalidate(descriptions);
      }
    };
    
    template <typename T>
    struct DoNothing {
      void operator()(ConfigurationDescriptions & descriptions) {
      }
    };

  }

  // Not needed at the moment
  //void prevalidateService(ConfigurationDescriptions &);
  
  template< typename T>
  class DescriptionFillerForServices : public ParameterSetDescriptionFillerBase
  {
  public:
    DescriptionFillerForServices() {}

    // If T has a fillDescriptions function then just call that, otherwise
    // put in an "unknown description" as a default.
    virtual void fill(ConfigurationDescriptions & descriptions) const {
      typename boost::mpl::if_c<edm::fillDetails::has_fillDescriptions_function<T>::value,
                                edm::fillDetails::DoFillDescriptions<T>,
                                edm::fillDetails::DoFillAsUnknown<T> >::type fill_descriptions;
      fill_descriptions(descriptions);
      //we don't have a need for prevalidation of services at the moment, so this is a placeholder
      // Probably the best package to declare this in would be FWCore/ServiceRegistry
      //prevalidateService(descriptions);
    }

    virtual const std::string& baseType() const {
      return kBaseForService;
    }

    virtual const std::string& extendedBaseType() const {
      return kEmpty;
    }

  private:
    void prevalidate(ConfigurationDescriptions & descriptions);
    DescriptionFillerForServices(const DescriptionFillerForServices&); // stop default
    const DescriptionFillerForServices& operator=(const DescriptionFillerForServices&); // stop default
  };

  template<typename T>
  class DescriptionFillerForESSources : public ParameterSetDescriptionFillerBase
  {
  public:
    DescriptionFillerForESSources() {}

    // If T has a fillDescriptions function then just call that, otherwise
    // put in an "unknown description" as a default.
    virtual void fill(ConfigurationDescriptions & descriptions) const {
      typename boost::mpl::if_c<edm::fillDetails::has_fillDescriptions_function<T>::value,
                                edm::fillDetails::DoFillDescriptions<T>,
                                edm::fillDetails::DoFillAsUnknown<T> >::type fill_descriptions;
      fill_descriptions(descriptions);
      
      typename boost::mpl::if_c<edm::fillDetails::has_prevalidate_function<T>::value,
      edm::fillDetails::DoPrevalidate<T>,
      edm::fillDetails::DoNothing<T> >::type prevalidate;
      prevalidate(descriptions);
    }

    virtual const std::string& baseType() const {
      return kBaseForESSource;
    }

    virtual const std::string& extendedBaseType() const {
      return kEmpty;
    }

  private:
    DescriptionFillerForESSources(const DescriptionFillerForESSources&); // stop default
    const DescriptionFillerForESSources& operator=(const DescriptionFillerForESSources&); // stop default
  };

  template<typename T>
  class DescriptionFillerForESProducers : public ParameterSetDescriptionFillerBase
  {
  public:
    DescriptionFillerForESProducers() {}

    // If T has a fillDescriptions function then just call that, otherwise
    // put in an "unknown description" as a default.
    virtual void fill(ConfigurationDescriptions & descriptions) const {
      typename boost::mpl::if_c<edm::fillDetails::has_fillDescriptions_function<T>::value,
                                edm::fillDetails::DoFillDescriptions<T>,
                                edm::fillDetails::DoFillAsUnknown<T> >::type fill_descriptions;
      fill_descriptions(descriptions);
      
      typename boost::mpl::if_c<edm::fillDetails::has_prevalidate_function<T>::value,
      edm::fillDetails::DoPrevalidate<T>,
      edm::fillDetails::DoNothing<T> >::type prevalidate;
      prevalidate(descriptions);
    }

    virtual const std::string& baseType() const {
      return kBaseForESProducer;
    }

    virtual const std::string& extendedBaseType() const {
      return kEmpty;
    }

  private:
    DescriptionFillerForESProducers(const DescriptionFillerForESProducers&); // stop default
    const DescriptionFillerForESProducers& operator=(const DescriptionFillerForESProducers&); // stop default
  };
}
#endif
