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

#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "boost/mpl/if.hpp"
#include <string>

namespace edm {
  template< typename T>
  class ParameterSetDescriptionFiller : public ParameterSetDescriptionFillerBase
  {
  public:
    ParameterSetDescriptionFiller() {}

    virtual void fill(ConfigurationDescriptions & descriptions) const {
      T::fillDescriptions(descriptions);
    }

    virtual std::string baseType() const {
      return T::baseType();
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
  }

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
    }

    virtual std::string baseType() const {
      return std::string("Service");
    }

  private:
    DescriptionFillerForServices(const DescriptionFillerForServices&); // stop default
    const DescriptionFillerForServices& operator=(const DescriptionFillerForServices&); // stop default
  };
}
#endif
