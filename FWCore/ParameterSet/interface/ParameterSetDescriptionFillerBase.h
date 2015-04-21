#ifndef FWCore_ParameterSet_ParameterSetDescriptionFillerBase_h
#define FWCore_ParameterSet_ParameterSetDescriptionFillerBase_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterSetDescriptionFillerBase
// 
/**\class ParameterSetDescriptionFillerBase ParameterSetDescriptionFillerBase.h FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h

 Description: Base class for a component which can fill a ParameterSetDescription object

 Usage:
    This base class provides an abstract interface for filling a ParameterSetDescription object.  This allows one to used by the 
ParameterSetDescriptionFillerPluginFactory to load a component of any type (e.g. cmsRun Source, cmsRun EDProducer or even a tracking plugin)
and query the component for its allowed ParameterSetDescription.

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Aug  1 16:46:53 EDT 2007
//

// system include files

// user include files

// forward declarations
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <string>

namespace edm {
class ParameterSetDescriptionFillerBase
{

   public:
      ParameterSetDescriptionFillerBase() {}
      virtual ~ParameterSetDescriptionFillerBase();

      // ---------- const member functions ---------------------
      virtual void fill(ConfigurationDescriptions & descriptions) const = 0;
      virtual const std::string& baseType() const = 0;
      virtual const std::string& extendedBaseType() const = 0;
  
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

protected:
     static const std::string kEmpty;
     static const std::string kBaseForService;
     static const std::string kBaseForESSource;
     static const std::string kBaseForESProducer;
     static const std::string kExtendedBaseForEDAnalyzer;
     static const std::string kExtendedBaseForEDProducer;
     static const std::string kExtendedBaseForEDFilter;
     static const std::string kExtendedBaseForOutputModule;
     static const std::string kExtendedBaseForOneEDAnalyzer;
     static const std::string kExtendedBaseForOneEDProducer;
     static const std::string kExtendedBaseForOneEDFilter;
     static const std::string kExtendedBaseForOneOutputModule;
     static const std::string kExtendedBaseForStreamEDAnalyzer;
     static const std::string kExtendedBaseForStreamEDProducer;
     static const std::string kExtendedBaseForStreamEDFilter;
     static const std::string kExtendedBaseForGlobalEDAnalyzer;
     static const std::string kExtendedBaseForGlobalEDProducer;
     static const std::string kExtendedBaseForGlobalEDFilter;

   private:
      ParameterSetDescriptionFillerBase(const ParameterSetDescriptionFillerBase&); // stop default

      const ParameterSetDescriptionFillerBase& operator=(const ParameterSetDescriptionFillerBase&); // stop default

      // ---------- member data --------------------------------

};

}
#endif
