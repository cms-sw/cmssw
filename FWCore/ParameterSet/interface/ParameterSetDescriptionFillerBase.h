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
   class EDProducer;
   class EDFilter;
   class EDAnalyzer;
   class OutputModule;

   namespace one {
      class EDProducerBase;
      class EDFilterBase;
      class EDAnalyzerBase;
      class OutputModuleBase;
   }

   namespace stream {
    class EDProducerBase;
    class EDFilterBase;
    class EDAnalyzerBase;
   }

   namespace global {
    class EDProducerBase;
    class EDFilterBase;
    class EDAnalyzerBase;
    class OutputModuleBase;
   }

   namespace limited {
      class EDProducerBase;
      class EDFilterBase;
      class EDAnalyzerBase;
      class OutputModuleBase;
   }

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
      static const std::string kExtendedBaseForGlobalOutputModule;
      static const std::string kExtendedBaseForLimitedEDAnalyzer;
      static const std::string kExtendedBaseForLimitedEDProducer;
      static const std::string kExtendedBaseForLimitedEDFilter;
      static const std::string kExtendedBaseForLimitedOutputModule;

      static const std::string& extendedBaseType(EDAnalyzer const*) {
         return kExtendedBaseForEDAnalyzer;
      }
      static const std::string& extendedBaseType(EDProducer const*)  {
       return kExtendedBaseForEDProducer;
      }
      static const std::string& extendedBaseType(EDFilter const*)  {
         return kExtendedBaseForEDFilter;
      }
      static const std::string& extendedBaseType(OutputModule const*)  {
         return kExtendedBaseForOutputModule;
      }
      static const std::string& extendedBaseType(one::EDAnalyzerBase const*)  {
         return kExtendedBaseForOneEDAnalyzer;
      }
      static const std::string& extendedBaseType(one::EDProducerBase const*)  {
         return kExtendedBaseForOneEDProducer;
      }
      static const std::string& extendedBaseType(one::EDFilterBase const*)  {
         return kExtendedBaseForOneEDFilter;
      }
      static const std::string& extendedBaseType(one::OutputModuleBase const*)  {
         return kExtendedBaseForOneOutputModule;
      }
      static const std::string& extendedBaseType(stream::EDAnalyzerBase const*) {
         return kExtendedBaseForStreamEDAnalyzer;
      }
      static const std::string& extendedBaseType(stream::EDProducerBase const*) {
         return kExtendedBaseForStreamEDProducer;
      }
      static const std::string& extendedBaseType(stream::EDFilterBase const*) {
         return kExtendedBaseForStreamEDFilter;
      }
      static const std::string& extendedBaseType(global::EDAnalyzerBase const*) {
         return kExtendedBaseForGlobalEDAnalyzer;
      }
      static const std::string& extendedBaseType(global::EDProducerBase const*) {
         return kExtendedBaseForGlobalEDProducer;
      }
      static const std::string& extendedBaseType(global::EDFilterBase const*) {
         return kExtendedBaseForGlobalEDFilter;
      }
      static const std::string& extendedBaseType(global::OutputModuleBase const*) {
         return kExtendedBaseForGlobalOutputModule;
      }
   static const std::string& extendedBaseType(limited::EDAnalyzerBase const*) {
      return kExtendedBaseForLimitedEDAnalyzer;
   }
   static const std::string& extendedBaseType(limited::EDProducerBase const*) {
      return kExtendedBaseForLimitedEDProducer;
   }
   static const std::string& extendedBaseType(limited::EDFilterBase const*) {
      return kExtendedBaseForLimitedEDFilter;
   }
   static const std::string& extendedBaseType(limited::OutputModuleBase const*) {
      return kExtendedBaseForLimitedOutputModule;
   }
      static const std::string& extendedBaseType(void const *) {
         return kEmpty;
      }
  
   private:
      ParameterSetDescriptionFillerBase(const ParameterSetDescriptionFillerBase&) = delete; // stop default

      const ParameterSetDescriptionFillerBase& operator=(const ParameterSetDescriptionFillerBase&) = delete; // stop default

      // ---------- member data --------------------------------

};

}
#endif
