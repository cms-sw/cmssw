#ifndef FWCore_Framework_stream_EDAnalyzerBase_h
#define FWCore_Framework_stream_EDAnalyzerBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDAnalyzerBase
// 
/**\class edm::stream::EDAnalyzerBase EDAnalyzerBase.h "FWCore/Framework/interface/stream/EDAnalyzerBase.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 00:11:27 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerAdaptor.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

// forward declarations
namespace edm {
  namespace stream {
    class EDAnalyzerAdaptorBase;
    
    class EDAnalyzerBase : public edm::EDConsumerBase
    {
      //This needs access to the parentage cache info
      friend class EDAnalyzerAdaptorBase;

    public:
      typedef EDAnalyzerAdaptorBase ModuleType;


      EDAnalyzerBase();
      virtual ~EDAnalyzerBase();
      
      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      static void prevalidate(ConfigurationDescriptions& descriptions);
      static const std::string& baseType();
      
      // Warning: the returned moduleDescription will be invalid during construction
      ModuleDescription const& moduleDescription() const {
        return *moduleDescriptionPtr_;
      }
    protected:

      void callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func);

    private:
      EDAnalyzerBase(const EDAnalyzerBase&) = delete; // stop default
      
      const EDAnalyzerBase& operator=(const EDAnalyzerBase&) = delete; // stop default
      
      void registerProductsAndCallbacks(EDAnalyzerBase const*, ProductRegistry* reg);
      
      virtual void beginStream(StreamID) {}
      virtual void beginRun(edm::Run const&, edm::EventSetup const&) {}
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
      virtual void analyze(Event const&, EventSetup const&) = 0;
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
      virtual void endRun(edm::Run const&, edm::EventSetup const&) {}
      virtual void endStream(){}

      void setModuleDescriptionPtr(ModuleDescription const* iDesc) {
        moduleDescriptionPtr_ = iDesc;
      }
      // ---------- member data --------------------------------
      std::function<void(BranchDescription const&)> callWhenNewProductsRegistered_;
      ModuleDescription const* moduleDescriptionPtr_;
    };
    
  }
}



#endif
