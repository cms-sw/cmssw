#ifndef FWCore_Framework_stream_EDFilterBase_h
#define FWCore_Framework_stream_EDFilterBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDFilterBase
//
/**\class edm::stream::EDFilterBase EDFilterBase.h "FWCore/Framework/interface/stream/EDFilterBase.h"

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
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilterAdaptor.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

// forward declarations
namespace edm {

  class ProductRegistry;
  class ThinnedAssociationsHelper;
  class WaitingTaskWithArenaHolder;

  namespace stream {
    class EDFilterAdaptorBase;
    template <typename>
    class ProducingModuleAdaptorBase;

    class EDFilterBase : public edm::ProducerBase, public edm::EDConsumerBase {
      //This needs access to the parentage cache info
      friend class EDFilterAdaptorBase;
      friend class ProducingModuleAdaptorBase<EDFilterBase>;

    public:
      typedef EDFilterAdaptorBase ModuleType;

      EDFilterBase();
      EDFilterBase(const EDFilterBase&) = delete;                   // stop default
      const EDFilterBase& operator=(const EDFilterBase&) = delete;  // stop default
      ~EDFilterBase() override;

      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      static void prevalidate(ConfigurationDescriptions& descriptions);
      static const std::string& baseType();

      // Warning: the returned moduleDescription will be invalid during construction
      ModuleDescription const& moduleDescription() const { return *moduleDescriptionPtr_; }

    private:
      virtual void beginStream(StreamID) {}
      virtual void beginRun(edm::Run const&, edm::EventSetup const&) {}
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
      virtual bool filter(Event&, EventSetup const&) = 0;
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
      virtual void endRun(edm::Run const&, edm::EventSetup const&) {}
      virtual void endStream() {}

      virtual void registerThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) {}

      virtual void doAcquire_(Event const&, EventSetup const&, WaitingTaskWithArenaHolder&) = 0;

      void setModuleDescriptionPtr(ModuleDescription const* iDesc) { moduleDescriptionPtr_ = iDesc; }
      // ---------- member data --------------------------------

      std::vector<BranchID> previousParentage_;
      std::vector<BranchID> gotBranchIDsFromAcquire_;
      ParentageID previousParentageId_;
      ModuleDescription const* moduleDescriptionPtr_;
    };
  }  // namespace stream
}  // namespace edm
#endif
