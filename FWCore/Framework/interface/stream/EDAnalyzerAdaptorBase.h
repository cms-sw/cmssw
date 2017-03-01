#ifndef FWCore_Framework_stream_EDAnalyzerAdaptorBase_h
#define FWCore_Framework_stream_EDAnalyzerAdaptorBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDAnalyzerAdaptorBase
// 
/**\class edm::stream::EDAnalyzerAdaptorBase EDAnalyzerAdaptorBase.h "FWCore/Framework/interface/stream/EDAnalyzerAdaptorBase.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 18:09:15 GMT
//

// system include files
#include <map>
#include <string>
#include <vector>

// user include files
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"


// forward declarations

namespace edm {
  class ModuleCallingContext;
  class ProductResolverIndexHelper;
  class EDConsumerBase;
  class PreallocationConfiguration;
  class ProductResolverIndexAndSkipBit;
  class ActivityRegistry;
  class ProductRegistry;
  class ThinnedAssociationsHelper;
  class WaitingTask;

  namespace maker {
    template<typename T> class ModuleHolderT;
  }
  
  namespace stream {
    class EDAnalyzerBase;

    class EDAnalyzerAdaptorBase
    {
      
    public:
      template <typename T> friend class edm::WorkerT;
      template <typename T> friend class edm::maker::ModuleHolderT;

      EDAnalyzerAdaptorBase();
      virtual ~EDAnalyzerAdaptorBase();
      
      // ---------- const member functions ---------------------
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      const ModuleDescription& moduleDescription() const { return moduleDescription_;}
      
      std::string workerType() const { return "WorkerT<EDAnalyzerAdaptorBase>";}
      void
      registerProductsAndCallbacks(EDAnalyzerAdaptorBase const*, ProductRegistry* reg);
    protected:
      template<typename T> void createStreamModules(T iFunc) {
        for(auto& m: m_streamModules) {
          m = iFunc();
          setModuleDescriptionPtr(m);
        }
      }
      
      //Same interface as EDConsumerBase
      void itemsToGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const;
      void itemsMayGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const;
      std::vector<ProductResolverIndexAndSkipBit> const& itemsToGetFromEvent() const;

      void updateLookup(BranchType iBranchType,
                        ProductResolverIndexHelper const&,
                        bool iPrefetchMayGet);
      
      const EDConsumerBase* consumer() const;
      
      void modulesWhoseProductsAreConsumed(std::vector<ModuleDescription const*>& modules,
                                           ProductRegistry const& preg,
                                           std::map<std::string, ModuleDescription const*> const& labelsToDesc,
                                           std::string const& processName) const;

      std::vector<ConsumesInfo> consumesInfo() const;

    private:
      EDAnalyzerAdaptorBase(const EDAnalyzerAdaptorBase&); // stop default
      
      const EDAnalyzerAdaptorBase& operator=(const EDAnalyzerAdaptorBase&); // stop default
      
      bool doEvent(EventPrincipal const& ep, EventSetup const& c,
                   ActivityRegistry*,
                   ModuleCallingContext const*) ;
      void doPreallocate(PreallocationConfiguration const&);
      
      //For now this is a placeholder
      /*virtual*/ void preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const {}
      
      virtual void setupStreamModules() = 0;
      void doBeginJob();
      virtual void doEndJob() = 0;
      
      void doBeginStream(StreamID id);
      void doEndStream(StreamID id);
      void doStreamBeginRun(StreamID id,
                            RunPrincipal const& ep,
                            EventSetup const& c,
                            ModuleCallingContext const*);
      virtual void setupRun(EDAnalyzerBase*, RunIndex) = 0;
      void doStreamEndRun(StreamID id,
                          RunPrincipal const& ep,
                          EventSetup const& c,
                          ModuleCallingContext const*);
      virtual void streamEndRunSummary(EDAnalyzerBase*,edm::Run const&, edm::EventSetup const&) = 0;

      void doStreamBeginLuminosityBlock(StreamID id,
                                        LuminosityBlockPrincipal const& ep,
                                        EventSetup const& c,
                                        ModuleCallingContext const*);
      virtual void setupLuminosityBlock(EDAnalyzerBase*, LuminosityBlockIndex) = 0;
      void doStreamEndLuminosityBlock(StreamID id,
                                      LuminosityBlockPrincipal const& ep,
                                      EventSetup const& c,
                                      ModuleCallingContext const*);
      virtual void streamEndLuminosityBlockSummary(EDAnalyzerBase*,edm::LuminosityBlock const&, edm::EventSetup const&) = 0;

      virtual void doBeginRun(RunPrincipal const& rp, EventSetup const& c,
                              ModuleCallingContext const*)=0;
      virtual void doEndRun(RunPrincipal const& rp, EventSetup const& c,
                            ModuleCallingContext const*)=0;
      virtual void doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                          ModuleCallingContext const*)=0;
      virtual void doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                        ModuleCallingContext const*)=0;

      void doPreForkReleaseResources();
      void doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);

      //For now, the following are just dummy implemenations with no ability for users to override
      void doRespondToOpenInputFile(FileBlock const& fb);
      void doRespondToCloseInputFile(FileBlock const& fb);
      void doRegisterThinnedAssociations(ProductRegistry const&,
                                         ThinnedAssociationsHelper&) { }

      // ---------- member data --------------------------------
      void setModuleDescriptionPtr(EDAnalyzerBase* m);
      void setModuleDescription(ModuleDescription const& md) {
        moduleDescription_ = md;
      }
      ModuleDescription moduleDescription_;

      std::vector<EDAnalyzerBase*> m_streamModules;
    };
  }
}
#endif
