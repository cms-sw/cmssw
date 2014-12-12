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
#include "FWCore/Utilities/interface/ProductHolderIndex.h"
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
  class ProductHolderIndexHelper;
  class EDConsumerBase;
  class PreallocationConfiguration;
  class ProductHolderIndexAndSkipBit;
  class ActivityRegistry;
  class ProductRegistry;
  class ThinnedAssociationsHelper;

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
      const ModuleDescription& moduleDescription() { return moduleDescription_;}
      
      std::string workerType() const { return "WorkerT<EDAnalyzerAdaptorBase>";}
      void
      registerProductsAndCallbacks(EDAnalyzerAdaptorBase const*, ProductRegistry* reg);
    protected:
      template<typename T> void createStreamModules(T iFunc) {
        for(auto& m: m_streamModules) {
          m = iFunc();
        }
      }
      
      //Same interface as EDConsumerBase
      void itemsToGet(BranchType, std::vector<ProductHolderIndexAndSkipBit>&) const;
      void itemsMayGet(BranchType, std::vector<ProductHolderIndexAndSkipBit>&) const;
      std::vector<ProductHolderIndexAndSkipBit> const& itemsToGetFromEvent() const;

      void updateLookup(BranchType iBranchType,
                        ProductHolderIndexHelper const&);
      
      const EDConsumerBase* consumer() const;
      
      void modulesDependentUpon(const std::string& iProcessName,
                                std::vector<const char*>& oModuleLabels) const;

      void modulesWhoseProductsAreConsumed(std::vector<ModuleDescription const*>& modules,
                                           ProductRegistry const& preg,
                                           std::map<std::string, ModuleDescription const*> const& labelsToDesc,
                                           std::string const& processName) const;

      std::vector<ConsumesInfo> consumesInfo() const;

    private:
      EDAnalyzerAdaptorBase(const EDAnalyzerAdaptorBase&); // stop default
      
      const EDAnalyzerAdaptorBase& operator=(const EDAnalyzerAdaptorBase&); // stop default
      
      bool doEvent(EventPrincipal& ep, EventSetup const& c,
                   ActivityRegistry*,
                   ModuleCallingContext const*) ;
      void doPreallocate(PreallocationConfiguration const&);
      
      virtual void setupStreamModules() = 0;
      void doBeginJob();
      virtual void doEndJob() = 0;
      
      void doBeginStream(StreamID id);
      void doEndStream(StreamID id);
      void doStreamBeginRun(StreamID id,
                            RunPrincipal& ep,
                            EventSetup const& c,
                            ModuleCallingContext const*);
      virtual void setupRun(EDAnalyzerBase*, RunIndex) = 0;
      void doStreamEndRun(StreamID id,
                          RunPrincipal& ep,
                          EventSetup const& c,
                          ModuleCallingContext const*);
      virtual void streamEndRunSummary(EDAnalyzerBase*,edm::Run const&, edm::EventSetup const&) = 0;

      void doStreamBeginLuminosityBlock(StreamID id,
                                        LuminosityBlockPrincipal& ep,
                                        EventSetup const& c,
                                        ModuleCallingContext const*);
      virtual void setupLuminosityBlock(EDAnalyzerBase*, LuminosityBlockIndex) = 0;
      void doStreamEndLuminosityBlock(StreamID id,
                                      LuminosityBlockPrincipal& ep,
                                      EventSetup const& c,
                                      ModuleCallingContext const*);
      virtual void streamEndLuminosityBlockSummary(EDAnalyzerBase*,edm::LuminosityBlock const&, edm::EventSetup const&) = 0;

      virtual void doBeginRun(RunPrincipal& rp, EventSetup const& c,
                              ModuleCallingContext const*)=0;
      virtual void doEndRun(RunPrincipal& rp, EventSetup const& c,
                            ModuleCallingContext const*)=0;
      virtual void doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                          ModuleCallingContext const*)=0;
      virtual void doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                        ModuleCallingContext const*)=0;

      void doPreForkReleaseResources();
      void doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);

      //For now, the following are just dummy implemenations with no ability for users to override
      void doRespondToOpenInputFile(FileBlock const& fb);
      void doRespondToCloseInputFile(FileBlock const& fb);
      void doRegisterThinnedAssociations(ProductRegistry const&,
                                         ThinnedAssociationsHelper&) { }

      // ---------- member data --------------------------------
      void setModuleDescription(ModuleDescription const& md) {
        moduleDescription_ = md;
      }
      ModuleDescription moduleDescription_;

      std::vector<EDAnalyzerBase*> m_streamModules;
    };
  }
}
#endif
