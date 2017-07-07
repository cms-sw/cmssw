#ifndef FWCore_Framework_stream_ProducingModuleAdaptorBase_h
#define FWCore_Framework_stream_ProducingModuleAdaptorBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ProducingModuleAdaptorBase
// 
/**\class edm::stream::ProducingModuleAdaptorBase ProducingModuleAdaptorBase.h "FWCore/Framework/interface/stream/ProducingModuleAdaptorBase.h"

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
#include <unordered_map>

// user include files
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"

// forward declarations

namespace edm {
  class Event;
  class ModuleCallingContext;
  class ProductResolverIndexHelper;
  class EDConsumerBase;
  class PreallocationConfiguration;
  class ProductResolverIndexAndSkipBit;
  class ProductRegistry;
  class ThinnedAssociationsHelper;

  namespace maker {
    template<typename T> class ModuleHolderT;
  }
  
  namespace stream {
    template<typename T>
    class ProducingModuleAdaptorBase
    {
      
    public:
      template <typename U> friend class edm::WorkerT;
      template <typename U> friend class edm::maker::ModuleHolderT;

      ProducingModuleAdaptorBase();
      virtual ~ProducingModuleAdaptorBase();
      
      // ---------- const member functions ---------------------
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      const ModuleDescription& moduleDescription() const { return moduleDescription_;}
      
      void
      registerProductsAndCallbacks(ProducingModuleAdaptorBase const*, ProductRegistry* reg);
      
      void itemsToGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const;
      void itemsMayGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const;
      std::vector<ProductResolverIndexAndSkipBit> const& itemsToGetFrom(BranchType) const;

      void updateLookup(BranchType iBranchType,
                        ProductResolverIndexHelper const&,
                        bool iPrefetchMayGet);

      void modulesWhoseProductsAreConsumed(std::vector<ModuleDescription const*>& modules,
                                           ProductRegistry const& preg,
                                           std::map<std::string, ModuleDescription const*> const& labelsToDesc,
                                           std::string const& processName) const;

      void convertCurrentProcessAlias(std::string const& processName);

      std::vector<ConsumesInfo> consumesInfo() const;

      void resolvePutIndicies(BranchType iBranchType,
                              std::unordered_multimap<std::string, edm::ProductResolverIndex> const& iIndicies,
                              std::string const& moduleLabel);
      
      std::vector<edm::ProductResolverIndex> const& indiciesForPutProducts(BranchType iBranchType) const;

    protected:
      template<typename F> void createStreamModules(F iFunc) {
        for(auto& m: m_streamModules) {
          m = iFunc();
          m->setModuleDescriptionPtr(&moduleDescription_);
        }
      }
      
      void commit(Run& iRun) {
        iRun.commit_(m_streamModules[0]->indiciesForPutProducts(InRun));
      }
      void commit(LuminosityBlock& iLumi) {
        iLumi.commit_(m_streamModules[0]->indiciesForPutProducts(InLumi));
      }
      template<typename L, typename I>
      void commit(Event& iEvent, L* iList, I* iID) {
        iEvent.commit_(m_streamModules[0]->indiciesForPutProducts(InEvent), iList,iID);
      }

      const EDConsumerBase* consumer() {
        return m_streamModules[0];
      }
    private:
      ProducingModuleAdaptorBase(const ProducingModuleAdaptorBase&) = delete; // stop default
      
      const ProducingModuleAdaptorBase& operator=(const ProducingModuleAdaptorBase&) = delete; // stop default

      void doPreallocate(PreallocationConfiguration const&);
      virtual void setupStreamModules() = 0;
      void doBeginJob();
      virtual void doEndJob() = 0;
      
      void doBeginStream(StreamID id);
      void doEndStream(StreamID id);
      void doStreamBeginRun(StreamID id,
                            RunPrincipal const& ep,
                            EventSetup const& c,
                            ModuleCallingContext const*);
      virtual void setupRun(T*, RunIndex) = 0;
      void doStreamEndRun(StreamID id,
                          RunPrincipal const& ep,
                          EventSetup const& c,
                          ModuleCallingContext const*);
      virtual void streamEndRunSummary(T*,edm::Run const&, edm::EventSetup const&) = 0;

      void doStreamBeginLuminosityBlock(StreamID id,
                                        LuminosityBlockPrincipal const& ep,
                                        EventSetup const& c,
                                        ModuleCallingContext const*);
      virtual void setupLuminosityBlock(T*, LuminosityBlockIndex) = 0;
      void doStreamEndLuminosityBlock(StreamID id,
                                      LuminosityBlockPrincipal const& ep,
                                      EventSetup const& c,
                                      ModuleCallingContext const*);
      virtual void streamEndLuminosityBlockSummary(T*,edm::LuminosityBlock const&, edm::EventSetup const&) = 0;
      
      
      virtual void doBeginRun(RunPrincipal const& rp, EventSetup const& c,
                              ModuleCallingContext const*)=0;
      virtual void doEndRun(RunPrincipal const& rp, EventSetup const& c,
                            ModuleCallingContext const*)=0;
      virtual void doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                          EventSetup const& c,
                                          ModuleCallingContext const*)=0;
      virtual void doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                        EventSetup const& c,
                                        ModuleCallingContext const*)=0;
      
      //For now, the following are just dummy implemenations with no ability for users to override
      void doRespondToOpenInputFile(FileBlock const& fb);
      void doRespondToCloseInputFile(FileBlock const& fb);
      void doRegisterThinnedAssociations(ProductRegistry const&,
                                         ThinnedAssociationsHelper&);

      // ---------- member data --------------------------------
      void setModuleDescription(ModuleDescription const& md) {
        moduleDescription_ = md;
      }
      ModuleDescription moduleDescription_;
    protected:
      std::vector<T*> m_streamModules;

    };
  }
}

#endif
