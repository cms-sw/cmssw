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

// user include files
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductHolderIndex.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

// forward declarations

namespace edm {
  class Event;
  class ModuleCallingContext;
  class ProductHolderIndexHelper;
  class EDConsumerBase;
  class PreallocationConfiguration;
  class ProductHolderIndexAndSkipBit;
  
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
      const ModuleDescription& moduleDescription() { return moduleDescription_;}
      
      void
      registerProductsAndCallbacks(ProducingModuleAdaptorBase const*, ProductRegistry* reg);
      
      void itemsToGet(BranchType, std::vector<ProductHolderIndexAndSkipBit>&) const;
      void itemsMayGet(BranchType, std::vector<ProductHolderIndexAndSkipBit>&) const;
      std::vector<ProductHolderIndexAndSkipBit> const& itemsToGetFromEvent() const;

      void updateLookup(BranchType iBranchType,
                        ProductHolderIndexHelper const&);

      void modulesDependentUpon(const std::string& iProcessName,
                                std::vector<const char*>& oModuleLabels) const;


    protected:
      template<typename F> void createStreamModules(F iFunc) {
        for(auto& m: m_streamModules) {
          m = iFunc();
          m->setModuleDescriptionPtr(&moduleDescription_);
        }
      }
      
      void commit(Run& iRun) {
        iRun.commit_();
      }
      void commit(LuminosityBlock& iLumi) {
        iLumi.commit_();
      }
      template<typename L, typename I>
      void commit(Event& iEvent, L* iList, I* iID) {
        iEvent.commit_(iList,iID);
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
                            RunPrincipal& ep,
                            EventSetup const& c,
                            ModuleCallingContext const*);
      virtual void setupRun(T*, RunIndex) = 0;
      void doStreamEndRun(StreamID id,
                          RunPrincipal& ep,
                          EventSetup const& c,
                          ModuleCallingContext const*);
      virtual void streamEndRunSummary(T*,edm::Run const&, edm::EventSetup const&) = 0;

      void doStreamBeginLuminosityBlock(StreamID id,
                                        LuminosityBlockPrincipal& ep,
                                        EventSetup const& c,
                                        ModuleCallingContext const*);
      virtual void setupLuminosityBlock(T*, LuminosityBlockIndex) = 0;
      void doStreamEndLuminosityBlock(StreamID id,
                                      LuminosityBlockPrincipal& ep,
                                      EventSetup const& c,
                                      ModuleCallingContext const*);
      virtual void streamEndLuminosityBlockSummary(T*,edm::LuminosityBlock const&, edm::EventSetup const&) = 0;
      
      
      virtual void doBeginRun(RunPrincipal& rp, EventSetup const& c,
                              ModuleCallingContext const*)=0;
      virtual void doEndRun(RunPrincipal& rp, EventSetup const& c,
                            ModuleCallingContext const*)=0;
      virtual void doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp,
                                          EventSetup const& c,
                                          ModuleCallingContext const*)=0;
      virtual void doEndLuminosityBlock(LuminosityBlockPrincipal& lbp,
                                        EventSetup const& c,
                                        ModuleCallingContext const*)=0;
      
      //For now, the following are just dummy implemenations with no ability for users to override
      void doRespondToOpenInputFile(FileBlock const& fb);
      void doRespondToCloseInputFile(FileBlock const& fb);
      void doPreForkReleaseResources();
      void doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);

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
