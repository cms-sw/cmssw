/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"

#include "FWCore/Framework/src/OutputModuleCommunicatorT.h"

namespace edm {

  template<typename T>
  void
  OutputModuleCommunicatorT<T>::closeFile() {
    module().doCloseFile();
  }

  template<typename T>
  bool
  OutputModuleCommunicatorT<T>::shouldWeCloseFile() const {
    return module().shouldWeCloseFile();
  }

  template<typename T>
  void
  OutputModuleCommunicatorT<T>::openFile(edm::FileBlock const& fb) {
    module().doOpenFile(fb);
  }

  template<typename T>
  void
  OutputModuleCommunicatorT<T>::writeRun(edm::RunPrincipal const& rp, ProcessContext const* processContext) {
    GlobalContext globalContext(GlobalContext::Transition::kWriteRun,
                                LuminosityBlockID(rp.run(), 0),
                                rp.index(),
                                LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                                rp.endTime(),
                                processContext);
    ParentContext parentContext(&globalContext);
    ModuleCallingContext mcc(&description());
    ModuleContextSentry moduleContextSentry(&mcc, parentContext);
    module().doWriteRun(rp, &mcc);
  }

  template<typename T>
  void
  OutputModuleCommunicatorT<T>::writeLumi(edm::LuminosityBlockPrincipal const& lbp, ProcessContext const* processContext) {
    GlobalContext globalContext(GlobalContext::Transition::kWriteLuminosityBlock,
                                lbp.id(),
                                lbp.runPrincipal().index(),
                                lbp.index(),
                                lbp.beginTime(),
                                processContext);
    ParentContext parentContext(&globalContext);
    ModuleCallingContext mcc(&description());
    ModuleContextSentry moduleContextSentry(&mcc, parentContext);
    module().doWriteLuminosityBlock(lbp, &mcc);
  }

  template<typename T>
  bool OutputModuleCommunicatorT<T>::wantAllEvents() const {return module().wantAllEvents();}

  template<typename T>
  bool OutputModuleCommunicatorT<T>::limitReached() const {return module().limitReached();}

  template<typename T>
  void OutputModuleCommunicatorT<T>::configure(OutputModuleDescription const& desc) {module().configure(desc);}

  template<typename T>
  edm::SelectedProductsForBranchType const& OutputModuleCommunicatorT<T>::keptProducts() const {
    return module().keptProducts();
  }

  template<typename T>
  void OutputModuleCommunicatorT<T>::selectProducts(edm::ProductRegistry const& preg, ThinnedAssociationsHelper const& helper) {
    module().selectProducts(preg, helper);
  }

  template<typename T>
  void OutputModuleCommunicatorT<T>::setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                                                    bool anyProductProduced) {
    module().setEventSelectionInfo(outputModulePathPositions, anyProductProduced);
  }

  template<typename T>
  ModuleDescription const& OutputModuleCommunicatorT<T>::description() const {
    return module().description();
  }

  namespace impl {
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(void *) {
      return std::unique_ptr<edm::OutputModuleCommunicator>{};
    }
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::OutputModule * iMod){
      return std::make_unique<OutputModuleCommunicatorT<edm::OutputModule>>(iMod);
    }
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::global::OutputModuleBase * iMod){
      return std::make_unique<OutputModuleCommunicatorT<edm::global::OutputModuleBase>>(iMod);
    }
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::one::OutputModuleBase * iMod){
      return std::make_unique<OutputModuleCommunicatorT<edm::one::OutputModuleBase>>(iMod);
    }
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::limited::OutputModuleBase * iMod){
      return std::make_unique<OutputModuleCommunicatorT<edm::limited::OutputModuleBase>>(iMod);
    }
  }
}

#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/global/OutputModuleBase.h"
#include "FWCore/Framework/interface/one/OutputModuleBase.h"
#include "FWCore/Framework/interface/limited/OutputModuleBase.h"

namespace edm {
  template class OutputModuleCommunicatorT<OutputModule>;
  template class OutputModuleCommunicatorT<one::OutputModuleBase>;
  template class OutputModuleCommunicatorT<global::OutputModuleBase>;
  template class OutputModuleCommunicatorT<limited::OutputModuleBase>;
}
