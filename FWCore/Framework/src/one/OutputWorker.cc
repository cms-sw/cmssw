
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/OutputModuleBase.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/src/one/OutputWorker.h"
#include "FWCore/Framework/src/OutputModuleCommunicator.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"

namespace edm {

  class ProcessContext;

  namespace one {
    OutputWorker::OutputWorker(OutputModuleBase* mod,
                               ModuleDescription const& md,
                               ExceptionToActionTable const* actions):
    WorkerT<OutputModuleBase>(mod, md, actions)
    {
    }
    
    OutputWorker::~OutputWorker() {
    }
    
    class OneOutputModuleCommunicator: public edm::OutputModuleCommunicator {
    public:
      OneOutputModuleCommunicator(edm::one::OutputModuleBase* iModule):
      module_(iModule){}
      virtual void closeFile() override;
      
      ///\return true if output module wishes to close its file
      virtual bool shouldWeCloseFile() const override;
      
      virtual void openNewFileIfNeeded() override;
      
      ///\return true if no event filtering is applied to OutputModule
      virtual bool wantAllEvents() const override;
      
      virtual void openFile(edm::FileBlock const& fb) override;
      
      virtual void writeRun(edm::RunPrincipal const& rp, ProcessContext const*) override;
      
      virtual void writeLumi(edm::LuminosityBlockPrincipal const& lbp, ProcessContext const*) override;
      
      ///\return true if OutputModule has reached its limit on maximum number of events it wants to see
      virtual bool limitReached() const override;
      
      virtual void configure(edm::OutputModuleDescription const& desc) override;
      
      virtual edm::SelectionsArray const& keptProducts() const override;
      
      virtual void selectProducts(edm::ProductRegistry const& preg) override;
      
      virtual void setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                                         bool anyProductProduced) override;
      
      virtual ModuleDescription const& description() const override;
      
      
    private:
      inline edm::one::OutputModuleBase& module() const { return *module_;}
      edm::one::OutputModuleBase* module_;
    };
    
    void
    OneOutputModuleCommunicator::closeFile() {
      module().doCloseFile();
    }
    
    bool
    OneOutputModuleCommunicator::shouldWeCloseFile() const {
      return module().shouldWeCloseFile();
    }
    
    void
    OneOutputModuleCommunicator::openNewFileIfNeeded() {
      module().maybeOpenFile();
    }
    
    void
    OneOutputModuleCommunicator::openFile(edm::FileBlock const& fb) {
      module().doOpenFile(fb);
    }
    
    void
    OneOutputModuleCommunicator::writeRun(edm::RunPrincipal const& rp, ProcessContext const* processContext) {
      GlobalContext globalContext(GlobalContext::Transition::kWriteRun,
                                  LuminosityBlockID(rp.run(), 0),
                                  rp.index(),
                                  LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                                  rp.endTime(),
                                  processContext);
      ParentContext parentContext(&globalContext);
      ModuleCallingContext mcc(&description(), ModuleCallingContext::State::kRunning, parentContext);
      module().doWriteRun(rp, &mcc);
    }

    void
    OneOutputModuleCommunicator::writeLumi(edm::LuminosityBlockPrincipal const& lbp, ProcessContext const* processContext) {
      GlobalContext globalContext(GlobalContext::Transition::kWriteLuminosityBlock,
                                  lbp.id(),
                                  lbp.runPrincipal().index(),
                                  lbp.index(),
                                  lbp.beginTime(),
                                  processContext);
      ParentContext parentContext(&globalContext);
      ModuleCallingContext mcc(&description(), ModuleCallingContext::State::kRunning, parentContext);
      module().doWriteLuminosityBlock(lbp, &mcc);
    }
    
    bool OneOutputModuleCommunicator::wantAllEvents() const {return module().wantAllEvents();}
    
    bool OneOutputModuleCommunicator::limitReached() const {return module().limitReached();}
    
    void OneOutputModuleCommunicator::configure(OutputModuleDescription const& desc) {module().configure(desc);}
    
    edm::SelectionsArray const& OneOutputModuleCommunicator::keptProducts() const {
      return module().keptProducts();
    }
    
    void OneOutputModuleCommunicator::selectProducts(edm::ProductRegistry const& preg) {
      module().selectProducts(preg);
    }
    
    void OneOutputModuleCommunicator::setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                                                                bool anyProductProduced) {
      module().setEventSelectionInfo(outputModulePathPositions, anyProductProduced);
    }
    
    ModuleDescription const& OneOutputModuleCommunicator::description() const {
      return module().description();
    }
    
    
    
    std::unique_ptr<OutputModuleCommunicator>
    OutputWorker::createOutputModuleCommunicator() {
      return std::move(std::unique_ptr<OutputModuleCommunicator>{new OneOutputModuleCommunicator{& this->module()}});
    }
  }
}
