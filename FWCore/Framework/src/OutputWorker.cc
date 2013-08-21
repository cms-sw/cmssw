
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModule.h"

#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/src/OutputWorker.h"
#include "FWCore/Framework/src/OutputModuleCommunicator.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"

namespace edm {

  class ProcessContext;

  OutputWorker::OutputWorker(OutputModule* mod,
			     ModuleDescription const& md,
			     WorkerParams const& wp):
  WorkerT<OutputModule>(mod, md, wp)
  {
  }

  OutputWorker::~OutputWorker() {
  }
  
  class ClassicOutputModuleCommunicator: public edm::OutputModuleCommunicator {
  public:
    ClassicOutputModuleCommunicator(edm::OutputModule* iModule):
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
    inline edm::OutputModule& module() const { return *module_;}
    edm::OutputModule* module_;
  };
  
  void
  ClassicOutputModuleCommunicator::closeFile() {
    module().doCloseFile();
  }
  
  bool
  ClassicOutputModuleCommunicator::shouldWeCloseFile() const {
    return module().shouldWeCloseFile();
  }
  
  void
  ClassicOutputModuleCommunicator::openNewFileIfNeeded() {
    module().maybeOpenFile();
  }
  
  void
  ClassicOutputModuleCommunicator::openFile(edm::FileBlock const& fb) {
    module().doOpenFile(fb);
  }

  void
  ClassicOutputModuleCommunicator::writeRun(edm::RunPrincipal const& rp, ProcessContext const* processContext) {
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
  ClassicOutputModuleCommunicator::writeLumi(edm::LuminosityBlockPrincipal const& lbp, ProcessContext const* processContext) {
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

  bool ClassicOutputModuleCommunicator::wantAllEvents() const {return module().wantAllEvents();}
  
  bool ClassicOutputModuleCommunicator::limitReached() const {return module().limitReached();}
  
  void ClassicOutputModuleCommunicator::configure(OutputModuleDescription const& desc) {module().configure(desc);}
  
  edm::SelectionsArray const& ClassicOutputModuleCommunicator::keptProducts() const {
    return module().keptProducts();
  }
  
  void ClassicOutputModuleCommunicator::selectProducts(edm::ProductRegistry const& preg) {
    module().selectProducts(preg);
  }
  
  void ClassicOutputModuleCommunicator::setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                                                    bool anyProductProduced) {
    module().setEventSelectionInfo(outputModulePathPositions, anyProductProduced);
  }

  ModuleDescription const& ClassicOutputModuleCommunicator::description() const {
    return module().description();
  }


  
  std::unique_ptr<OutputModuleCommunicator>
  OutputWorker::createOutputModuleCommunicator() {
    return std::move(std::unique_ptr<OutputModuleCommunicator>{new ClassicOutputModuleCommunicator{& this->module()}});
  }

}
