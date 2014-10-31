
#ifndef FWCore_Framework_OutputModuleCommunicatorT_h
#define FWCore_Framework_OutputModuleCommunicatorT_h
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/OutputModuleCommunicator.h"

namespace edm {
  class OutputModule;
  class ThinnedAssociationsHelper;

  namespace one {
    class OutputModuleBase;
  }
  namespace impl {
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(void *);
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::OutputModule *);
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::one::OutputModuleBase *);
  }
  
  template <typename T>
  
  class OutputModuleCommunicatorT : public edm::OutputModuleCommunicator {
  public:
    OutputModuleCommunicatorT(T* iModule):
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
    
    virtual edm::SelectedProductsForBranchType const& keptProducts() const override;
    
    virtual void selectProducts(edm::ProductRegistry const& preg, ThinnedAssociationsHelper const&) override;
    
    virtual void setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                                       bool anyProductProduced) override;
    
    virtual ModuleDescription const& description() const override;

    static std::unique_ptr<edm::OutputModuleCommunicator> createIfNeeded(T* iMod) {
      return std::move(impl::createCommunicatorIfNeeded(iMod));
      return std::move(std::unique_ptr<edm::OutputModuleCommunicator>{});
    }

  private:
    inline T& module() const { return *module_;}
    T* module_;
  };
}
#endif
