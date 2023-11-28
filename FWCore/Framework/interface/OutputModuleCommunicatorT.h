
#ifndef FWCore_Framework_OutputModuleCommunicatorT_h
#define FWCore_Framework_OutputModuleCommunicatorT_h
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModuleCommunicator.h"
#include "FWCore/Common/interface/FWCoreCommonFwd.h"

namespace edm {
  class ActivityRegistry;
  class MergeableRunProductMetadata;
  class ThinnedAssociationsHelper;

  namespace one {
    class OutputModuleBase;
  }
  namespace global {
    class OutputModuleBase;
  }
  namespace limited {
    class OutputModuleBase;
  }
  namespace impl {
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(void*);
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::one::OutputModuleBase*);
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::global::OutputModuleBase*);
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::limited::OutputModuleBase*);
  }  // namespace impl

  template <typename T>

  class OutputModuleCommunicatorT : public edm::OutputModuleCommunicator {
  public:
    OutputModuleCommunicatorT(T* iModule) : module_(iModule) {}
    void closeFile() override;

    ///\return true if output module wishes to close its file
    bool shouldWeCloseFile() const override;

    ///\return true if no event filtering is applied to OutputModule
    bool wantAllEvents() const override;

    void openFile(edm::FileBlock const& fb) override;

    void writeProcessBlockAsync(WaitingTaskHolder iTask,
                                ProcessBlockPrincipal const&,
                                ProcessContext const*,
                                ActivityRegistry*) override;

    void writeRunAsync(WaitingTaskHolder iTask,
                       edm::RunPrincipal const& rp,
                       ProcessContext const*,
                       ActivityRegistry*,
                       MergeableRunProductMetadata const*) override;

    void writeLumiAsync(WaitingTaskHolder iTask,
                        edm::LuminosityBlockPrincipal const& lbp,
                        ProcessContext const*,
                        ActivityRegistry*) override;

    ///\return true if OutputModule has reached its limit on maximum number of events it wants to see
    bool limitReached() const override;

    void configure(edm::OutputModuleDescription const& desc) override;

    edm::SelectedProductsForBranchType const& keptProducts() const override;

    void selectProducts(edm::ProductRegistry const& preg,
                        ThinnedAssociationsHelper const&,
                        ProcessBlockHelperBase const&) override;

    void setEventSelectionInfo(
        std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
        bool anyProductProduced) override;

    ModuleDescription const& description() const override;

    static std::unique_ptr<edm::OutputModuleCommunicator> createIfNeeded(T* iMod) {
      return impl::createCommunicatorIfNeeded(iMod);
    }

  private:
    inline T& module() const { return *module_; }
    T* module_;
  };
}  // namespace edm
#endif
