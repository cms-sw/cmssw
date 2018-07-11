#ifndef FWCore_Framework_OutputModule_h
#define FWCore_Framework_OutputModule_h

/*----------------------------------------------------------------------

OutputModule: The base class of all "modules" that write Events to an
output stream.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/SelectedProducts.h"

#include "FWCore/Framework/interface/TriggerResultsBasedEventSelector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Framework/interface/ProductSelector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/getAllTriggerNames.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <atomic>
#include <set>

namespace edm {

  class MergeableRunProductMetadata;
  class ModuleCallingContext;
  class PreallocationConfiguration;
  class ActivityRegistry;
  class ProductRegistry;
  class ThinnedAssociationsHelper;
  class WaitingTask;

  namespace maker {
    template<typename T> class ModuleHolderT;
  }

  typedef detail::TriggerResultsBasedEventSelector::handle_t Trig;

  class OutputModule : public EDConsumerBase {
  public:
    template <typename T> friend class maker::ModuleHolderT;
    template <typename T> friend class WorkerT;
    template <typename T> friend class OutputModuleCommunicatorT;
    typedef OutputModule ModuleType;

    explicit OutputModule(ParameterSet const& pset);
    ~OutputModule() override;

    OutputModule(OutputModule const&) = delete; // Disallow copying and moving
    OutputModule& operator=(OutputModule const&) = delete; // Disallow copying and moving

    /// Accessor for maximum number of events to be written.
    /// -1 is used for unlimited.
    int maxEvents() const {return maxEvents_;}

    /// Accessor for remaining number of events to be written.
    /// -1 is used for unlimited.
    int remainingEvents() const {return remainingEvents_;}

    bool selected(BranchDescription const& desc) const;

    void selectProducts(ProductRegistry const& preg, ThinnedAssociationsHelper const&);
    std::string const& processName() const {return process_name_;}
    SelectedProductsForBranchType const& keptProducts() const {return keptProducts_;}
    std::array<bool, NumBranchTypes> const& hasNewlyDroppedBranch() const {return hasNewlyDroppedBranch_;}

    static void fillDescription(ParameterSetDescription & desc, std::vector<std::string> const& iDefaultOutputCommands = ProductSelectorRules::defaultSelectionStrings());
    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    static const std::string& baseType();
    static void prevalidate(ConfigurationDescriptions& );

    static bool wantsGlobalRuns() {return true;}
    static bool wantsGlobalLuminosityBlocks() {return true;}
    static bool wantsStreamRuns() {return false;}
    static bool wantsStreamLuminosityBlocks() {return false;};

    SerialTaskQueue* globalRunsQueue() { return &runQueue_;}
    SerialTaskQueue* globalLuminosityBlocksQueue() { return &luminosityBlockQueue_;}
    SharedResourcesAcquirer& sharedResourcesAcquirer() {
      return resourceAcquirer_;
    }

    bool wantAllEvents() const {return wantAllEvents_;}

    BranchIDLists const* branchIDLists();

    ThinnedAssociationsHelper const* thinnedAssociationsHelper() const;

  protected:

    // This function is needed for compatibility with older code. We
    // need to clean up the use of EventForOutputand EventPrincipal, to avoid
    // creation of multiple EventForOutputobjects when handling a single
    // event.
    Trig getTriggerResults(EDGetTokenT<TriggerResults> const& token, EventForOutput const& e) const;

    ModuleDescription const& description() const;
    ModuleDescription const& moduleDescription() const { return moduleDescription_;
    }

    ParameterSetID selectorConfig() const { return selector_config_id_; }

    void doPreallocate(PreallocationConfiguration const&);

    void doBeginJob();
    void doEndJob();
    bool doEvent(EventPrincipal const& ep, EventSetup const& c,
                 ActivityRegistry* act,
                 ModuleCallingContext const* mcc);
    //Needed by WorkerT but not supported
    void preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const {}

    bool doBeginRun(RunPrincipal const& rp, EventSetup const& c,
                    ModuleCallingContext const* mcc);
    bool doEndRun(RunPrincipal const& rp, EventSetup const& c,
                  ModuleCallingContext const* mcc);
    bool doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                ModuleCallingContext const* mcc);
    bool doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                              ModuleCallingContext const* mcc);

    void setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                               bool anyProductProduced);

    void configure(OutputModuleDescription const& desc);

    std::map<BranchID::value_type, BranchID::value_type> const& droppedBranchIDToKeptBranchID() {
      return droppedBranchIDToKeptBranchID_;
    }

  private:

    int maxEvents_;
    std::atomic<int> remainingEvents_;

    // TODO: Give OutputModule
    // an interface (protected?) that supplies client code with the
    // needed functionality *without* giving away implementation
    // details ... don't just return a reference to keptProducts_, because
    // we are looking to have the flexibility to change the
    // implementation of keptProducts_ without modifying clients. When this
    // change is made, we'll have a one-time-only task of modifying
    // clients (classes derived from OutputModule) to use the
    // newly-introduced interface.
    // TODO: Consider using shared pointers here?

    // keptProducts_ are pointers to the BranchDescription objects describing
    // the branches we are to write.
    //
    // We do not own the BranchDescriptions to which we point.
    SelectedProductsForBranchType keptProducts_;
    std::array<bool, NumBranchTypes> hasNewlyDroppedBranch_;

    std::string process_name_;
    ProductSelectorRules productSelectorRules_;
    ProductSelector productSelector_;
    ModuleDescription moduleDescription_;

    bool wantAllEvents_;
    std::vector<detail::TriggerResultsBasedEventSelector> selectors_;
    // ID of the ParameterSet that configured the event selector
    // subsystem.
    ParameterSet selectEvents_;
    ParameterSetID selector_config_id_;

    // needed because of possible EDAliases.
    // filled in only if key and value are different.
    std::map<BranchID::value_type, BranchID::value_type> droppedBranchIDToKeptBranchID_;
    edm::propagate_const<std::unique_ptr<BranchIDLists>> branchIDLists_;
    BranchIDLists const* origBranchIDLists_;

    edm::propagate_const<std::unique_ptr<ThinnedAssociationsHelper>> thinnedAssociationsHelper_;
    std::map<BranchID, bool> keepAssociation_;

    SharedResourcesAcquirer resourceAcquirer_;
    SerialTaskQueue runQueue_;
    SerialTaskQueue luminosityBlockQueue_;

    //------------------------------------------------------------------
    // private member functions
    //------------------------------------------------------------------
    void doWriteRun(RunPrincipal const& rp, ModuleCallingContext const* mcc, MergeableRunProductMetadata const*);
    void doWriteLuminosityBlock(LuminosityBlockPrincipal const& lbp, ModuleCallingContext const* mcc);
    void doOpenFile(FileBlock const& fb);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doRegisterThinnedAssociations(ProductRegistry const&,
                                       ThinnedAssociationsHelper&) { }

    std::string workerType() const {return "WorkerT<OutputModule>";}
    
    /// Tell the OutputModule that is must end the current file.
    void doCloseFile();

    // Do the end-of-file tasks; this is only called internally, after
    // the appropriate tests have been done.
    virtual void reallyCloseFile();

    void registerProductsAndCallbacks(OutputModule const*, ProductRegistry const*) {}
    
    bool needToRunSelection() const;
    std::vector<ProductResolverIndexAndSkipBit> productsUsedBySelection() const;
    bool prePrefetchSelection(StreamID id, EventPrincipal const&, ModuleCallingContext const*);

    /// Ask the OutputModule if we should end the current file.
    virtual bool shouldWeCloseFile() const {return false;}

    virtual void write(EventForOutput const&) = 0;
    virtual void beginJob(){}
    virtual void endJob(){}
    virtual void beginRun(RunForOutput const&){}
    virtual void endRun(RunForOutput const&){}
    virtual void writeRun(RunForOutput const&) = 0;
    virtual void beginLuminosityBlock(LuminosityBlockForOutput const&){}
    virtual void endLuminosityBlock(LuminosityBlockForOutput const&){}
    virtual void writeLuminosityBlock(LuminosityBlockForOutput const&) = 0;
    virtual void openFile(FileBlock const&) {}
    virtual void respondToOpenInputFile(FileBlock const&) {}
    virtual void respondToCloseInputFile(FileBlock const&) {}

    virtual void setProcessesWithSelectedMergeableRunProducts(std::set<std::string> const&) {}

    bool hasAcquire() const { return false; }
    bool hasAccumulator() const { return false; }

    virtual bool isFileOpen() const { return true; }

    void keepThisBranch(BranchDescription const& desc,
                        std::map<BranchID, BranchDescription const*>& trueBranchIDToKeptBranchDesc,
                        std::set<BranchID>& keptProductsInEvent);

    void setModuleDescription(ModuleDescription const& md) {
      moduleDescription_ = md;
    }

    bool limitReached() const {return remainingEvents_ == 0;}
  };
}
#endif
