#ifndef FWCore_Framework_OutputModule_h
#define FWCore_Framework_OutputModule_h

/*----------------------------------------------------------------------

OutputModule: The base class of all "modules" that write Events to an
output stream.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchChildren.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/Selections.h"

#include "FWCore/Framework/interface/CachedProducts.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Framework/interface/ProductSelector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <array>
#include <string>
#include <vector>
#include <map>

namespace edm {

  typedef detail::CachedProducts::handle_t Trig;

  std::vector<std::string> const& getAllTriggerNames();

  class OutputModule : public EDConsumerBase {
  public:
    template <typename T> friend class WorkerT;
    friend class OutputWorker;
    typedef OutputModule ModuleType;
    typedef OutputWorker WorkerType;

    explicit OutputModule(ParameterSet const& pset);
    virtual ~OutputModule();

    OutputModule(OutputModule const&) = delete; // Disallow copying and moving
    OutputModule& operator=(OutputModule const&) = delete; // Disallow copying and moving

    /// Accessor for maximum number of events to be written.
    /// -1 is used for unlimited.
    int maxEvents() const {return maxEvents_;}

    /// Accessor for remaining number of events to be written.
    /// -1 is used for unlimited.
    int remainingEvents() const {return remainingEvents_;}

    bool selected(BranchDescription const& desc) const;

    void selectProducts(ProductRegistry const& preg);
    std::string const& processName() const {return process_name_;}
    SelectionsArray const& keptProducts() const {return keptProducts_;}
    std::array<bool, NumBranchTypes> const& hasNewlyDroppedBranch() const {return hasNewlyDroppedBranch_;}

    static void fillDescription(ParameterSetDescription & desc);
    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    static const std::string& baseType();
    static void prevalidate(ConfigurationDescriptions& );

    BranchChildren const& branchChildren() const {return branchChildren_;}

    bool wantAllEvents() const {return wantAllEvents_;}

    BranchIDLists const* branchIDLists() const;

  protected:

    //Trig const& getTriggerResults(Event const& ep) const;
    Trig getTriggerResults(Event const& ep) const;

    // This function is needed for compatibility with older code. We
    // need to clean up the use of Event and EventPrincipal, to avoid
    // creation of multiple Event objects when handling a single
    // event.
    Trig getTriggerResults(EventPrincipal const& ep) const;

    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('write').
    CurrentProcessingContext const* currentContext() const;

    ModuleDescription const& description() const;

    ParameterSetID selectorConfig() const { return selector_config_id_; }

    void doBeginJob();
    void doEndJob();
    bool doEvent(EventPrincipal const& ep, EventSetup const& c,
                 CurrentProcessingContext const* cpc);
    bool doBeginRun(RunPrincipal const& rp, EventSetup const& c,
                 CurrentProcessingContext const* cpc);
    bool doEndRun(RunPrincipal const& rp, EventSetup const& c,
                 CurrentProcessingContext const* cpc);
    bool doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                 CurrentProcessingContext const* cpc);
    bool doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                 CurrentProcessingContext const* cpc);

    void setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                               bool anyProductProduced);

    void configure(OutputModuleDescription const& desc);

    std::map<BranchID::value_type, BranchID::value_type> const& droppedBranchIDToKeptBranchID() {
      return droppedBranchIDToKeptBranchID_;
    }

  private:

    int maxEvents_;
    int remainingEvents_;

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
    SelectionsArray keptProducts_;
    std::array<bool, NumBranchTypes> hasNewlyDroppedBranch_;

    std::string process_name_;
    ProductSelectorRules productSelectorRules_;
    ProductSelector productSelector_;
    ModuleDescription moduleDescription_;

    // We do not own the pointed-to CurrentProcessingContext.
    CurrentProcessingContext const* current_context_;

    //This will store TriggerResults objects for the current event.
    // mutable std::vector<Trig> prods_;
    mutable bool prodsValid_;

    bool wantAllEvents_;
    mutable detail::CachedProducts selectors_;
    // ID of the ParameterSet that configured the event selector
    // subsystem.
    ParameterSetID selector_config_id_;

    // needed because of possible EDAliases.
    // filled in only if key and value are different.
    std::map<BranchID::value_type, BranchID::value_type> droppedBranchIDToKeptBranchID_;
    std::unique_ptr<BranchIDLists> branchIDLists_;
    BranchIDLists const* origBranchIDLists_;

    typedef std::map<BranchID, std::set<ParentageID> > BranchParents;
    BranchParents branchParents_;

    BranchChildren branchChildren_;

    //------------------------------------------------------------------
    // private member functions
    //------------------------------------------------------------------
    void doWriteRun(RunPrincipal const& rp);
    void doWriteLuminosityBlock(LuminosityBlockPrincipal const& lbp);
    void doOpenFile(FileBlock const& fb);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doRespondToOpenOutputFiles(FileBlock const& fb);
    void doRespondToCloseOutputFiles(FileBlock const& fb);
    void doPreForkReleaseResources();
    void doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);

    std::string workerType() const {return "OutputWorker";}

    /// Tell the OutputModule that is must end the current file.
    void doCloseFile();

    /// Tell the OutputModule to open an output file, if one is not
    /// already open.
    void maybeOpenFile();


    // Do the end-of-file tasks; this is only called internally, after
    // the appropriate tests have been done.
    void reallyCloseFile();

    void registerProductsAndCallbacks(OutputModule const*, ProductRegistry const*) {}

    /// Ask the OutputModule if we should end the current file.
    virtual bool shouldWeCloseFile() const {return false;}

    virtual void write(EventPrincipal const& e) = 0;
    virtual void beginJob(){}
    virtual void endJob(){}
    virtual void beginRun(RunPrincipal const&){}
    virtual void endRun(RunPrincipal const&){}
    virtual void writeRun(RunPrincipal const&) = 0;
    virtual void beginLuminosityBlock(LuminosityBlockPrincipal const&){}
    virtual void endLuminosityBlock(LuminosityBlockPrincipal const&){}
    virtual void writeLuminosityBlock(LuminosityBlockPrincipal const&) = 0;
    virtual void openFile(FileBlock const&) {}
    virtual void respondToOpenInputFile(FileBlock const&) {}
    virtual void respondToCloseInputFile(FileBlock const&) {}
    virtual void respondToOpenOutputFiles(FileBlock const&) {}
    virtual void respondToCloseOutputFiles(FileBlock const&) {}
    virtual void preForkReleaseResources() {}
    virtual void postForkReacquireResources(unsigned int /*iChildIndex*/, unsigned int /*iNumberOfChildren*/) {}

    virtual bool isFileOpen() const { return true; }

    virtual void doOpenFile() { }

    void setModuleDescription(ModuleDescription const& md) {
      moduleDescription_ = md;
    }

    void updateBranchParents(EventPrincipal const& ep);
    void fillDependencyGraph();

    bool limitReached() const {return remainingEvents_ == 0;}

    // The following member functions are part of the Template Method
    // pattern, used for implementing doCloseFile() and maybeEndFile().

    virtual void startEndFile() {}
    virtual void writeFileFormatVersion() {}
    virtual void writeFileIdentifier() {}
    virtual void writeIndexIntoFile() {}
    virtual void writeProcessConfigurationRegistry() {}
    virtual void writeProcessHistoryRegistry() {}
    virtual void writeParameterSetRegistry() {}
    virtual void writeBranchIDListRegistry() {}
    virtual void writeParentageRegistry() {}
    virtual void writeProductDescriptionRegistry() {}
    virtual void writeProductDependencies() {}
    virtual void writeBranchMapper() {}
    virtual void finishEndFile() {}
  };
}

//this is included after the class definition since this header also needs to know about OutputModule
// we put this here since all OutputModules need this header to create their plugin
#include "FWCore/Framework/src/OutputWorker.h"

#endif
