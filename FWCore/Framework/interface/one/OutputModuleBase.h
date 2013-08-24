#ifndef FWCore_Framework_one_OutputModuleBase_h
#define FWCore_Framework_one_OutputModuleBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     OutputModuleBase
// 
/**\class OutputModuleBase OutputModuleBase.h "FWCore/Framework/interface/one/OutputModuleBase.h"

 Description: Base class for all 'one' OutputModules

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 31 Jul 2013 15:37:16 GMT
//

// system include files
#include <array>
#include <string>
#include <vector>
#include <map>


// user include files
#include "DataFormats/Provenance/interface/BranchChildren.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/Selections.h"

#include "FWCore/Framework/interface/TriggerResultsBasedEventSelector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Framework/interface/ProductSelector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/getAllTriggerNames.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

// forward declarations
namespace edm {

  class ModuleCallingContext;
  template <typename T> class OutputModuleCommunicatorT;
  
  namespace maker {
    template<typename T> class ModuleHolderT;
  }

  namespace one {
    
    typedef detail::TriggerResultsBasedEventSelector::handle_t Trig;
    
    class OutputModuleBase : public EDConsumerBase {
    public:
      template <typename U> friend class edm::maker::ModuleHolderT;
      template <typename T> friend class ::edm::WorkerT;
      template <typename T> friend class ::edm::OutputModuleCommunicatorT;
      typedef OutputModuleBase ModuleType;
      typedef OutputWorker WorkerType;
      
      explicit OutputModuleBase(ParameterSet const& pset);
      virtual ~OutputModuleBase();
      
      OutputModuleBase(OutputModuleBase const&) = delete; // Disallow copying and moving
      OutputModuleBase& operator=(OutputModuleBase const&) = delete; // Disallow copying and moving
      
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
      
      Trig getTriggerResults(EventPrincipal const& ep, ModuleCallingContext const*) const;
      
      ModuleDescription const& description() const;
      
      ParameterSetID selectorConfig() const { return selector_config_id_; }
      
      void doBeginJob();
      void doEndJob();
      bool doEvent(EventPrincipal const& ep, EventSetup const& c,
                   ModuleCallingContext const*);
      bool doBeginRun(RunPrincipal const& rp, EventSetup const& c,
                      ModuleCallingContext const*);
      bool doEndRun(RunPrincipal const& rp, EventSetup const& c,
                    ModuleCallingContext const*);
      bool doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                  ModuleCallingContext const*);
      bool doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                ModuleCallingContext const*);
      
      void setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                                 bool anyProductProduced);
      
      void configure(OutputModuleDescription const& desc);
      
      std::map<BranchID::value_type, BranchID::value_type> const& droppedBranchIDToKeptBranchID() {
        return droppedBranchIDToKeptBranchID_;
      }
      
      std::map<BranchID::value_type, BranchID::value_type> const& keptBranchIDToDroppedBranchID() {
        return keptBranchIDToDroppedBranchID_;
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
      
      bool wantAllEvents_;
      mutable detail::TriggerResultsBasedEventSelector selectors_;
      // ID of the ParameterSet that configured the event selector
      // subsystem.
      ParameterSetID selector_config_id_;
      
      // needed because of possible EDAliases.
      // filled in only if key and value are different.
      std::map<BranchID::value_type, BranchID::value_type> droppedBranchIDToKeptBranchID_;
      std::map<BranchID::value_type, BranchID::value_type> keptBranchIDToDroppedBranchID_;
      std::unique_ptr<BranchIDLists> branchIDLists_;
      BranchIDLists const* origBranchIDLists_;
      
      typedef std::map<BranchID, std::set<ParentageID> > BranchParents;
      BranchParents branchParents_;
      
      BranchChildren branchChildren_;
      
      //------------------------------------------------------------------
      // private member functions
      //------------------------------------------------------------------
      void doWriteRun(RunPrincipal const& rp, ModuleCallingContext const*);
      void doWriteLuminosityBlock(LuminosityBlockPrincipal const& lbp, ModuleCallingContext const*);
      void doOpenFile(FileBlock const& fb);
      void doRespondToOpenInputFile(FileBlock const& fb);
      void doRespondToCloseInputFile(FileBlock const& fb);
      void doPreForkReleaseResources();
      void doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);
      
      std::string workerType() const {return "OutputWorker";}
      
      /// Tell the OutputModule that is must end the current file.
      void doCloseFile();
      
      /// Tell the OutputModule to open an output file, if one is not
      /// already open.
      void maybeOpenFile();
      
      void registerProductsAndCallbacks(OutputModuleBase const*, ProductRegistry const*) {}

      
      // Do the end-of-file tasks; this is only called internally, after
      // the appropriate tests have been done.
      virtual void reallyCloseFile();
      
      
      /// Ask the OutputModule if we should end the current file.
      virtual bool shouldWeCloseFile() const {return false;}
      
      virtual void write(EventPrincipal const& e, ModuleCallingContext const*) = 0;
      virtual void beginJob(){}
      virtual void endJob(){}
      virtual void writeLuminosityBlock(LuminosityBlockPrincipal const&, ModuleCallingContext const*) = 0;
      virtual void writeRun(RunPrincipal const&, ModuleCallingContext const*) = 0;
      virtual void openFile(FileBlock const&) {}
      virtual bool isFileOpen() const { return true; }
      virtual void reallyOpenFile() {}
      

      virtual void doBeginRun_(RunPrincipal const&, ModuleCallingContext const*){}
      virtual void doEndRun_(RunPrincipal const&, ModuleCallingContext const*){}
      virtual void doBeginLuminosityBlock_(LuminosityBlockPrincipal const&, ModuleCallingContext const*){}
      virtual void doEndLuminosityBlock_(LuminosityBlockPrincipal const&, ModuleCallingContext const*){}
      virtual void doRespondToOpenInputFile_(FileBlock const&) {}
      virtual void doRespondToCloseInputFile_(FileBlock const&) {}
      
      void setModuleDescription(ModuleDescription const& md) {
        moduleDescription_ = md;
      }
      
      void updateBranchParents(EventPrincipal const& ep);
      void fillDependencyGraph();
      
      bool limitReached() const {return remainingEvents_ == 0;}
    };
  }
}

//this is included after the class definition since this header also needs to know about OutputModule
// we put this here since all OutputModules need this header to create their plugin
#include "FWCore/Framework/src/one/OutputWorker.h"


#endif
