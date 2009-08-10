/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModule.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/CPCSentry.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  // This grotesque little function exists just to allow calling of
  // ConstProductRegistry::allBranchDescriptions in the context of
  // OutputModule's initialization list, rather than in the body of
  // the constructor.

  std::vector<BranchDescription const*>
  getAllBranchDescriptions() {
    Service<ConstProductRegistry> reg;
    return reg->allBranchDescriptions();
  }

  std::vector<std::string> const& getAllTriggerNames() {
    Service<service::TriggerNamesService> tns;
    return tns->getTrigPaths();
  }
}


namespace {
  //--------------------------------------------------------
  // Remove whitespace (spaces and tabs) from a std::string.
  void remove_whitespace(std::string& s) {
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    s.erase(std::remove(s.begin(), s.end(), '\t'), s.end());
  }

  void test_remove_whitespace() {
    std::string a("noblanks");
    std::string b("\t   no   blanks    \t");

    remove_whitespace(b);
    assert(a == b);
  }

  //--------------------------------------------------------
  // Given a path-spec (std::string of the form "a:b", where the ":b" is
  // optional), return a parsed_path_spec_t containing "a" and "b".

  typedef std::pair<std::string, std::string> parsed_path_spec_t;
  void parse_path_spec(std::string const& path_spec, 
		       parsed_path_spec_t& output) {
    std::string trimmed_path_spec(path_spec);
    remove_whitespace(trimmed_path_spec);
    
    std::string::size_type colon = trimmed_path_spec.find(":");
    if (colon == std::string::npos) {
	output.first = trimmed_path_spec;
    } else {
	output.first  = trimmed_path_spec.substr(0, colon);
	output.second = trimmed_path_spec.substr(colon + 1, 
						 trimmed_path_spec.size());
    }
  }

  void test_parse_path_spec() {
    std::vector<std::string> paths;
    paths.push_back("a:p1");
    paths.push_back("b:p2");
    paths.push_back("  c");
    paths.push_back("ddd\t:p3");
    paths.push_back("eee:  p4  ");
    
    std::vector<parsed_path_spec_t> parsed(paths.size());
    for (size_t i = 0; i < paths.size(); ++i)
      parse_path_spec(paths[i], parsed[i]);

    assert(parsed[0].first  == "a");
    assert(parsed[0].second == "p1");
    assert(parsed[1].first  == "b");
    assert(parsed[1].second == "p2");
    assert(parsed[2].first  == "c");
    assert(parsed[2].second == "");
    assert(parsed[3].first  == "ddd");
    assert(parsed[3].second == "p3");
    assert(parsed[4].first  == "eee");
    assert(parsed[4].second == "p4");
  }
}

namespace edm {
  namespace test {
    void run_all_output_module_tests() {
      test_remove_whitespace();
      test_parse_path_spec();
    }
  }


  // -------------------------------------------------------
  OutputModule::OutputModule(ParameterSet const& pset) : 
    maxEvents_(-1),
    remainingEvents_(maxEvents_),
    keptProducts_(),
    hasNewlyDroppedBranch_(),
    process_name_(),
    groupSelectorRules_(pset, "outputCommands", "OutputModule"),
    groupSelector_(),
    moduleDescription_(),
    current_context_(0),
    prodsValid_(false),
    wantAllEvents_(false),
    selectors_(),
    selector_config_id_(),
    branchParents_(),
    branchChildren_() {

    hasNewlyDroppedBranch_.assign(false);

    Service<service::TriggerNamesService> tns;
    process_name_ = tns->getProcessName();

    ParameterSet selectevents =
      pset.getUntrackedParameter("SelectEvents", ParameterSet());

    selector_config_id_ = selectevents.id();
    // If selectevents is an emtpy ParameterSet, then we are to write
    // all events, or one which contains a vstrig 'SelectEvents' that
    // is empty, we are to write all events. We have no need for any
    // EventSelectors.
    if (selectevents.empty()) {
	wantAllEvents_ = true;
	selectors_.setupDefault(getAllTriggerNames());
	return;
    }

    std::vector<std::string> path_specs = 
      selectevents.getParameter<std::vector<std::string> >("SelectEvents");

    if (path_specs.empty()) {
	wantAllEvents_ = true;
	selectors_.setupDefault(getAllTriggerNames());
	return;
    }

    // If we get here, we have the possibility of having to deal with
    // path_specs that look at more than one process.
    std::vector<parsed_path_spec_t> parsed_paths(path_specs.size());
    for (size_t i = 0; i < path_specs.size(); ++i) {
      parse_path_spec(path_specs[i], parsed_paths[i]);
    }
    selectors_.setup(parsed_paths, getAllTriggerNames(), process_name_);
  }

  void OutputModule::configure(OutputModuleDescription const& desc) {
    remainingEvents_ = maxEvents_ = desc.maxEvents_;
  }

  void OutputModule::selectProducts() {
    if (groupSelector_.initialized()) return;
    groupSelector_.initialize(groupSelectorRules_, getAllBranchDescriptions());
    Service<ConstProductRegistry> reg;

    // TODO: See if we can collapse keptProducts_ and groupSelector_ into a
    // single object. See the notes in the header for GroupSelector
    // for more information.

    ProductRegistry::ProductList::const_iterator it  = 
      reg->productList().begin();
    ProductRegistry::ProductList::const_iterator end = 
      reg->productList().end();

    for (; it != end; ++it) {
      BranchDescription const& desc = it->second;
      if(desc.transient()) {
	// if the class of the branch is marked transient, output nothing
      } else if(!desc.present() && !desc.produced()) {
	// else if the branch containing the product has been previously dropped,
	// output nothing
      } else if (selected(desc)) {
	// else if the branch has been selected, put it in the list of selected branches
        keptProducts_[desc.branchType()].push_back(&desc);
      } else {
	// otherwise, output nothing,
	// and mark the fact that there is a newly dropped branch of this type.
        hasNewlyDroppedBranch_[desc.branchType()] = true;
      }
    }
  }

  OutputModule::~OutputModule() { }

  void OutputModule::doBeginJob(EventSetup const& c) {
    selectProducts();
    beginJob(c);
  }

  void OutputModule::doEndJob() {
    endJob();
  }


  Trig OutputModule::getTriggerResults(Event const& ev) const {
    return selectors_.getOneTriggerResults(ev);
  }

  Trig OutputModule::getTriggerResults(EventPrincipal const& ep) const {
    // This is bad, because we're returning handles into an Event that
    // is destructed before the return. It might not fail, because the
    // actual EventPrincipal is not destroyed, but it still needs to
    // be cleaned up.
    Event ev(const_cast<EventPrincipal&>(ep), 
	     *current_context_->moduleDescription());
    return getTriggerResults(ev);
  }

   namespace {
     class  PVSentry {
     public:
       PVSentry (detail::CachedProducts& prods, bool& valid) : p(prods), v(valid) {}
       ~PVSentry() {
         p.clear();
         v = false;
       }
     private:
       detail::CachedProducts& p;
       bool&           v;

       PVSentry(PVSentry const&);  // not implemented
       PVSentry& operator=(PVSentry const&); // not implemented
     };
   }

  bool
  OutputModule::doEvent(EventPrincipal const& ep,
			EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    PVSentry          products_sentry(selectors_, prodsValid_);

    FDEBUG(2) << "writeEvent called\n";

    // This ugly little bit is here to prevent making the Event
    // if we don't need it.
    if (!wantAllEvents_) {
      // use module description and const_cast unless interface to
      // event is changed to just take a const EventPrincipal
      Event e(const_cast<EventPrincipal&>(ep), moduleDescription_);
      if (!selectors_.wantEvent(e)) {
	return true;
      }
    }
    write(ep);
    updateBranchParents(ep);
    if (remainingEvents_ > 0) {
      --remainingEvents_;
    }
    return true;
  }

//   bool OutputModule::wantEvent(Event const& ev)
//   {
//     getTriggerResults(ev);
//     bool eventAccepted = false;

//     typedef std::vector<NamedEventSelector>::const_iterator iter;
//     for (iter i = selectResult_.begin(), e = selectResult_.end();
// 	 !eventAccepted && i != e; ++i) 
//       {
// 	eventAccepted = i->acceptEvent(*prods_);
//       }

//     FDEBUG(2) << "Accept event " << ep.id() << " " << eventAccepted << "\n";
//     return eventAccepted;
//   }

  bool
  OutputModule::doBeginRun(RunPrincipal const& rp,
				EventSetup const& c,
				CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    FDEBUG(2) << "beginRun called\n";
    beginRun(rp);
    return true;
  }

  bool
  OutputModule::doEndRun(RunPrincipal const& rp,
			      EventSetup const& c,
			      CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    FDEBUG(2) << "endRun called\n";
    endRun(rp);
    return true;
  }

  void
  OutputModule::doWriteRun(RunPrincipal const& rp) {
    FDEBUG(2) << "writeRun called\n";
    writeRun(rp);
  }

  bool
  OutputModule::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
					    EventSetup const& c,
					    CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    FDEBUG(2) << "beginLuminosityBlock called\n";
    beginLuminosityBlock(lbp);
    return true;
  }

  bool
  OutputModule::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
					  EventSetup const& c,
					  CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    FDEBUG(2) << "endLuminosityBlock called\n";
    endLuminosityBlock(lbp);
    return true;
  }

  void OutputModule::doWriteLuminosityBlock(LuminosityBlockPrincipal const& lbp) {
    FDEBUG(2) << "writeLuminosityBlock called\n";
    writeLuminosityBlock(lbp);
  }

  void OutputModule::doOpenFile(FileBlock const& fb) {
    openFile(fb);
  }

  void OutputModule::doRespondToOpenInputFile(FileBlock const& fb) {
    respondToOpenInputFile(fb);
  }

  void OutputModule::doRespondToCloseInputFile(FileBlock const& fb) {
    respondToCloseInputFile(fb);
  }

  void OutputModule::doRespondToOpenOutputFiles(FileBlock const& fb) {
    respondToOpenOutputFiles(fb);
  }

  void OutputModule::doRespondToCloseOutputFiles(FileBlock const& fb) {
    respondToCloseOutputFiles(fb);
  }

  void 
  OutputModule::doPreForkReleaseResources() {
    preForkReleaseResources();
  }
  
  void 
  OutputModule::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
    postForkReacquireResources(iChildIndex, iNumberOfChildren);
  }
  
  void OutputModule::maybeOpenFile() {
    if (!isFileOpen()) doOpenFile();
  }
  
  void OutputModule::doCloseFile() {
    if (isFileOpen()) reallyCloseFile();
  }

  void OutputModule::reallyCloseFile() {
    fillDependencyGraph();
    startEndFile();
    writeFileFormatVersion();
    writeFileIdentifier();
    writeFileIndex();
    writeEventHistory();
    writeProcessConfigurationRegistry();
    writeProcessHistoryRegistry();
    writeParameterSetRegistry();
    writeProductDescriptionRegistry();
    writeParentageRegistry();
    writeBranchIDListRegistry();
    writeProductDependencies();
    writeBranchMapper();
    finishEndFile();
    branchParents_.clear();
    branchChildren_.clear();
  }

  CurrentProcessingContext const*
  OutputModule::currentContext() const {
    return current_context_;
  }

  ModuleDescription const&
  OutputModule::description() const {
    return moduleDescription_;
  }

  bool OutputModule::selected(BranchDescription const& desc) const {
    return groupSelector_.selected(desc);
  }

  void
  OutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addUnknownLabel(desc);
  }

  std::string
  OutputModule::baseType() {
    return std::string("OutputModule");
  }

  void
  OutputModule::updateBranchParents(EventPrincipal const& ep) {
    for (EventPrincipal::const_iterator i = ep.begin(), iEnd = ep.end(); i != iEnd; ++i) {
      if (*i && (*i)->productProvenancePtr() != 0) {
	BranchID const& bid = (*i)->branchDescription().branchID();
	BranchParents::iterator it = branchParents_.find(bid);
	if (it == branchParents_.end()) {
	   it = branchParents_.insert(std::make_pair(bid, std::set<ParentageID>())).first;
	}
	it->second.insert((*i)->productProvenancePtr()->parentageID());
	branchChildren_.insertEmpty(bid);
      }
    }
  }

  void
  OutputModule::fillDependencyGraph() {
    for (BranchParents::const_iterator i = branchParents_.begin(), iEnd = branchParents_.end();
        i != iEnd; ++i) {
      BranchID const& child = i->first;
      std::set<ParentageID> const& eIds = i->second;
      for (std::set<ParentageID>::const_iterator it = eIds.begin(), itEnd = eIds.end();
          it != itEnd; ++it) {
        Parentage entryDesc;
        ParentageRegistry::instance()->getMapped(*it, entryDesc);
	std::vector<BranchID> const& parents = entryDesc.parents();
	for (std::vector<BranchID>::const_iterator j = parents.begin(), jEnd = parents.end();
	  j != jEnd; ++j) {
	  branchChildren_.insertChild(*j, child);
	}
      }
    }
  }
}
