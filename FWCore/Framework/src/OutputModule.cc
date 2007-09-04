/*----------------------------------------------------------------------

$Id: OutputModule.cc,v 1.35 2007/07/09 07:29:51 llista Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModule.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/CPCSentry.h"

using std::vector;
using std::string;


namespace edm {
  // This grotesque little function exists just to allow calling of
  // ConstProductRegistry::allBranchDescriptions in the context of
  // OutputModule's initialization list, rather than in the body of
  // the constructor.

  vector<edm::BranchDescription const*>
  getAllBranchDescriptions() {
    edm::Service<edm::ConstProductRegistry> reg;
    return reg->allBranchDescriptions();
  }

  vector<string> const& getAllTriggerNames() {
    edm::Service<edm::service::TriggerNamesService> tns;
    return tns->getTrigPaths();
  }
}


namespace
{
  //--------------------------------------------------------
  // Remove whitespace (spaces and tabs) from a string.
  void remove_whitespace(string& s) {
    s.erase(remove(s.begin(), s.end(), ' '), s.end());
    s.erase(remove(s.begin(), s.end(), '\t'), s.end());
  }

  void test_remove_whitespace() {
    string a("noblanks");
    string b("\t   no   blanks    \t");

    remove_whitespace(b);
    assert(a == b);
  }

  //--------------------------------------------------------
  // Given a path-spec (string of the form "a:b", where the ":b" is
  // optional), return a parsed_path_spec_t containing "a" and "b".

  typedef std::pair<string,string> parsed_path_spec_t;
  void parse_path_spec(string const& path_spec, 
		       parsed_path_spec_t& output) {
    string trimmed_path_spec(path_spec);
    remove_whitespace(trimmed_path_spec);
    
    string::size_type colon = trimmed_path_spec.find(":");
    if (colon == string::npos) {
	output.first = trimmed_path_spec;
    } else {
	output.first  = trimmed_path_spec.substr(0, colon);
	output.second = trimmed_path_spec.substr(colon+1, 
						 trimmed_path_spec.size());
    }
  }

  void test_parse_path_spec() {
    vector<string> paths;
    paths.push_back("a:p1");
    paths.push_back("b:p2");
    paths.push_back("  c");
    paths.push_back("ddd\t:p3");
    paths.push_back("eee:  p4  ");
    
    vector<parsed_path_spec_t> parsed(paths.size());
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
    nextID_(),
    descVec_(),
    droppedVec_(),
    hasNewlyDroppedBranch_(),
    process_name_(),
    groupSelector_(pset),
    //eventSelectors_(),
    //selectResult_("*"),  // use the most recent process name
    moduleDescription_(),
    current_context_(0),
    //prods_(),
    prodsValid_(false),
    current_md_(0),
    wantAllEvents_(false),
    selectors_(),
    eventCount_(0)
  {
    hasNewlyDroppedBranch_.assign(false);

    edm::Service<edm::service::TriggerNamesService> tns;
    process_name_ = tns->getProcessName();

    ParameterSet selectevents =
      pset.getUntrackedParameter("SelectEvents", ParameterSet());


    // If selectevents is an emtpy ParameterSet, then we are to write
    // all events, or one which contains a vstrig 'SelectEvents' that
    // is empty, we are to write all events. We have no need for any
    // EventSelectors.
    if (selectevents.empty()) {
	wantAllEvents_ = true;
	selectors_.setupDefault(getAllTriggerNames());
	return;
    }

    vector<string> path_specs = 
      selectevents.getParameter<vector<string> >("SelectEvents");

    if (path_specs.empty()) {
	wantAllEvents_ = true;
	selectors_.setupDefault(getAllTriggerNames());
	return;
    }

    // If we get here, we have the possibility of having to deal with
    // path_specs that look at more than one process.
    vector<parsed_path_spec_t> parsed_paths(path_specs.size());
    for (size_t i = 0; i < path_specs.size(); ++i)
      parse_path_spec(path_specs[i], parsed_paths[i]);

    selectors_.setup(parsed_paths, getAllTriggerNames(), process_name_);
  }

  void OutputModule::selectProducts() {
    if (groupSelector_.initialized()) return;
    groupSelector_.initialize(getAllBranchDescriptions());
    Service<ConstProductRegistry> reg;
    nextID_ = reg->nextID();

    // TODO: See if we can collapse descVec_ and groupSelector_ into a
    // single object. See the notes in the header for GroupSelector
    // for more information.

    ProductRegistry::ProductList::const_iterator it  = 
      reg->productList().begin();
    ProductRegistry::ProductList::const_iterator end = 
      reg->productList().end();

    for (; it != end; ++it) {
      BranchDescription const& desc = it->second;
      if(!desc.provenancePresent() & !desc.produced()) {
	// If the branch containing the provenance has been previously dropped,
	// and the product has not been produced again, output nothing
	continue;
      } else if(desc.transient()) {
	// else if the class of the branch is marked transient, drop the product branch
	droppedVec_[desc.branchType()].push_back(&desc);
      } else if(!desc.present() & !desc.produced()) {
	// else if the branch containing the product has been previously dropped,
	// and the product has not been produced again, drop the product branch again.
	droppedVec_[desc.branchType()].push_back(&desc);
      } else if (selected(desc)) {
	// else if the branch has been selected, put it in the list of selected branches
	descVec_[desc.branchType()].push_back(&desc);
      } else {
	// otherwise, drop the product branch.
	droppedVec_[desc.branchType()].push_back(&desc);
	// mark the fact that there is a newly dropped branch of this type.
	hasNewlyDroppedBranch_[desc.branchType()] = true;
      }
    }
  }

  OutputModule::~OutputModule() { }

  void OutputModule::doBeginJob(EventSetup const& c) {
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
       ~PVSentry() { p.clear(); v=false; }
     private:
       detail::CachedProducts& p;
       bool&           v;

       PVSentry(PVSentry const&);  // not implemented
       PVSentry& operator=(PVSentry const&); // not implemented
     };
   }

  void OutputModule::writeEvent(EventPrincipal const& ep,
				ModuleDescription const& md,
				CurrentProcessingContext const* c)
  {
    detail::CPCSentry sentry(current_context_, c);
    PVSentry          products_sentry(selectors_, prodsValid_);

    //Save the current Mod Desc
    current_md_ = &md;
    assert(current_md_ == c->moduleDescription());

    FDEBUG(2) << "writeEvent called\n";

    // This ugly little bit is here to prevent making the Event if
    // don't need it.
    if (wantAllEvents_) {
      write(ep); 
      ++eventCount_;
    } else {
      // use module description and const_cast unless interface to
      // event is changed to just take a const EventPrincipal
      Event e(const_cast<EventPrincipal&>(ep), *c->moduleDescription());
      if (selectors_.wantEvent(e)) {
	write(ep);
        ++eventCount_;
      }
    }
  }

//   bool OutputModule::wantEvent(Event const& ev)
//   {
//     getTriggerResults(ev);
//     bool eventAccepted = false;

//     typedef vector<NamedEventSelector>::const_iterator iter;
//     for (iter i = selectResult_.begin(), e = selectResult_.end();
// 	 !eventAccepted && i!=e; ++i) 
//       {
// 	eventAccepted = i->acceptEvent(*prods_);
//       }

//     FDEBUG(2) << "Accept event " << ep.id() << " " << eventAccepted << "\n";
//     return eventAccepted;
//   }

  void OutputModule::doBeginRun(RunPrincipal const& rp,
				ModuleDescription const& md,
				CurrentProcessingContext const* c)
  {
    detail::CPCSentry sentry(current_context_, c);
    //Save the current Mod Desc
    current_md_ = &md;
    assert (current_md_ == c->moduleDescription());
    FDEBUG(2) << "beginRun called\n";
    beginRun(rp);
  }

  void OutputModule::doEndRun(RunPrincipal const& rp,
			      ModuleDescription const& md,
			      CurrentProcessingContext const* c)
  {
    detail::CPCSentry sentry(current_context_, c);
    //Save the current Mod Desc
    current_md_ = &md;
    assert (current_md_ == c->moduleDescription());
    FDEBUG(2) << "endRun called\n";
    endRun(rp);
  }

  void OutputModule::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
					    ModuleDescription const& md,
					    CurrentProcessingContext const* c)
  {
    detail::CPCSentry sentry(current_context_, c);
    //Save the current Mod Desc
    current_md_ = &md;
    assert (current_md_ == c->moduleDescription());
    FDEBUG(2) << "beginLuminosityBlock called\n";
    beginLuminosityBlock(lbp);
  }

  void OutputModule::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
					  ModuleDescription const& md,
					  CurrentProcessingContext const* c)
  {
    detail::CPCSentry sentry(current_context_, c);
    //Save the current Mod Desc
    current_md_ = &md;
    assert (current_md_ == c->moduleDescription());
    FDEBUG(2) << "endLuminosityBlock called\n";
    endLuminosityBlock(lbp);
  }

  void OutputModule::maybeEndFile()
  {
    if (isFileOpen() && isFileFull()) doEndFile();
    // Where should we open a new file? It does not seem that here is
    // the right place...
  }
  
  void OutputModule::doEndFile()
  {
    startEndFile();
    writeFileFormatVersion();
    writeProcessConfigurationRegistry();
    writeProcessHistoryRegistry();
    writeModuleDescriptionRegistry();
    writeParameterSetRegistry();
    writeProductDescriptionRegistry();
    finishEndFile();
  }

  CurrentProcessingContext const*
  OutputModule::currentContext() const
  {
    return current_context_;
  }

  bool OutputModule::selected(BranchDescription const& desc) const
  {
    return groupSelector_.selected(desc);
  }

  unsigned int OutputModule::nextID() const 
  {
    return nextID_;
  }
}
