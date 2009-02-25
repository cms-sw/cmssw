#include "TFile.h"
#include "TTree.h"
#include "Cintex/Cintex.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <assert.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <map>

#include "boost/utility.hpp"
#include "boost/scoped_ptr.hpp"

namespace {
typedef std::map<edm::ParameterSetID, edm::ParameterSetBlob> ParameterSetMap;

  class HistoryNode {
  public:
    HistoryNode() :
      config_(),
      simpleId_(0)
    { }

    HistoryNode(const edm::ProcessConfiguration& iConfig, unsigned int iSimpleId) :
      config_(iConfig), 
      simpleId_(iSimpleId) 
    { }

    void addChild(HistoryNode const& child) {
      children_.push_back(child);
    }

    edm::ParameterSetID const& 
    parameterSetID() const {
      return config_.parameterSetID();
    }

    std::string const&
    processName() const {
      return config_.processName();
    }

    std::size_t
    size() const {
      return children_.size();
    }

    HistoryNode *
    lastChildAddress() {
      return &children_.back();
    }

    typedef std::vector<HistoryNode>::const_iterator const_iterator;
    typedef std::vector<HistoryNode>::iterator iterator;
    
    iterator begin() { return children_.begin();}
    iterator end() { return children_.end();}

    const_iterator begin() const { return children_.begin();}
    const_iterator end() const { return children_.end();}

    void print(std::ostream& os) const {
      os << config_.processName() 
	 << " '" << config_.passID() << "' '"
	 << config_.releaseVersion() << "' ["
	 << simpleId_<<"]  (" 
	 << config_.parameterSetID() << ")"
	 << std::endl;
    }

    void printHistory(const std::string& iIndent = std::string("  ")) const;
    void printEventSetupHistory(const ParameterSetMap& iPSM, std::ostream& oErrorLog) const;

    edm::ProcessConfigurationID
    configurationID() const {
      return config_.id();
    }

    static bool sort_;
  private:
    edm::ProcessConfiguration config_;
    std::vector<HistoryNode>  children_;
    unsigned int              simpleId_;    
  };

  std::ostream& operator<<(std::ostream& os, const HistoryNode& node) {
    node.print(os);
    return os;
  }
  bool HistoryNode::sort_ = false;
}


std::ostream&
operator<< (std::ostream& os, edm::ProcessHistory& iHist)
{
  const std::string indentDelta("  ");
  std::string indent = indentDelta;
  for (edm::ProcessHistory::const_iterator i = iHist.begin(), e = iHist.end();
       i != e;
       ++i) {
    os << indent 
       <<i->processName() <<" '"
       <<i->passID()      <<"' '"
       <<i->releaseVersion() << "' ("
       <<i->parameterSetID()<<")"
       << std::endl;
    indent += indentDelta;
  }
  return os;
}

void HistoryNode::printHistory(const std::string& iIndent) const
{
  const std::string indentDelta("  ");
  std::string indent = iIndent;
  for (const_iterator i = begin(), e = end();
       i != e;
       ++i) {
    std::cout << indent << *i;
    i->printHistory(indent+indentDelta);
  }
}

std::string eventSetupComponent(const char* iType, const std::string& iCompName, const edm::ParameterSet& iProcessConfig, const std::string& iProcessName) {
  std::ostringstream result;
  const edm::ParameterSet& pset = iProcessConfig.getParameter<edm::ParameterSet>(iCompName);
  std::string name( pset.getParameter<std::string>("@module_label") );
  if(0 == name.size() ) {
    name = pset.getParameter<std::string>("@module_type");
  }
  
  result <<iType<<": "<< name<<" "<<iProcessName << "\n"
         <<" parameters: "<<pset;
  return result.str();
}

void HistoryNode::printEventSetupHistory(const ParameterSetMap& iPSM, ostream& oErrorLog) const
{
  for (const_iterator itH = begin(), e = end();
       itH != e;
       ++itH) {
    //Get ParameterSet for process
    ParameterSetMap::const_iterator itFind = iPSM.find(itH->parameterSetID());
    if(itFind == iPSM.end()){
      oErrorLog << "No ParameterSetID for " << itH->parameterSetID() << std::endl;
    } else {
      edm::ParameterSet processConfig(itFind->second.pset_);
      std::vector<std::string> sourceStrings, moduleStrings;
      //get the sources
      std::vector<std::string> sources = processConfig.getParameter<std::vector<std::string> >("@all_essources");
      for(std::vector<std::string>::iterator itM = sources.begin(); itM != sources.end(); ++itM) {
        sourceStrings.push_back(eventSetupComponent("ESSource",
                                 *itM,
                                 processConfig,
                                 itH->processName()) );
      }
      //get the modules
      std::vector<std::string> modules = processConfig.getParameter<std::vector<std::string> >("@all_esmodules");
      for(std::vector<std::string>::iterator itM = modules.begin(); itM != modules.end(); ++itM) {
        moduleStrings.push_back(eventSetupComponent("ESModule",
                                 *itM,
                                 processConfig,
                                 itH->processName()));
      }
      if(sort_) 
      {
        std::sort(sourceStrings.begin(), sourceStrings.end());
        std::sort(moduleStrings.begin(), moduleStrings.end());
      }
      std::copy(sourceStrings.begin(), sourceStrings.end(), 
                std::ostream_iterator<std::string>(std::cout,"\n"));
      std::copy(moduleStrings.begin(), moduleStrings.end(), 
                std::ostream_iterator<std::string>(std::cout,"\n"));

    }
    itH->printEventSetupHistory(iPSM, oErrorLog);
  }
}

namespace {

  // Open the input file, returning the TFile object that represents
  // it. The returned auto_ptr will not be null. The argument must not
  // be null.
  std::auto_ptr<TFile> 
  makeTFile(const char* filename) 
  {
    std::auto_ptr<TFile> result(TFile::Open(filename));
    if (!result.get()) {
      throw cms::Exception("FileNotFound","RootFile::RootFile()")
	<< "File " << filename << " was not found or could not be opened.\n";
    }
    return result;
  }

}



class ProvenanceDumper : boost::noncopyable
{
public:
  // It is illegal to call this constructor with a null pointer; a
  // legal C-style string is required.
  explicit ProvenanceDumper(const char* filename);

  // Write the provenenace information to the given stream.
  void dump(std::ostream& os);
  void printErrors(std::ostream& os);
  int exitCode() const;

private:
  std::string              filename_;
  boost::scoped_ptr<TFile> inputFile_;
  int                      exitCode_;
  std::stringstream        errorLog_;
  int                      errorCount_;
  edm::ProductRegistry     reg_;
  edm::ProcessHistoryVector phv_;
  ParameterSetMap          psm_;
  HistoryNode              historyGraph_;

  void work_();
  void dumpProcessHistory_(TTree& history);
  void dumpEventFilteringParameterSets_(TTree& history);
  void dumpParameterSetForID_(edm::ParameterSetID const& id);
};

ProvenanceDumper::ProvenanceDumper(const char* filename) :
  filename_( filename),
  inputFile_(makeTFile(filename)),
  exitCode_(0),
  errorLog_(),
  errorCount_(0)
{ 
}

void
ProvenanceDumper::dump(std::ostream& os)
{
  work_();
}

void
ProvenanceDumper::printErrors(std::ostream& os)
{
  if (errorCount_ > 0) os << errorLog_.str() << std::endl;
}

int
ProvenanceDumper::exitCode() const
{
  return exitCode_;
}

void
ProvenanceDumper::dumpEventFilteringParameterSets_(TTree& history)
{
  // This is how one reads a TTree ...
  edm::History h; 
  edm::History* ph = &h;

  history.SetBranchAddress(edm::poolNames::eventHistoryBranchName().c_str(), &ph);
  if (history.GetEntry(0) <= 0)
    {
      std::cout << "No event filtering information is available; the event history tree has no entries\n";
    }
  else
    {
      edm::EventSelectionIDVector const& ids = h.eventSelectionIDs();
      edm::EventSelectionIDVector::size_type num_ids = ids.size();
      if ( num_ids == 0)
	{
	  std::cout << "No event filtering information is available.\n";
	  std::cout << "------------------------------\n";
	}
      else
	{
	  std::cout << "Event filtering information for "
		    << num_ids
		    << " processing steps is available.\n"
		    << "The ParameterSets will be printed out, "
		    << "with the oldest printed first.\n";
	  for (edm::EventSelectionIDVector::size_type i = 0; i != num_ids; ++i)
	    dumpParameterSetForID_(ids[i]);
	}
    }
}

void
ProvenanceDumper::dumpParameterSetForID_(edm::ParameterSetID const& id)
{
  std::cout << "ParameterSetID: " << id << '\n';
  if (id.isValid())
    {
      ParameterSetMap::const_iterator i = psm_.find(id);
      if (i == psm_.end())
 	{
 	  std::cout << "We are unable to find the corresponding ParameterSet\n";
 	  edm::ParameterSet empty;
 	  if (id == empty.id())
 	    {
 	      std::cout << "But it would have been empty anyway\n";
 	    }
 	}
      else
 	{
	  edm::ParameterSet ps(i->second.pset_);
 	  std::cout << ps << '\n';
 	}      
    }
  else
    {
      std::cout << "This ID is not valid\n";
    }
  std::cout << "     -------------------------\n";
}

void
ProvenanceDumper::dumpProcessHistory_(TTree& history)
{
  dumpEventFilteringParameterSets_(history);
  std::cout << "Processing History:"<<std::endl;
  if (1 == phv_.size()) {
    std::cout << *phv_.begin();
    historyGraph_.addChild(HistoryNode(*(phv_.begin()->begin()), 1));
  } else {
    bool multipleHistories =false;
    std::map<edm::ProcessConfigurationID, unsigned int> simpleIDs;
    for (edm::ProcessHistoryVector::const_iterator it = phv_.begin(), itEnd = phv_.end();
	 it != itEnd;
	 ++it) {
      //loop over the history entries looking for matches
      HistoryNode* parent = &historyGraph_;
      for (edm::ProcessHistory::const_iterator itH = it->begin(), e = it->end();
	   itH != e;
	   ++itH) {
	if (parent->size() == 0) {
	  unsigned int id = simpleIDs[itH->id()];
	  if (0 == id) {
	    id = 1;
	    simpleIDs[itH->id()] = id;
	  }
	  parent->addChild(HistoryNode(*itH,id));
	  parent = parent->lastChildAddress();
	} else {
	  //see if this is unique
	  bool unique = true;
	  for (HistoryNode::iterator itChild = parent->begin(), itChildEnd = parent->end();
	       itChild != itChildEnd;
	       ++itChild) {
	    if (itChild->configurationID() == itH->id()) {
	      unique = false;
	      parent = &(*itChild);
	      break;
	    }
	  }
	  if (unique) {
	    multipleHistories = true;
	    simpleIDs[itH->id()]=parent->size()+1;
	    parent->addChild(HistoryNode(*itH,simpleIDs[itH->id()]));
	    parent = parent->lastChildAddress();
	  }
	}
      }
    }
    historyGraph_.printHistory();
  }
}

void
ProvenanceDumper::work_() {

  std::auto_ptr<TFile> f = makeTFile(filename_.c_str());

  TTree* history = dynamic_cast<TTree*>(f->Get(edm::poolNames::eventHistoryTreeName().c_str()));
  assert(0!=history);

  TTree* meta = dynamic_cast<TTree*>(f->Get(edm::poolNames::metaDataTreeName().c_str()));
  assert(0!=meta);

  edm::ProductRegistry* pReg=&reg_;
  meta->SetBranchAddress(edm::poolNames::productDescriptionBranchName().c_str(),&pReg);

  ParameterSetMap* pPsm =&psm_;
  meta->SetBranchAddress(edm::poolNames::parameterSetMapBranchName().c_str(),&pPsm);

  edm::ProcessHistoryVector* pPhv=&phv_;
  if (meta->FindBranch(edm::poolNames::processHistoryBranchName().c_str()) != 0) {
    meta->SetBranchAddress(edm::poolNames::processHistoryBranchName().c_str(),&pPhv);
  }

  edm::ProcessHistoryMap phm;
  edm::ProcessHistoryMap* pPhm=&phm;
  if (meta->FindBranch(edm::poolNames::processHistoryMapBranchName().c_str()) != 0) {
    meta->SetBranchAddress(edm::poolNames::processHistoryMapBranchName().c_str(),&pPhm);
  }

  meta->GetEntry(0);
  assert(0!=pReg);
  pReg->setFrozen();

  edm::pset::Registry& psetRegistry = *edm::pset::Registry::instance();
  for (ParameterSetMap::const_iterator i = psm_.begin(), iEnd = psm_.end(); i != iEnd; ++i) {
    edm::ParameterSet pset(i->second.pset_);
    pset.setID(i->first);
    pset.setFullyTracked();
    psetRegistry.insertMapped(pset);
  }
 

  // backward compatibility
  for (edm::ProcessHistoryMap::const_iterator i = phm.begin(), e = phm.end(); i != e; ++i) {
    phv_.push_back(i->second);
  }

  dumpProcessHistory_(*history);
  
  std::cout <<"---------Event---------"<<std::endl;
  /*
    for (std::vector<edm::ProcessHistory>::const_iterator it = uniqueLongHistories.begin(),
    itEnd = uniqueLongHistories.end();
    it != itEnd;
    ++it) {
    //ParameterSetMap::const_iterator itpsm = psm.find(psid);
    for (edm::ProcessHistory::const_iterator itH = it->begin(), e = it->end();
    itH != e;
    ++itH) {
    std::cout << edm::ParameterSet(psm[ itH->parameterSetID() ].pset_) <<std::endl;
    }
    }
  */
  //using edm::ParameterSetID as the key does not work
  //   typedef std::map<edm::ParameterSetID,std::vector<edm::BranchDescription> > IdToBranches
  typedef std::map<std::string,std::vector<edm::BranchDescription> > IdToBranches;
  typedef std::map<std::pair<std::string,std::string>,IdToBranches> ModuleToIdBranches;
  ModuleToIdBranches moduleToIdBranches;
  //IdToBranches idToBranches;
  for (edm::ProductRegistry::ProductList::const_iterator it = 
	 reg_.productList().begin(), itEnd = reg_.productList().end();
       it != itEnd;
       ++it) {
    //force it to rebuild the branch name
    it->second.init();

    /*
      std::cout << it->second.branchName()
      << " id " << it->second.productID() << std::endl;
    */
    for (std::map<edm::ProcessConfigurationID, edm::ParameterSetID>::const_iterator
	   itId = it->second.parameterSetIDs().begin(),
	   itIdEnd = it->second.parameterSetIDs().end();
	   itId != itIdEnd;
	   ++itId) {
	 
      std::stringstream s;
      s << itId->second;
      moduleToIdBranches[std::make_pair(it->second.processName(),it->second.moduleLabel())][s.str()].push_back(it->second);
      //idToBranches[*itId].push_back(it->second);
    }
  }
  for (ModuleToIdBranches::const_iterator it = moduleToIdBranches.begin(),
	 itEnd = moduleToIdBranches.end();
       it != itEnd;
       ++it) {
    std::cout <<"Module: "<<it->first.second<<" "<<it->first.first<<std::endl;
    const IdToBranches& idToBranches = it->second;
    for (IdToBranches::const_iterator itIdBranch = idToBranches.begin(),
	   itIdBranchEnd = idToBranches.end();
	 itIdBranch != itIdBranchEnd;
	 ++itIdBranch) {
      std::cout <<" PSet id:"<<itIdBranch->first<<std::endl;
      std::cout <<" products: {"<<std::endl;
      for (std::vector<edm::BranchDescription>::const_iterator itBranch = itIdBranch->second.begin(),
	     itBranchEnd = itIdBranch->second.end();
	   itBranch != itBranchEnd;
	   ++itBranch) {
	std::cout << "  "<< itBranch->branchName()<<std::endl;
      }
      std::cout <<"}"<<std::endl;
      edm::ParameterSetID psid(itIdBranch->first);
      ParameterSetMap::const_iterator itpsm = psm_.find(psid);
      if (psm_.end() == itpsm) {
	++errorCount_;
	errorLog_ << "No ParameterSetID for " << psid << std::endl;
	exitCode_ = 1;
      } else {
	std::cout <<" parameters: "<<
	  edm::ParameterSet((*itpsm).second.pset_)<<std::endl;
      }
      std::cout << std::endl;
    }
  }
  std::cout <<"---------EventSetup---------"<<std::endl;
  historyGraph_.printEventSetupHistory(psm_, errorLog_);
  if (errorCount_ != 0) {
    exitCode_ = 1;
  }
}


int main(int argc, char* argv[])
{
  // will need boost::program_options someday
  std::string fileName;
  if(argc == 3 && std::string(argv[1]) == "--sort")
  {
    HistoryNode::sort_ = true;
    fileName = argv[2];
  }
  else if (argc == 2)
  {
     fileName = argv[1];
  }
  else
  {
    std::cerr << "Usage: " << argv[0] << " [--sort] <filename> \n";
    return 2;
  }

  ROOT::Cintex::Cintex::Enable();
  ProvenanceDumper dumper(fileName.c_str());
  int exitCode(0);
  try 
    {
      dumper.dump(std::cout);
    }
  
  catch (cms::Exception const& x) 
    {
      std::cerr << "cms::Exception caught\n";
      std::cerr << x.what() << '\n';
      exitCode = 2;
    }
  catch (std::exception& x) 
    {
      std::cerr << "std::exception caught\n";
      std::cerr << x.what() << '\n';
      exitCode = 3;
    }
  catch (...) 
    {
      std::cerr << "Unknown exception caught\n";
      exitCode = 4;
    }
  
  dumper.printErrors(std::cerr);
  return dumper.exitCode();
}
