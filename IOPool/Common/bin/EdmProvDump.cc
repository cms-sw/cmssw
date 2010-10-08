#include "TFile.h"
#include "TTree.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationRegistry.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/FillProductRegistryTransients.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"



#include <assert.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <map>
#include <vector>
#include <set>

#include "boost/scoped_ptr.hpp"

#include "boost/program_options.hpp"

typedef std::map<std::string, std::vector<edm::BranchDescription> > IdToBranches;
typedef std::map<std::pair<std::string, std::string>, IdToBranches> ModuleToIdBranches;

static std::ostream& prettyPrint(std::ostream& oStream, edm::ParameterSet const& iPSet, std::string const& iIndent, std::string const& iIndentDelta);


namespace {
typedef std::map<edm::ParameterSetID, edm::ParameterSetBlob> ParameterSetMap;

  class HistoryNode {
  public:
    HistoryNode() :
      config_(),
      simpleId_(0) {
    }

    HistoryNode(edm::ProcessConfiguration const& iConfig, unsigned int iSimpleId) :
      config_(iConfig),
      simpleId_(iSimpleId) {
    }

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
         << simpleId_ << "]  ("
         << config_.parameterSetID() << ")"
         << std::endl;
    }

    void printHistory(std::string const& iIndent = std::string("  ")) const;
    void printEventSetupHistory(ParameterSetMap const& iPSM, std::ostream& oErrorLog) const;
    void printOtherModulesHistory(ParameterSetMap const& iPSM, const ModuleToIdBranches&, std::ostream& oErrorLog) const;

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

  std::ostream& operator<<(std::ostream& os, HistoryNode const& node) {
    node.print(os);
    return os;
  }
  bool HistoryNode::sort_ = false;
}


std::ostream&
operator<<(std::ostream& os, edm::ProcessHistory& iHist) {
  std::string const indentDelta("  ");
  std::string indent = indentDelta;
  for (edm::ProcessHistory::const_iterator i = iHist.begin(), e = iHist.end();
       i != e;
       ++i) {
    os << indent
       << i->processName() << " '"
       << i->passID()      << "' '"
       << i->releaseVersion() << "' ("
       << i->parameterSetID() << ")"
       << std::endl;
    indent += indentDelta;
  }
  return os;
}

void HistoryNode::printHistory(std::string const& iIndent) const {
  std::string const indentDelta("  ");
  std::string indent = iIndent;
  for (const_iterator i = begin(), e = end();
       i != e;
       ++i) {
    std::cout << indent << *i;
    i->printHistory(indent+indentDelta);
  }
}

std::string eventSetupComponent(char const* iType, std::string const& iCompName, edm::ParameterSet const& iProcessConfig, std::string const& iProcessName) {
  std::ostringstream result;
  edm::ParameterSet const& pset = iProcessConfig.getParameter<edm::ParameterSet>(iCompName);
  std::string name(pset.getParameter<std::string>("@module_label"));
  if(0 == name.size()) {
    name = pset.getParameter<std::string>("@module_type");
  }

  result << iType << ": " << name << " " << iProcessName << "\n"
         << " parameters: ";
  prettyPrint(result,pset," "," ");
  return result.str();
}

void HistoryNode::printEventSetupHistory(ParameterSetMap const& iPSM, std::ostream& oErrorLog) const {
  for (const_iterator itH = begin(), e = end();
       itH != e;
       ++itH) {
    //Get ParameterSet for process
    ParameterSetMap::const_iterator itFind = iPSM.find(itH->parameterSetID());
    if(itFind == iPSM.end()){
      oErrorLog << "No ParameterSetID for " << itH->parameterSetID() << std::endl;
    } else {
      edm::ParameterSet processConfig(itFind->second.pset());
      std::vector<std::string> sourceStrings, moduleStrings;
      //get the sources
      std::vector<std::string> sources = processConfig.getParameter<std::vector<std::string> >("@all_essources");
      for(std::vector<std::string>::iterator itM = sources.begin(); itM != sources.end(); ++itM) {
        sourceStrings.push_back(eventSetupComponent("ESSource",
                                 *itM,
                                 processConfig,
                                 itH->processName()));
      }
      //get the modules
      std::vector<std::string> modules = processConfig.getParameter<std::vector<std::string> >("@all_esmodules");
      for(std::vector<std::string>::iterator itM = modules.begin(); itM != modules.end(); ++itM) {
        moduleStrings.push_back(eventSetupComponent("ESModule",
                                 *itM,
                                 processConfig,
                                 itH->processName()));
      }
      if(sort_) {
        std::sort(sourceStrings.begin(), sourceStrings.end());
        std::sort(moduleStrings.begin(), moduleStrings.end());
      }
      std::copy(sourceStrings.begin(), sourceStrings.end(),
                std::ostream_iterator<std::string>(std::cout, "\n"));
      std::copy(moduleStrings.begin(), moduleStrings.end(),
                std::ostream_iterator<std::string>(std::cout, "\n"));

    }
    itH->printEventSetupHistory(iPSM, oErrorLog);
  }
}

std::string nonProducerComponent(std::string const& iCompName, edm::ParameterSet const& iProcessConfig, std::string const& iProcessName) {
  std::ostringstream result;
  edm::ParameterSet const& pset = iProcessConfig.getParameter<edm::ParameterSet>(iCompName);
  std::string label(pset.getParameter<std::string>("@module_label"));
  
  result <<"Module: " << label << " " << iProcessName << "\n"
  << " parameters: ";
  prettyPrint(result,pset," "," ");
  return result.str();
}

void HistoryNode::printOtherModulesHistory(ParameterSetMap const& iPSM,
                                          const ModuleToIdBranches& iModules,
                                          std::ostream& oErrorLog) const {
  for (const_iterator itH = begin(), e = end();
       itH != e;
       ++itH) {
    //Get ParameterSet for process
    ParameterSetMap::const_iterator itFind = iPSM.find(itH->parameterSetID());
    if(itFind == iPSM.end()){
      oErrorLog << "No ParameterSetID for " << itH->parameterSetID() << std::endl;
    } else {
      edm::ParameterSet processConfig(itFind->second.pset());
      std::vector<std::string> moduleStrings;
      //get all modules
      std::vector<std::string> modules = processConfig.getParameter<std::vector<std::string> >("@all_modules");
      for(std::vector<std::string>::iterator itM = modules.begin(); itM != modules.end(); ++itM) {
        //if we didn't already handle this from the branches
        if( iModules.end() == iModules.find(std::make_pair(itH->processName(),*itM))) {
          
          moduleStrings.push_back(nonProducerComponent(
                                                       *itM,
                                                       processConfig,
                                                       itH->processName()));
        }
      }
      if(sort_) {
        std::sort(moduleStrings.begin(), moduleStrings.end());
      }
      std::copy(moduleStrings.begin(), moduleStrings.end(),
                std::ostream_iterator<std::string>(std::cout, "\n"));
      
    }
    itH->printOtherModulesHistory(iPSM, iModules, oErrorLog);
  }
}



namespace {

  // Open the input file, returning the TFile object that represents
  // it. The returned auto_ptr will not be null. The argument must not
  // be null.
  std::auto_ptr<TFile>
  makeTFile(char const* filename) {
    std::auto_ptr<TFile> result(TFile::Open(filename));
    if (!result.get()) {
      throw cms::Exception("FileNotFound", "RootFile::RootFile()")
          << "File " << filename << " was not found or could not be opened.\n";
    }
    return result;
  }

}


static std::ostream & prettyPrint(std::ostream & os, edm::ParameterSetEntry const& psetEntry, std::string const& iIndent, std::string const& iIndentDelta) {
  char const* trackiness = (psetEntry.isTracked()?"tracked":"untracked");
  os << "PSet " << trackiness << " = (";
  prettyPrint(os,psetEntry.pset(),iIndent+iIndentDelta,iIndentDelta);
  os << ")";
  return os;
}

static std::ostream & prettyPrint(std::ostream & os, edm::VParameterSetEntry const& vpsetEntry, std::string const& iIndent, std::string const& iIndentDelta) {
  std::vector<edm::ParameterSet> const& vps = vpsetEntry.vpset();
  os << "VPSet " << (vpsetEntry.isTracked() ? "tracked" : "untracked") << " = ({" << std::endl;
  std::string newIndent = iIndent+iIndentDelta;
  std::string start;
  std::string const between(",\n");
  for(std::vector<edm::ParameterSet>::const_iterator i = vps.begin(), e = vps.end(); i != e; ++i) {
    os << start << newIndent;
    prettyPrint(os, *i, newIndent, iIndentDelta);
    start = between;
  }
  if (!vps.empty()) {
    os << std::endl;
  }
  os << iIndent<< "})";
  return os;
}


static std::ostream& prettyPrint(std::ostream& oStream, edm::ParameterSet const& iPSet, std::string const& iIndent, std::string const& iIndentDelta) {
  std::string newIndent = iIndent+iIndentDelta;

  oStream << "{" << std::endl;
  for(edm::ParameterSet::table::const_iterator i = iPSet.tbl().begin(), e = iPSet.tbl().end(); i != e; ++i) {
    // indent a bit
    oStream << newIndent<< i->first << ": " << i->second << std::endl;
  }
  for(edm::ParameterSet::psettable::const_iterator i = iPSet.psetTable().begin(), e = iPSet.psetTable().end(); i != e; ++i) {
    // indent a bit
    edm::ParameterSetEntry const& pe = i->second;
    oStream << newIndent << i->first << ": ";
    prettyPrint(oStream,pe, iIndent,iIndentDelta);
    oStream<<  std::endl;
  }
  for(edm::ParameterSet::vpsettable::const_iterator i = iPSet.vpsetTable().begin(), e = iPSet.vpsetTable().end(); i != e; ++i) {
    // indent a bit
    edm::VParameterSetEntry const& pe = i->second;
    oStream << newIndent << i->first << ": ";
    prettyPrint(oStream,pe,newIndent,iIndentDelta);
    oStream<<  std::endl;
  }
  oStream << iIndent<< "}";

  return oStream;
}


class ProvenanceDumper : private boost::noncopyable {
public:
  // It is illegal to call this constructor with a null pointer; a
  // legal C-style string is required.
  ProvenanceDumper(char const* filename,bool showDependencies, bool excludeESModules, bool showAllModules);

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
  edm::ProcessConfigurationVector phc_;
  edm::ProcessHistoryVector phv_;
  ParameterSetMap          psm_;
  HistoryNode              historyGraph_;
  bool                     showDependencies_;
  bool                     excludeESModules_;
  bool                     showOtherModules_;

  void work_();
  void dumpProcessHistory_();
  void dumpEventFilteringParameterSets_(TFile * file);
  void dumpEventFilteringParameterSets(edm::EventSelectionIDVector const& ids);
  void dumpParameterSetForID_(edm::ParameterSetID const& id);
};

ProvenanceDumper::ProvenanceDumper(char const* filename, bool showDependencies, bool excludeESModules, bool showOtherModules) :
  filename_(filename),
  inputFile_(makeTFile(filename)),
  exitCode_(0),
  errorLog_(),
  errorCount_(0),
  showDependencies_(showDependencies),
  excludeESModules_(excludeESModules),
  showOtherModules_(showOtherModules)
{
}

void
ProvenanceDumper::dump(std::ostream& os) {
  work_();
}

void
ProvenanceDumper::printErrors(std::ostream& os) {
  if (errorCount_ > 0) os << errorLog_.str() << std::endl;
}

int
ProvenanceDumper::exitCode() const {
  return exitCode_;
}

void
ProvenanceDumper::dumpEventFilteringParameterSets(edm::EventSelectionIDVector const& ids) {
  edm::EventSelectionIDVector::size_type num_ids = ids.size();
  if (num_ids == 0) {
    std::cout << "No event filtering information is available.\n";
    std::cout << "------------------------------\n";
  } else {
    std::cout << "Event filtering information for "
              << num_ids
              << " processing steps is available.\n"
              << "The ParameterSets will be printed out, "
              << "with the oldest printed first.\n";
    for (edm::EventSelectionIDVector::size_type i = 0; i != num_ids; ++i) {
      dumpParameterSetForID_(ids[i]);
    }
  }
}

void
ProvenanceDumper::dumpEventFilteringParameterSets_(TFile* file) {

  TTree* history = dynamic_cast<TTree*>(file->Get(edm::poolNames::eventHistoryTreeName().c_str()));
  if (history != 0) {
    edm::History h;
    edm::History* ph = &h;

    history->SetBranchAddress(edm::poolNames::eventHistoryBranchName().c_str(), &ph);
    if (history->GetEntry(0) <= 0) {
      std::cout << "No event filtering information is available; the event history tree has no entries\n";
    } else {
      dumpEventFilteringParameterSets(h.eventSelectionIDs());
    }
  } else {
    TTree* events = dynamic_cast<TTree*>(file->Get(edm::poolNames::eventTreeName().c_str()));
    assert (events != 0);
    TBranch* eventSelectionsBranch = events->GetBranch(edm::poolNames::eventSelectionsBranchName().c_str());
    assert (eventSelectionsBranch != 0);
    edm::EventSelectionIDVector ids;
    edm::EventSelectionIDVector* pids = &ids;
    eventSelectionsBranch->SetAddress(&pids);
    if (eventSelectionsBranch->GetEntry(0) <= 0) {
      std::cout << "No event filtering information is available; the event selections branch has no entries\n";
    } else {
      dumpEventFilteringParameterSets(ids);
    }
  }
}

void
ProvenanceDumper::dumpParameterSetForID_(edm::ParameterSetID const& id) {
  std::cout << "ParameterSetID: " << id << '\n';
  if (id.isValid()) {
    ParameterSetMap::const_iterator i = psm_.find(id);
    if (i == psm_.end()) {
      std::cout << "We are unable to find the corresponding ParameterSet\n";
      edm::ParameterSet empty;
      if (id == empty.id()) {
        std::cout << "But it would have been empty anyway\n";
      }
    } else {
      edm::ParameterSet ps(i->second.pset());
      prettyPrint(std::cout, ps," "," ");
      std::cout<< '\n';
    }
  } else {
    std::cout << "This ID is not valid\n";
  }
  std::cout << "     -------------------------\n";
}

void
ProvenanceDumper::dumpProcessHistory_() {
  std::cout << "Processing History:" << std::endl;
  if (1 == phv_.size()) {
    std::cout << *phv_.begin();
    historyGraph_.addChild(HistoryNode(*(phv_.begin()->begin()), 1));
  } else {
    bool multipleHistories = false;
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
          parent->addChild(HistoryNode(*itH, id));
          parent = parent->lastChildAddress();
        } else {
          //see if this is unique
          bool isUnique = true;
          for (HistoryNode::iterator itChild = parent->begin(), itChildEnd = parent->end();
               itChild != itChildEnd;
               ++itChild) {
            if (itChild->configurationID() == itH->id()) {
              isUnique = false;
              parent = &(*itChild);
              break;
            }
          }
          if (isUnique) {
            multipleHistories = true;
            simpleIDs[itH->id()] = parent->size() + 1;
            parent->addChild(HistoryNode(*itH, simpleIDs[itH->id()]));
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

  TTree* meta = dynamic_cast<TTree*>(f->Get(edm::poolNames::metaDataTreeName().c_str()));
  assert(0 != meta);

  edm::ProductRegistry* pReg = &reg_;
  meta->SetBranchAddress(edm::poolNames::productDescriptionBranchName().c_str(), &pReg);

  ParameterSetMap* pPsm = &psm_;
  if(meta->FindBranch(edm::poolNames::parameterSetMapBranchName().c_str()) != 0) {
    meta->SetBranchAddress(edm::poolNames::parameterSetMapBranchName().c_str(), &pPsm);
  } else {
    TTree* psetTree = dynamic_cast<TTree *>(f->Get(edm::poolNames::parameterSetsTreeName().c_str()));
    assert(0!=psetTree);
    typedef std::pair<edm::ParameterSetID, edm::ParameterSetBlob> IdToBlobs;
    IdToBlobs idToBlob;
    IdToBlobs* pIdToBlob = &idToBlob;
    psetTree->SetBranchAddress(edm::poolNames::idToParameterSetBlobsBranchName().c_str(), &pIdToBlob);
    for(long long i = 0; i != psetTree->GetEntries(); ++i) {
      psetTree->GetEntry(i);
      psm_.insert(idToBlob);
    }
  }
  edm::ProcessConfigurationVector* pPhc = &phc_;
  if (meta->FindBranch(edm::poolNames::processConfigurationBranchName().c_str()) != 0) {
    meta->SetBranchAddress(edm::poolNames::processConfigurationBranchName().c_str(), &pPhc);
  }

  edm::ProcessHistoryVector* pPhv = &phv_;
  if (meta->FindBranch(edm::poolNames::processHistoryBranchName().c_str()) != 0) {
    meta->SetBranchAddress(edm::poolNames::processHistoryBranchName().c_str(), &pPhv);
  }

  edm::ProcessHistoryMap phm;
  edm::ProcessHistoryMap* pPhm = &phm;
  if (meta->FindBranch(edm::poolNames::processHistoryMapBranchName().c_str()) != 0) {
    meta->SetBranchAddress(edm::poolNames::processHistoryMapBranchName().c_str(), &pPhm);
  }

  if (meta->FindBranch(edm::poolNames::moduleDescriptionMapBranchName().c_str()) != 0) {
    if (meta->GetBranch(edm::poolNames::moduleDescriptionMapBranchName().c_str())->GetSplitLevel() != 0) {
      meta->SetBranchStatus((edm::poolNames::moduleDescriptionMapBranchName() + ".*").c_str(), 0);
    } else {
      meta->SetBranchStatus(edm::poolNames::moduleDescriptionMapBranchName().c_str(), 0);
    }
  }

  meta->GetEntry(0);
  assert(0 != pReg);

  edm::pset::Registry& psetRegistry = *edm::pset::Registry::instance();
  for (ParameterSetMap::const_iterator i = psm_.begin(), iEnd = psm_.end(); i != iEnd; ++i) {
    edm::ParameterSet pset(i->second.pset());
    pset.setID(i->first);
    psetRegistry.insertMapped(pset);
  }


  // backward compatibility
  if (!phm.empty()) {
    for (edm::ProcessHistoryMap::const_iterator i = phm.begin(), e = phm.end(); i != e; ++i) {
      phv_.push_back(i->second);
      for (edm::ProcessConfigurationVector::const_iterator j = i->second.begin(), f = i->second.end(); j != f; ++j) {
        phc_.push_back(*j);
      }
    }
    edm::sort_all(phc_);
    phc_.erase(std::unique(phc_.begin(), phc_.end()), phc_.end());
  }

  fillProductRegistryTransients(phc_, reg_, true);

  //Prepare the parentage information if requested
  std::map<edm::BranchID, std::set<edm::ParentageID> > perProductParentage;

  if(showDependencies_){
    TTree* parentageTree = dynamic_cast<TTree*>(f->Get(edm::poolNames::parentageTreeName().c_str()));
    if(0==parentageTree) {
      std::cerr << "no Parentage tree available so can not show dependencies/n";
      showDependencies_ = false;
    } else {
      edm::Parentage parentageBuffer;
      edm::Parentage *pParentageBuffer = &parentageBuffer;
      parentageTree->SetBranchAddress(edm::poolNames::parentageBranchName().c_str(), &pParentageBuffer);

      edm::ParentageRegistry& registry = *edm::ParentageRegistry::instance();

      for(Long64_t i = 0, numEntries = parentageTree->GetEntries(); i < numEntries; ++i) {
        parentageTree->GetEntry(i);
        registry.insertMapped(parentageBuffer);
      }
      parentageTree->SetBranchAddress(edm::poolNames::parentageBranchName().c_str(), 0);

      TTree* eventMetaTree = dynamic_cast<TTree*>(f->Get(edm::BranchTypeToMetaDataTreeName(edm::InEvent).c_str()));
      if(0==eventMetaTree) {
        eventMetaTree = dynamic_cast<TTree*>(f->Get(edm::BranchTypeToProductTreeName(edm::InEvent).c_str()));
      }
      if(0==eventMetaTree) {
        std::cerr << "no '" << edm::BranchTypeToProductTreeName(edm::InEvent)<< "' Tree in file so can not show dependencies\n";
        showDependencies_ = false;
      } else {
        TBranch* productProvBranch = eventMetaTree->GetBranch(edm::BranchTypeToBranchEntryInfoBranchName(edm::InEvent).c_str());

        std::vector<edm::ProductProvenance> info;
        std::vector<edm::ProductProvenance>* pInfo = &info;
        productProvBranch->SetAddress(&pInfo);
        for(Long64_t i = 0, numEntries = eventMetaTree->GetEntries(); i < numEntries; ++i) {
          productProvBranch->GetEntry(i);
          for(std::vector<edm::ProductProvenance>::const_iterator it = info.begin(), itEnd = info.end();
              it != itEnd; ++it) {
            perProductParentage[it->branchID()].insert(it->parentageID());
          }
        }
      }
    }
  }


  dumpEventFilteringParameterSets_(f.get());

  dumpProcessHistory_();

  std::cout << "---------Data---------" << std::endl;

  //using edm::ParameterSetID as the key does not work
  //   typedef std::map<edm::ParameterSetID, std::vector<edm::BranchDescription> > IdToBranches
  ModuleToIdBranches moduleToIdBranches;
  //IdToBranches idToBranches;

  std::map<edm::BranchID, std::string> branchIDToBranchName;

  for (edm::ProductRegistry::ProductList::const_iterator it =
         reg_.productList().begin(), itEnd = reg_.productList().end();
       it != itEnd;
       ++it) {
    //force it to rebuild the branch name
    it->second.init();

    if(showDependencies_) {
      branchIDToBranchName[it->second.branchID()]=it->second.branchName();
    }
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
      moduleToIdBranches[std::make_pair(it->second.processName(), it->second.moduleLabel())][s.str()].push_back(it->second);
      //idToBranches[*itId].push_back(it->second);
    }
  }
  for (ModuleToIdBranches::const_iterator it = moduleToIdBranches.begin(),
         itEnd = moduleToIdBranches.end();
       it != itEnd;
       ++it) {
    std::cout << "Module: " << it->first.second << " " << it->first.first << std::endl;
    IdToBranches const& idToBranches = it->second;
    for (IdToBranches::const_iterator itIdBranch = idToBranches.begin(),
           itIdBranchEnd = idToBranches.end();
         itIdBranch != itIdBranchEnd;
         ++itIdBranch) {
      std::cout << " PSet id:" << itIdBranch->first << std::endl;
      std::cout << " products: {" << std::endl;
      std::set<edm::BranchID> branchIDs;
      for (std::vector<edm::BranchDescription>::const_iterator itBranch = itIdBranch->second.begin(),
             itBranchEnd = itIdBranch->second.end();
           itBranch != itBranchEnd;
           ++itBranch) {
        std::cout << "  " << itBranch->branchName() << std::endl;
        branchIDs.insert(itBranch->branchID());
      }
      std::cout << " }" << std::endl;
      edm::ParameterSetID psid(itIdBranch->first);
      ParameterSetMap::const_iterator itpsm = psm_.find(psid);
      if (psm_.end() == itpsm) {
        ++errorCount_;
        errorLog_ << "No ParameterSetID for " << psid << std::endl;
        exitCode_ = 1;
      } else {
        std::cout << " parameters: ";
        prettyPrint(std::cout, edm::ParameterSet((*itpsm).second.pset()), " ", " ");
        std::cout << std::endl;
      }
      if(showDependencies_) {
        edm::ParentageRegistry& registry = *edm::ParentageRegistry::instance();

        std::cout << " dependencies: {" << std::endl;
        std::set<edm::ParentageID> parentageIDs;
        for(std::set<edm::BranchID>::const_iterator itBranch = branchIDs.begin(), itBranchEnd=branchIDs.end();
            itBranch != itBranchEnd;
            ++itBranch) {
          std::set<edm::ParentageID> const& temp = perProductParentage[*itBranch];
          parentageIDs.insert(temp.begin(),temp.end());
        }
        for(std::set<edm::ParentageID>::const_iterator itParentID = parentageIDs.begin(), itEndParentID=parentageIDs.end();
            itParentID != itEndParentID;
            ++itParentID) {
          edm::Parentage const* parentage = registry.getMapped(*itParentID);
          if(0!=parentage) {
            for(std::vector<edm::BranchID>::const_iterator itBranch=parentage->parents().begin(), itEndBranch=parentage->parents().end();
                itBranch != itEndBranch;
                ++itBranch) {
              std::cout << "  " << branchIDToBranchName[*itBranch] << std::endl;
            }
          } else {
            std::cout << "  ERROR:parentage info not in registry ParentageID=" << *itParentID << std::endl;
          }
        }
        if(parentageIDs.empty()) {
          std::cout << "  no dependencies recorded (event may not contain data from this module)" << std::endl;
        }
        std::cout << " }" << std::endl;
      }
      std::cout << std::endl;
    }
  }
  if(showOtherModules_) {
    std::cout << "---------Other Modules---------" << std::endl;
    historyGraph_.printOtherModulesHistory(psm_,moduleToIdBranches, errorLog_);
  }
  
  if(!excludeESModules_) {
    std::cout << "---------EventSetup---------" << std::endl;
    historyGraph_.printEventSetupHistory(psm_, errorLog_);
  }
  if (errorCount_ != 0) {
    exitCode_ = 1;
  }
}


static char const* const kSortOpt = "sort";
static char const* const kSortCommandOpt = "sort,s";
static char const* const kDependenciesOpt = "dependencies";
static char const* const kDependenciesCommandOpt = "dependencies,d";
static char const* const kExcludeESModulesOpt = "excludeESModules";
static char const* const kExcludeESModulesCommandOpt = "excludeESModules,e";
static char const* const kShowAllModulesOpt = "showAllModules";
static char const* const kShowAllModulesCommandOpt = "showAllModules,a";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";
static char const* const kFileNameOpt ="input-file";
static char const* const kFileNameCommandOpt ="input-file";


int main(int argc, char* argv[]) {
  using namespace boost::program_options;


  std::string descString(argv[0]);
  descString += " [options] <filename>";
  descString += "\nAllowed options";
  options_description desc(descString);
  desc.add_options()
  (kHelpCommandOpt, "show help message")
  (kSortCommandOpt
   , "alphabetially sort EventSetup components")
  (kDependenciesCommandOpt
   , "print what data each EDProducer is dependent upon")
  (kExcludeESModulesCommandOpt
   , "do not print ES module information")
  (kShowAllModulesCommandOpt
   , "show all modules (not just those that created data in the file)")
  ;
  //we don't want users to see these in the help messages since this
  // name only exists since the parser needs it
  options_description hidden;
  hidden.add_options()(kFileNameOpt,value<std::string>(),"file name");

  //full list of options for the parser
  options_description cmdline_options;
  cmdline_options.add(desc).add(hidden);

  positional_options_description p;
  p.add(kFileNameOpt,-1);


  variables_map vm;
  try {
    store(command_line_parser(argc,argv).options(cmdline_options).positional(p).run(),vm);
    notify(vm);
  } catch(error const& iException) {
    std::cerr << iException.what();
    return 1;
  }

  if(vm.count(kHelpOpt)) {
    std::cout << desc << std::endl;
    return 0;
  }

  if(vm.count(kSortOpt) ) {
    HistoryNode::sort_ = true;
  }

  bool showDependencies = false;
  if(vm.count(kDependenciesOpt)) {
    showDependencies=true;
  }

  bool excludeESModules = false;
  if(vm.count(kExcludeESModulesOpt)) {
    excludeESModules=true;
  }
  
  bool showAllModules = false;
  if(vm.count(kShowAllModulesOpt)) {
    showAllModules=true;
  }
  
  std::string fileName;
  if(vm.count(kFileNameOpt)) {
    fileName = vm[kFileNameOpt].as<std::string>();
  } else {
    std::cout << "Data file not specified." << std::endl;
    std::cout << desc << std::endl;
    return 2;
  }


  try {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  } catch(std::exception& e) {
    std::cout << "exception caught in "
    << "EdmProvDump while configuring the PluginManager"
    << '\n'
    << e.what();
    return 1;
  }

  try {
    // The InitRootHandler service will enable Cintex and load DataFormats/StdDictionaries.
    // We do not enable the autoloader because we do not need any other dictionaries dynamically loaded.
    std::string config =
    "import FWCore.ParameterSet.Config as cms\n"
    "process = cms.Process('edmProvDump')\n"
     "process.InitRootHandlers = cms.Service('InitRootHandlers', AutoLibraryLoader = cms.untracked.bool(False))\n";

    //create the services
    edm::ServiceToken tempToken = edm::ServiceRegistry::createServicesFromConfig(config);

    //make the services available
    edm::ServiceRegistry::Operate operate(tempToken);
  } catch(cms::Exception& e) {
    std::cout << "cms::Exception caught in "
    << "EdmProvDump while setting up the ServiceRegistry"
    << '\n'
    << e.what();
    return 1;
  }

  ProvenanceDumper dumper(fileName.c_str(),showDependencies,excludeESModules,showAllModules);
  int exitCode(0);
  try {
    dumper.dump(std::cout);
  }
  catch (cms::Exception const& x) {
    std::cerr << "cms::Exception caught\n";
    std::cerr << x.what() << '\n';
    exitCode = 2;
  }
  catch (std::exception& x) {
    std::cerr << "std::exception caught\n";
    std::cerr << x.what() << '\n';
    exitCode = 3;
  }
  catch (...) {
    std::cerr << "Unknown exception caught\n";
    exitCode = 4;
  }

  dumper.printErrors(std::cerr);
  return dumper.exitCode();
}
