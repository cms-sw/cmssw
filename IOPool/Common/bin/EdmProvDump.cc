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
#include "DataFormats/Provenance/interface/StoredProductProvenance.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/FillProductRegistryTransients.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Services/src/SiteLocalConfigService.h"

#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Cintex/Cintex.h"
#include "TError.h"
#include "TFile.h"
#include "TTree.h"

#include "boost/program_options.hpp"

#include <assert.h>
#include <iostream>
#include <memory>
#include <map>
#include <set>
#include <sstream>
#include <vector>

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
    void printEventSetupHistory(ParameterSetMap const& iPSM,
                                std::vector<std::string> const& iFindMatch,
                                std::ostream& oErrorLog) const;
    void printOtherModulesHistory(ParameterSetMap const& iPSM,
                                  ModuleToIdBranches const&,
                                  std::vector<std::string> const& iFindMatch,
                                  std::ostream& oErrorLog) const;
    void printTopLevelPSetsHistory(ParameterSetMap const& iPSM,
                                   std::vector<std::string> const& iFindMatch,
                                   std::ostream& oErrorLog) const;

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
  for(edm::ProcessHistory::const_iterator i = iHist.begin(), e = iHist.end();
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
  for(const_iterator i = begin(), e = end();
       i != e;
       ++i) {
    std::cout << indent << *i;
    i->printHistory(indent+indentDelta);
  }
}

std::string eventSetupComponent(char const* iType,
                                std::string const& iCompName,
                                edm::ParameterSet const& iProcessConfig,
                                std::string const& iProcessName) {
  std::ostringstream result;
  edm::ParameterSet const& pset = iProcessConfig.getParameterSet(iCompName);
  std::string name(pset.getParameter<std::string>("@module_label"));
  if(0 == name.size()) {
    name = pset.getParameter<std::string>("@module_type");
  }

  result << iType << ": " << name << " " << iProcessName << "\n"
         << " parameters: ";
  prettyPrint(result, pset, " ", " ");
  return result.str();
}

void HistoryNode::printEventSetupHistory(ParameterSetMap const& iPSM,
                                         std::vector<std::string> const& iFindMatch,
                                         std::ostream& oErrorLog) const {
  for(const_iterator itH = begin(), e = end();
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
        std::string retValue = eventSetupComponent("ESSource",
                                                   *itM,
                                                   processConfig,
                                                   itH->processName());
        bool foundMatch = true;
        if(!iFindMatch.empty()) {
          for (auto const& stringToFind : iFindMatch) {
            if (retValue.find(stringToFind) == std::string::npos) {
              foundMatch = false;
              break;
            }
          }
        }
        if (foundMatch) {
          sourceStrings.push_back(std::move(retValue));
        }
      }
      //get the modules
      std::vector<std::string> modules = processConfig.getParameter<std::vector<std::string> >("@all_esmodules");
      for(std::vector<std::string>::iterator itM = modules.begin(); itM != modules.end(); ++itM) {
        std::string retValue = eventSetupComponent("ESModule",
                                                   *itM,
                                                   processConfig,
                                                   itH->processName());
        bool foundMatch = true;
        if(!iFindMatch.empty()) {
          for (auto const& stringToFind : iFindMatch) {
            if (retValue.find(stringToFind) == std::string::npos) {
              foundMatch = false;
              break;
            }
          }
        }
        if (foundMatch) {
          moduleStrings.push_back(std::move(retValue));
        }
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
    itH->printEventSetupHistory(iPSM, iFindMatch, oErrorLog);
  }
}

std::string nonProducerComponent(std::string const& iCompName,
                                 edm::ParameterSet const& iProcessConfig,
                                 std::string const& iProcessName) {
  std::ostringstream result;
  edm::ParameterSet const& pset = iProcessConfig.getParameterSet(iCompName);
  std::string label(pset.getParameter<std::string>("@module_label"));

  result << "Module: " << label << " " << iProcessName << "\n" << " parameters: ";
  prettyPrint(result, pset, " ", " ");
  return result.str();
}

void HistoryNode::printOtherModulesHistory(ParameterSetMap const& iPSM,
                                           ModuleToIdBranches const& iModules,
                                           std::vector<std::string> const& iFindMatch,
                                           std::ostream& oErrorLog) const {
  for(const_iterator itH = begin(), e = end();
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
        if(iModules.end() == iModules.find(std::make_pair(itH->processName(), *itM))) {
          std::string retValue(nonProducerComponent(
                                                    *itM,
                                                    processConfig,
                                                    itH->processName()));
          bool foundMatch = true;
          if(!iFindMatch.empty()) {
            for (auto const& stringToFind : iFindMatch) {
              if (retValue.find(stringToFind) == std::string::npos) {
                foundMatch = false;
                break;
              }
            }
          }
          if (foundMatch) {
            moduleStrings.push_back(std::move(retValue));
          }
        }
      }
      if(sort_) {
        std::sort(moduleStrings.begin(), moduleStrings.end());
      }
      std::copy(moduleStrings.begin(), moduleStrings.end(),
                std::ostream_iterator<std::string>(std::cout, "\n"));
    }
    itH->printOtherModulesHistory(iPSM, iModules, iFindMatch, oErrorLog);
  }
}

static void appendToSet(std::set<std::string>&iSet, std::vector<std::string> const& iFrom){
  for( auto const& n: iFrom){
    iSet.insert(n);
  }
}

static std::string topLevelPSet(std::string const& iName,
                                edm::ParameterSet const& iProcessConfig,
                                std::string const& iProcessName) {
  std::ostringstream result;
  edm::ParameterSet const& pset = iProcessConfig.getParameterSet(iName);
  
  result << "PSet: " << iName << " " << iProcessName << "\n" << " parameters: ";
  prettyPrint(result, pset, " ", " ");
  return result.str();
}


void HistoryNode::printTopLevelPSetsHistory(ParameterSetMap const& iPSM,
                                            std::vector<std::string> const& iFindMatch,
                                            std::ostream& oErrorLog) const {
  for(const_iterator itH = begin(), e = end();
      itH != e;
      ++itH) {
    //Get ParameterSet for process
    ParameterSetMap::const_iterator itFind = iPSM.find(itH->parameterSetID());
    if(itFind == iPSM.end()){
      oErrorLog << "No ParameterSetID for " << itH->parameterSetID() << std::endl;
    } else {
      edm::ParameterSet processConfig(itFind->second.pset());
      //Need to get the names of PSets which are used by the framework (e.g. names of modules)
      std::set<std::string> namesToExclude;
      appendToSet(namesToExclude,processConfig.getParameter<std::vector<std::string> >("@all_modules"));
      appendToSet(namesToExclude,processConfig.getParameter<std::vector<std::string> >("@all_sources"));
      appendToSet(namesToExclude,processConfig.getParameter<std::vector<std::string> >("@all_loopers"));
      //appendToSet(namesToExclude,processConfig.getParameter<std::vector<std::string> >("@all_subprocesses"));//untracked
      appendToSet(namesToExclude,processConfig.getParameter<std::vector<std::string> >("@all_esmodules"));
      appendToSet(namesToExclude,processConfig.getParameter<std::vector<std::string> >("@all_essources"));
      appendToSet(namesToExclude,processConfig.getParameter<std::vector<std::string> >("@all_esprefers"));
      if (processConfig.existsAs<std::vector<std::string>>("all_aliases")) {
        appendToSet(namesToExclude,processConfig.getParameter<std::vector<std::string> >("@all_aliases"));
      }

      std::vector<std::string> allNames{};
      processConfig.getParameterSetNames(allNames);

      std::vector<std::string> results;
      for(auto const& name: allNames){
        if (name.size() == 0 || '@' == name[0] || namesToExclude.find(name)!=namesToExclude.end()) {
          continue;
        }
        std::string retValue = topLevelPSet(name,processConfig,itH->processName());

        bool foundMatch = true;
        if(!iFindMatch.empty()) {
          for (auto const& stringToFind : iFindMatch) {
            if (retValue.find(stringToFind) == std::string::npos) {
              foundMatch = false;
              break;
            }
          }
        }
        if (foundMatch) {
          results.push_back(std::move(retValue));
        }
      }
      if(sort_) {
        std::sort(results.begin(), results.end());
      }
      std::copy(results.begin(), results.end(),
                std::ostream_iterator<std::string>(std::cout, "\n"));
    }
    itH->printTopLevelPSetsHistory(iPSM, iFindMatch, oErrorLog);
  }
}


namespace {
  std::unique_ptr<TFile>
  makeTFileWithLookup(std::string const& filename) {
    // See if it is a logical file name.
    std::auto_ptr<edm::SiteLocalConfig> slcptr(new edm::service::SiteLocalConfigService(edm::ParameterSet()));
    boost::shared_ptr<edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig> > slc(new edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig>(slcptr));
    edm::ServiceToken slcToken = edm::ServiceRegistry::createContaining(slc);
    edm::ServiceRegistry::Operate operate(slcToken);
    std::string override;
    std::vector<std::string> fileNames;
    fileNames.push_back(filename);
    edm::InputFileCatalog catalog(fileNames, override, true);
    if(catalog.fileNames()[0] == filename) {
      throw cms::Exception("FileNotFound", "RootFile::RootFile()")
        << "File " << filename << " was not found or could not be opened.\n";
    }
    // filename is a valid LFN.
    std::unique_ptr<TFile> result(TFile::Open(catalog.fileNames()[0].c_str()));
    if(!result.get()) {
      throw cms::Exception("FileNotFound", "RootFile::RootFile()")
        << "File " << fileNames[0] << " was not found or could not be opened.\n";
    }
    return result;
  }

  // Open the input file, returning the TFile object that represents it.
  // The returned unique_ptr will not be null. The argument must not be null.
  // We first try the file name as a PFN, so that the catalog and related
  // services are not loaded unless needed.
  std::unique_ptr<TFile>
  makeTFile(std::string const& filename) {
    gErrorIgnoreLevel = kFatal;
    std::unique_ptr<TFile> result(TFile::Open(filename.c_str()));
    gErrorIgnoreLevel = kError;
    if(!result.get()) {
      // Try again with catalog.
      return makeTFileWithLookup(filename);
    }
    return result;
  }
}


static std::ostream& prettyPrint(std::ostream& os, edm::ParameterSetEntry const& psetEntry, std::string const& iIndent, std::string const& iIndentDelta) {
  char const* trackiness = (psetEntry.isTracked()?"tracked":"untracked");
  os << "PSet " << trackiness << " = (";
  prettyPrint(os, psetEntry.pset(), iIndent + iIndentDelta, iIndentDelta);
  os << ")";
  return os;
}

static std::ostream& prettyPrint(std::ostream& os, edm::VParameterSetEntry const& vpsetEntry, std::string const& iIndent, std::string const& iIndentDelta) {
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
  if(!vps.empty()) {
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
    prettyPrint(oStream, pe, iIndent, iIndentDelta);
    oStream<<  std::endl;
  }
  for(edm::ParameterSet::vpsettable::const_iterator i = iPSet.vpsetTable().begin(), e = iPSet.vpsetTable().end(); i != e; ++i) {
    // indent a bit
    edm::VParameterSetEntry const& pe = i->second;
    oStream << newIndent << i->first << ": ";
    prettyPrint(oStream, pe, newIndent, iIndentDelta);
    oStream<<  std::endl;
  }
  oStream << iIndent<< "}";

  return oStream;
}


class ProvenanceDumper {
public:
  // It is illegal to call this constructor with a null pointer; a
  // legal C-style string is required.
  ProvenanceDumper(std::string const& filename,
                   bool showDependencies,
                   bool extendedAncestors,
                   bool extendedDescendants,
                   bool excludeESModules,
                   bool showAllModules,
                   bool showTopLevelPSets,
                   std::vector<std::string> const& findMatch,
                   bool dontPrintProducts);

  ProvenanceDumper(ProvenanceDumper const&) = delete; // Disallow copying and moving
  ProvenanceDumper& operator=(ProvenanceDumper const&) = delete; // Disallow copying and moving

  // Write the provenenace information to the given stream.
  void dump();
  void printErrors(std::ostream& os);
  int exitCode() const;

private:

  void addAncestors(edm::BranchID const& branchID,
                    std::set<edm::BranchID>& ancestorBranchIDs,
                    std::ostringstream& sout,
                    std::map<edm::BranchID, std::set<edm::ParentageID> >& perProductParentage) const;

  void addDescendants(edm::BranchID const& branchID, std::set<edm::BranchID>& descendantBranchIDs,
                      std::ostringstream& sout,
                      std::map<edm::BranchID, std::set<edm::BranchID> >& parentToChildren) const;

  std::string              filename_;
  std::unique_ptr<TFile>   inputFile_;
  int                      exitCode_;
  std::stringstream        errorLog_;
  int                      errorCount_;
  edm::ProductRegistry     reg_;
  edm::ProcessConfigurationVector phc_;
  edm::ProcessHistoryVector phv_;
  ParameterSetMap          psm_;
  HistoryNode              historyGraph_;
  bool                     showDependencies_;
  bool                     extendedAncestors_;
  bool                     extendedDescendants_;
  bool                     excludeESModules_;
  bool                     showOtherModules_;
  bool                     showTopLevelPSets_;
  std::vector<std::string> findMatch_;
  bool                     dontPrintProducts_;

  void work_();
  void dumpProcessHistory_();
  void dumpEventFilteringParameterSets_(TFile * file);
  void dumpEventFilteringParameterSets(edm::EventSelectionIDVector const& ids);
  void dumpParameterSetForID_(edm::ParameterSetID const& id);
};

ProvenanceDumper::ProvenanceDumper(std::string const& filename,
                                   bool showDependencies,
                                   bool extendedAncestors,
                                   bool extendedDescendants,
                                   bool excludeESModules,
                                   bool showOtherModules,
                                   bool showTopLevelPSets,
                                   std::vector<std::string> const& findMatch,
                                   bool dontPrintProducts) :
  filename_(filename),
  inputFile_(makeTFile(filename)),
  exitCode_(0),
  errorLog_(),
  errorCount_(0),
  showDependencies_(showDependencies),
  extendedAncestors_(extendedAncestors),
  extendedDescendants_(extendedDescendants),
  excludeESModules_(excludeESModules),
  showOtherModules_(showOtherModules),
  showTopLevelPSets_(showTopLevelPSets),
  findMatch_(findMatch),
  dontPrintProducts_(dontPrintProducts) {
}

void
ProvenanceDumper::dump() {
  work_();
}

void
ProvenanceDumper::printErrors(std::ostream& os) {
  if(errorCount_ > 0) os << errorLog_.str() << std::endl;
}

int
ProvenanceDumper::exitCode() const {
  return exitCode_;
}

void
ProvenanceDumper::dumpEventFilteringParameterSets(edm::EventSelectionIDVector const& ids) {
  edm::EventSelectionIDVector::size_type num_ids = ids.size();
  if(num_ids == 0) {
    std::cout << "No event filtering information is available.\n";
    std::cout << "------------------------------\n";
  } else {
    std::cout << "Event filtering information for "
              << num_ids
              << " processing steps is available.\n"
              << "The ParameterSets will be printed out, "
              << "with the oldest printed first.\n";
    for(edm::EventSelectionIDVector::size_type i = 0; i != num_ids; ++i) {
      dumpParameterSetForID_(ids[i]);
    }
  }
}

void
ProvenanceDumper::dumpEventFilteringParameterSets_(TFile* file) {

  TTree* history = dynamic_cast<TTree*>(file->Get(edm::poolNames::eventHistoryTreeName().c_str()));
  if(history != 0) {
    edm::History h;
    edm::History* ph = &h;

    history->SetBranchAddress(edm::poolNames::eventHistoryBranchName().c_str(), &ph);
    if(history->GetEntry(0) <= 0) {
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
    if(eventSelectionsBranch->GetEntry(0) <= 0) {
      std::cout << "No event filtering information is available; the event selections branch has no entries\n";
    } else {
      dumpEventFilteringParameterSets(ids);
    }
  }
}

void
ProvenanceDumper::dumpParameterSetForID_(edm::ParameterSetID const& id) {
  std::cout << "ParameterSetID: " << id << '\n';
  if(id.isValid()) {
    ParameterSetMap::const_iterator i = psm_.find(id);
    if(i == psm_.end()) {
      std::cout << "We are unable to find the corresponding ParameterSet\n";
      edm::ParameterSet empty;
      if(id == empty.id()) {
        std::cout << "But it would have been empty anyway\n";
      }
    } else {
      edm::ParameterSet ps(i->second.pset());
      prettyPrint(std::cout, ps, " ", " ");
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
  if(1 == phv_.size()) {
    std::cout << *phv_.begin();
    historyGraph_.addChild(HistoryNode(*(phv_.begin()->begin()), 1));
  } else {
    std::map<edm::ProcessConfigurationID, unsigned int> simpleIDs;
    for(edm::ProcessHistoryVector::const_iterator it = phv_.begin(), itEnd = phv_.end();
         it != itEnd;
         ++it) {
      //loop over the history entries looking for matches
      HistoryNode* parent = &historyGraph_;
      for(edm::ProcessHistory::const_iterator itH = it->begin(), e = it->end();
           itH != e;
           ++itH) {
        if(parent->size() == 0) {
          unsigned int id = simpleIDs[itH->id()];
          if(0 == id) {
            id = 1;
            simpleIDs[itH->id()] = id;
          }
          parent->addChild(HistoryNode(*itH, id));
          parent = parent->lastChildAddress();
        } else {
          //see if this is unique
          bool isUnique = true;
          for(HistoryNode::iterator itChild = parent->begin(), itChildEnd = parent->end();
               itChild != itChildEnd;
               ++itChild) {
            if(itChild->configurationID() == itH->id()) {
              isUnique = false;
              parent = &(*itChild);
              break;
            }
          }
          if(isUnique) {
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

  TTree* meta = dynamic_cast<TTree*>(inputFile_->Get(edm::poolNames::metaDataTreeName().c_str()));
  assert(0 != meta);

  edm::ProductRegistry* pReg = &reg_;
  meta->SetBranchAddress(edm::poolNames::productDescriptionBranchName().c_str(), &pReg);

  ParameterSetMap* pPsm = &psm_;
  if(meta->FindBranch(edm::poolNames::parameterSetMapBranchName().c_str()) != 0) {
    meta->SetBranchAddress(edm::poolNames::parameterSetMapBranchName().c_str(), &pPsm);
  } else {
    TTree* psetTree = dynamic_cast<TTree *>(inputFile_->Get(edm::poolNames::parameterSetsTreeName().c_str()));
    assert(0 != psetTree);
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
  if(meta->FindBranch(edm::poolNames::processConfigurationBranchName().c_str()) != 0) {
    meta->SetBranchAddress(edm::poolNames::processConfigurationBranchName().c_str(), &pPhc);
  }

  edm::ProcessHistoryVector* pPhv = &phv_;
  if(meta->FindBranch(edm::poolNames::processHistoryBranchName().c_str()) != 0) {
    meta->SetBranchAddress(edm::poolNames::processHistoryBranchName().c_str(), &pPhv);
  }

  edm::ProcessHistoryMap phm;
  edm::ProcessHistoryMap* pPhm = &phm;
  if(meta->FindBranch(edm::poolNames::processHistoryMapBranchName().c_str()) != 0) {
    meta->SetBranchAddress(edm::poolNames::processHistoryMapBranchName().c_str(), &pPhm);
  }

  if(meta->FindBranch(edm::poolNames::moduleDescriptionMapBranchName().c_str()) != 0) {
    if(meta->GetBranch(edm::poolNames::moduleDescriptionMapBranchName().c_str())->GetSplitLevel() != 0) {
      meta->SetBranchStatus((edm::poolNames::moduleDescriptionMapBranchName() + ".*").c_str(), 0);
    } else {
      meta->SetBranchStatus(edm::poolNames::moduleDescriptionMapBranchName().c_str(), 0);
    }
  }

  meta->GetEntry(0);
  assert(0 != pReg);

  edm::pset::Registry& psetRegistry = *edm::pset::Registry::instance();
  for(ParameterSetMap::const_iterator i = psm_.begin(), iEnd = psm_.end(); i != iEnd; ++i) {
    edm::ParameterSet pset(i->second.pset());
    pset.setID(i->first);
    psetRegistry.insertMapped(pset);
  }


  // backward compatibility
  if(!phm.empty()) {
    for(edm::ProcessHistoryMap::const_iterator i = phm.begin(), e = phm.end(); i != e; ++i) {
      phv_.push_back(i->second);
      for(edm::ProcessConfigurationVector::const_iterator j = i->second.begin(), f = i->second.end(); j != f; ++j) {
        phc_.push_back(*j);
      }
    }
    edm::sort_all(phc_);
    phc_.erase(std::unique(phc_.begin(), phc_.end()), phc_.end());
  }

  fillProductRegistryTransients(phc_, reg_, true);

  //Prepare the parentage information if requested
  std::map<edm::BranchID, std::set<edm::ParentageID> > perProductParentage;

  if(showDependencies_ || extendedAncestors_ || extendedDescendants_){
    TTree* parentageTree = dynamic_cast<TTree*>(inputFile_->Get(edm::poolNames::parentageTreeName().c_str()));
    if(0 == parentageTree) {
      std::cerr << "ERROR, no Parentage tree available so can not show dependencies/n";
      showDependencies_ = false;
      extendedAncestors_ = false;
      extendedDescendants_ = false;
    } else {

      edm::ParentageRegistry& registry = *edm::ParentageRegistry::instance();

      std::vector<edm::ParentageID> orderedParentageIDs;
      orderedParentageIDs.reserve(parentageTree->GetEntries());
      for(Long64_t i = 0, numEntries = parentageTree->GetEntries(); i < numEntries; ++i) {
        edm::Parentage parentageBuffer;
        edm::Parentage *pParentageBuffer = &parentageBuffer;
        parentageTree->SetBranchAddress(edm::poolNames::parentageBranchName().c_str(), &pParentageBuffer);
        parentageTree->GetEntry(i);
        registry.insertMapped(parentageBuffer);
        orderedParentageIDs.push_back(parentageBuffer.id());
      }
      parentageTree->SetBranchAddress(edm::poolNames::parentageBranchName().c_str(), 0);

      TTree* eventMetaTree = dynamic_cast<TTree*>(inputFile_->Get(edm::BranchTypeToMetaDataTreeName(edm::InEvent).c_str()));
      if(0 == eventMetaTree) {
        eventMetaTree = dynamic_cast<TTree*>(inputFile_->Get(edm::BranchTypeToProductTreeName(edm::InEvent).c_str()));
      }
      if(0 == eventMetaTree) {
        std::cerr << "ERROR, no '" << edm::BranchTypeToProductTreeName(edm::InEvent)<< "' Tree in file so can not show dependencies\n";
        showDependencies_ = false;
        extendedAncestors_ = false;
        extendedDescendants_ = false;
      } else {
        TBranch* storedProvBranch = eventMetaTree->GetBranch(edm::BranchTypeToProductProvenanceBranchName(edm::InEvent).c_str());

        if(0!=storedProvBranch) {
          std::vector<edm::StoredProductProvenance> info;
          std::vector<edm::StoredProductProvenance>* pInfo = &info;
          storedProvBranch->SetAddress(&pInfo);
          for(Long64_t i = 0, numEntries = eventMetaTree->GetEntries(); i < numEntries; ++i) {
            storedProvBranch->GetEntry(i);
            for(std::vector<edm::StoredProductProvenance>::const_iterator it = info.begin(), itEnd = info.end();
                it != itEnd; ++it) {
              edm::BranchID bid(it->branchID_);
              perProductParentage[bid].insert(orderedParentageIDs[it->parentageIDIndex_]);
            }
          }
        } else {
          //backwards compatible check
          TBranch* productProvBranch = eventMetaTree->GetBranch(edm::BranchTypeToBranchEntryInfoBranchName(edm::InEvent).c_str());
          if (0 != productProvBranch) {
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
          } else {
            std::cerr <<" ERROR, could not find provenance information so can not show dependencies\n";
            showDependencies_=false;
            extendedAncestors_ = false;
            extendedDescendants_ = false;
          }
        }
      }
    }
  }

  std::map<edm::BranchID, std::set<edm::BranchID> > parentToChildren;
  edm::ParentageRegistry& registry = *edm::ParentageRegistry::instance();

  if (extendedDescendants_) {
    for (auto const& itParentageSet : perProductParentage) {
      edm::BranchID childBranchID = itParentageSet.first;
      for (auto const& itParentageID : itParentageSet.second) {
        edm::Parentage const* parentage = registry.getMapped(itParentageID);
        if(0 != parentage) {
            for(std::vector<edm::BranchID>::const_iterator itBranch = parentage->parents().begin(), itEndBranch = parentage->parents().end();
                itBranch != itEndBranch;
                ++itBranch) {
              parentToChildren[*itBranch].insert(childBranchID);
            }
        } else {
          std::cerr << "  ERROR:parentage info not in registry ParentageID=" << itParentageID << std::endl;
        }
      }
    }
  }

  dumpEventFilteringParameterSets_(inputFile_.get());

  dumpProcessHistory_();

  std::cout << "---------Producers with data in file---------" << std::endl;

  //using edm::ParameterSetID as the key does not work
  //   typedef std::map<edm::ParameterSetID, std::vector<edm::BranchDescription> > IdToBranches
  ModuleToIdBranches moduleToIdBranches;
  //IdToBranches idToBranches;

  std::map<edm::BranchID, std::string> branchIDToBranchName;

  for(edm::ProductRegistry::ProductList::const_iterator it =
         reg_.productList().begin(), itEnd = reg_.productList().end();
       it != itEnd;
       ++it) {
    //force it to rebuild the branch name
    const_cast<edm::BranchDescription&>(it->second).init();

    if(showDependencies_ || extendedAncestors_ || extendedDescendants_) {
      branchIDToBranchName[it->second.branchID()] = it->second.branchName();
    }
    /*
      std::cout << it->second.branchName()
      << " id " << it->second.productID() << std::endl;
    */
    for(std::map<edm::ProcessConfigurationID, edm::ParameterSetID>::const_iterator
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

  for(ModuleToIdBranches::const_iterator it = moduleToIdBranches.begin(),
         itEnd = moduleToIdBranches.end();
       it != itEnd;
       ++it) {
    std::ostringstream sout;
    sout << "Module: " << it->first.second << " " << it->first.first << std::endl;
    std::set<edm::BranchID> allBranchIDsForLabelAndProcess;
    IdToBranches const& idToBranches = it->second;
    for(IdToBranches::const_iterator itIdBranch = idToBranches.begin(),
           itIdBranchEnd = idToBranches.end();
         itIdBranch != itIdBranchEnd;
         ++itIdBranch) {
      sout << " PSet id:" << itIdBranch->first << std::endl;
      if(!dontPrintProducts_) {
        sout << " products: {" << std::endl;
      }
      std::set<edm::BranchID> branchIDs;
      for(std::vector<edm::BranchDescription>::const_iterator itBranch = itIdBranch->second.begin(),
             itBranchEnd = itIdBranch->second.end();
           itBranch != itBranchEnd;
           ++itBranch) {
        if(!dontPrintProducts_) {
          sout << "  " << itBranch->branchName() << std::endl;
        }
        branchIDs.insert(itBranch->branchID());
        allBranchIDsForLabelAndProcess.insert(itBranch->branchID());
      }
      sout << " }" << std::endl;
      edm::ParameterSetID psid(itIdBranch->first);
      ParameterSetMap::const_iterator itpsm = psm_.find(psid);
      if(psm_.end() == itpsm) {
        ++errorCount_;
        errorLog_ << "No ParameterSetID for " << psid << std::endl;
        exitCode_ = 1;
      } else {
        sout << " parameters: ";
        prettyPrint(sout, edm::ParameterSet((*itpsm).second.pset()), " ", " ");
        sout << std::endl;
      }
      if(showDependencies_) {

        sout << " dependencies: {" << std::endl;
        std::set<edm::ParentageID> parentageIDs;
        for(std::set<edm::BranchID>::const_iterator itBranch = branchIDs.begin(), itBranchEnd = branchIDs.end();
            itBranch != itBranchEnd;
            ++itBranch) {

          //Save these BranchIDs
          std::set<edm::ParentageID> const& temp = perProductParentage[*itBranch];
          parentageIDs.insert(temp.begin(), temp.end());
        }
        for(std::set<edm::ParentageID>::const_iterator itParentID = parentageIDs.begin(), itEndParentID = parentageIDs.end();
            itParentID != itEndParentID;
            ++itParentID) {
          edm::Parentage const* parentage = registry.getMapped(*itParentID);
          if(0 != parentage) {
            for(std::vector<edm::BranchID>::const_iterator itBranch = parentage->parents().begin(), itEndBranch = parentage->parents().end();
                itBranch != itEndBranch;
                ++itBranch) {
              sout << "  " << branchIDToBranchName[*itBranch] << std::endl;
            }
          } else {
            sout << "  ERROR:parentage info not in registry ParentageID=" << *itParentID << std::endl;
          }
        }
        if(parentageIDs.empty()) {
          sout << "  no dependencies recorded (event may not contain data from this module)" << std::endl;
        }
        sout << " }" << std::endl;
      }
    } // end loop over PSetIDs
    if (extendedAncestors_) {
      sout << " extendedAncestors: {" << std::endl;
      std::set<edm::BranchID> ancestorBranchIDs;
      for (auto const& branchID : allBranchIDsForLabelAndProcess) {
        addAncestors(branchID, ancestorBranchIDs, sout, perProductParentage);
      }
      for (auto const& ancestorBranchID : ancestorBranchIDs) {
        sout << "  " << branchIDToBranchName[ancestorBranchID] << "\n";
      }
      sout << " }" << std::endl;
    }

    if (extendedDescendants_) {
      sout << " extendedDescendants: {" << std::endl;
      std::set<edm::BranchID> descendantBranchIDs;
      for (auto const& branchID : allBranchIDsForLabelAndProcess) {
        addDescendants(branchID, descendantBranchIDs, sout, parentToChildren);
      }
      for (auto const& descendantBranchID : descendantBranchIDs) {
        sout << "  " << branchIDToBranchName[descendantBranchID] << "\n";
      }
      sout << " }" << std::endl;
    }
    bool foundMatch = true;
    if(!findMatch_.empty()) {
      for (auto const& stringToFind : findMatch_) {
        if (sout.str().find(stringToFind) == std::string::npos) {
          foundMatch = false;
          break;
        }
      }
    }
    if (foundMatch) {
      std::cout <<sout.str()<<std::endl;
    }
  } // end loop over module label/process
  if(showOtherModules_) {
    std::cout << "---------Other Modules---------" << std::endl;
    historyGraph_.printOtherModulesHistory(psm_, moduleToIdBranches, findMatch_, errorLog_);
  }

  if(!excludeESModules_) {
    std::cout << "---------EventSetup---------" << std::endl;
    historyGraph_.printEventSetupHistory(psm_, findMatch_, errorLog_);
  }
  
  if(showTopLevelPSets_) {
    std::cout << "---------Top Level PSets---------" << std::endl;
    historyGraph_.printTopLevelPSetsHistory(psm_, findMatch_, errorLog_);
  }
  if(errorCount_ != 0) {
    exitCode_ = 1;
  }
}

void
ProvenanceDumper::addAncestors(edm::BranchID const& branchID, std::set<edm::BranchID>& ancestorBranchIDs, std::ostringstream& sout,
                               std::map<edm::BranchID, std::set<edm::ParentageID> >& perProductParentage) const {

  edm::ParentageRegistry& registry = *edm::ParentageRegistry::instance();

  std::set<edm::ParentageID> const& parentIDs = perProductParentage[branchID];
  for (auto const& parentageID : parentIDs) {
    edm::Parentage const* parentage = registry.getMapped(parentageID);
    if(0 != parentage) {
      for(std::vector<edm::BranchID>::const_iterator itBranch = parentage->parents().begin(), itEndBranch = parentage->parents().end();
          itBranch != itEndBranch;
          ++itBranch) {

        if (ancestorBranchIDs.insert(*itBranch).second) {
          addAncestors(*itBranch, ancestorBranchIDs, sout, perProductParentage);
        }
      }
    } else {
      sout << "  ERROR:parentage info not in registry ParentageID=" << parentageID << std::endl;
    }
  }
}

void
ProvenanceDumper::addDescendants(edm::BranchID const& branchID, std::set<edm::BranchID>& descendantBranchIDs, std::ostringstream& sout,
                                 std::map<edm::BranchID, std::set<edm::BranchID> >& parentToChildren) const {

  for (auto const& childBranchID : parentToChildren[branchID]) {
    if (descendantBranchIDs.insert(childBranchID).second) {
      addDescendants(childBranchID, descendantBranchIDs, sout, parentToChildren);
    }
  }
}

static char const* const kSortOpt = "sort";
static char const* const kSortCommandOpt = "sort,s";
static char const* const kDependenciesOpt = "dependencies";
static char const* const kDependenciesCommandOpt = "dependencies,d";
static char const* const kExtendedAncestorsOpt = "extendedAncestors";
static char const* const kExtendedAncestorsCommandOpt = "extendedAncestors,x";
static char const* const kExtendedDescendantsOpt = "extendedDescendants";
static char const* const kExtendedDescendantsCommandOpt = "extendedDescendants,c";
static char const* const kExcludeESModulesOpt = "excludeESModules";
static char const* const kExcludeESModulesCommandOpt = "excludeESModules,e";
static char const* const kShowAllModulesOpt = "showAllModules";
static char const* const kShowAllModulesCommandOpt = "showAllModules,a";
static char const* const kFindMatchOpt = "findMatch";
static char const* const kFindMatchCommandOpt = "findMatch,f";
static char const* const kDontPrintProductsOpt = "dontPrintProducts";
static char const* const kDontPrintProductsCommandOpt = "dontPrintProducts,p";
static char const* const kShowTopLevelPSetsOpt = "showTopLevelPSets";
static char const* const kShowTopLevelPSetsCommandOpt ="showTopLevelPSets,t";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";
static char const* const kFileNameOpt = "input-file";
static char const* const kFileNameCommandOpt = "input-file";

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
   , "print what data each EDProducer is directly dependent upon")
  (kExtendedAncestorsCommandOpt
   , "print what data each EDProducer is dependent upon including indirect dependences")
  (kExtendedDescendantsCommandOpt
   , "print what data depends on the data each EDProducer produces including indirect dependences")
  (kExcludeESModulesCommandOpt
   , "do not print ES module information")
  (kShowAllModulesCommandOpt
   , "show all modules (not just those that created data in the file)")
  (kShowTopLevelPSetsCommandOpt,"show all top level PSets")
  (kFindMatchCommandOpt, boost::program_options::value<std::vector<std::string> >(),
    "show only modules whose information contains the matching string (or all the matching strings, this option can be repeated with different strings)")
  (kDontPrintProductsCommandOpt
   , "do not print products produced by module")
  ;
  //we don't want users to see these in the help messages since this
  // name only exists since the parser needs it
  options_description hidden;
  hidden.add_options()(kFileNameOpt, value<std::string>(), "file name");

  //full list of options for the parser
  options_description cmdline_options;
  cmdline_options.add(desc).add(hidden);

  positional_options_description p;
  p.add(kFileNameOpt, -1);

  variables_map vm;
  try {
    store(command_line_parser(argc, argv).options(cmdline_options).positional(p).run(), vm);
    notify(vm);
  } catch(error const& iException) {
    std::cerr << iException.what();
    return 1;
  }

  if(vm.count(kHelpOpt)) {
    std::cout << desc << std::endl;
    return 0;
  }

  if(vm.count(kSortOpt)) {
    HistoryNode::sort_ = true;
  }

  bool showDependencies = false;
  if(vm.count(kDependenciesOpt)) {
    showDependencies = true;
  }

  bool extendedAncestors = false;
  if(vm.count(kExtendedAncestorsOpt)) {
    extendedAncestors = true;
  }

  bool extendedDescendants = false;
  if(vm.count(kExtendedDescendantsOpt)) {
    extendedDescendants = true;
  }

  bool excludeESModules = false;
  if(vm.count(kExcludeESModulesOpt)) {
    excludeESModules = true;
  }

  bool showAllModules = false;
  if(vm.count(kShowAllModulesOpt)) {
    showAllModules = true;
  }
  
  bool showTopLevelPSets = false;
  if(vm.count(kShowTopLevelPSetsOpt)) {
    showTopLevelPSets=true;
  }

  std::string fileName;
  if(vm.count(kFileNameOpt)) {
    try {
      fileName = vm[kFileNameOpt].as<std::string>();
    } catch(boost::bad_any_cast const& e) {
      std::cout << e.what() << std::endl;
      return 2;
    }
  } else {
    std::cout << "Data file not specified." << std::endl;
    std::cout << desc << std::endl;
    return 2;
  }

  std::vector<std::string> findMatch;
  if(vm.count(kFindMatchOpt)) {
    try {
      findMatch = vm[kFindMatchOpt].as<std::vector<std::string> >();
    } catch(boost::bad_any_cast const& e) {
      std::cout << e.what() << std::endl;
      return 2;
    }
  }

  bool dontPrintProducts = false;
  if(vm.count(kDontPrintProductsOpt)) {
    dontPrintProducts=true;
  }

  //silence ROOT warnings about missing dictionaries
  gErrorIgnoreLevel = kError;

  //make sure dictionaries can be used for reading
  ROOT::Cintex::Cintex::Enable();

  ProvenanceDumper dumper(fileName, showDependencies, extendedAncestors, extendedDescendants,
                          excludeESModules, showAllModules, showTopLevelPSets, findMatch, dontPrintProducts);
  int exitCode(0);
  try {
    dumper.dump();
    exitCode = dumper.exitCode();
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
  return exitCode;
}
