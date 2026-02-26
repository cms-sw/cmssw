#include "DataFormats/Common/interface/setIsMergeable.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/branchIDToProductID.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/StoredProductProvenance.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Catalog/interface/StorageURLModifier.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWStorage/Services/interface/setupSiteLocalConfig.h"

#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "TError.h"
#include "TFile.h"
#include "TTree.h"

#include "boost/program_options.hpp"

#include <format>

#include <cassert>
#include <iostream>
#include <memory>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <vector>

static std::ostream& prettyPrint(std::ostream& oStream,
                                 edm::ParameterSet const& iPSet,
                                 std::string const& iIndent,
                                 std::string const& iIndentDelta);

static std::string const triggerResults = std::string("TriggerResults");
static std::string const triggerPaths = std::string("@trigger_paths");
static std::string const source = std::string("source");
static std::string const input = std::string("@main_input");

namespace {
  struct ModuleInformation {
    std::set<unsigned int> processSimpleIDs;
    std::set<edm::ProductDescription> products;
  };
  // key is ParameterSetID string representation
  using IdToBranches = std::map<std::string, ModuleInformation>;
  // key is data product's (process name, module label)
  using ModuleToIdBranches = std::map<std::pair<std::string, std::string>, IdToBranches>;

  using ParameterSetMap = std::map<edm::ParameterSetID, edm::ParameterSetBlob>;

  class HistoryNode {
  public:
    HistoryNode() = default;

    HistoryNode(edm::ProcessConfiguration const& iConfig, unsigned int iSimpleId)
        : config_(iConfig), simpleId_(iSimpleId) {}

    edm::ParameterSetID const& parameterSetID() const { return config_.parameterSetID(); }

    std::string const& processName() const { return config_.processName(); }

    void print(std::ostream& os) const { os << config_.processName() << " [" << simpleId_ << "]"; }

    edm::ProcessConfigurationID configurationID() const { return config_.id(); }

  private:
    edm::ProcessConfiguration config_;
    unsigned int simpleId_ = 0;
  };

  class HistoryChain {
  public:
    explicit HistoryChain(edm::ProcessHistoryID id) : historyId_(id) {}

    void addChild(HistoryNode const& child) { children_.push_back(child); }

    void reserve(std::size_t s) { children_.reserve(s); }

    std::size_t size() const { return children_.size(); }

    typedef std::vector<HistoryNode>::const_iterator const_iterator;
    typedef std::vector<HistoryNode>::iterator iterator;

    iterator begin() { return children_.begin(); }
    iterator end() { return children_.end(); }

    const_iterator begin() const { return children_.begin(); }
    const_iterator end() const { return children_.end(); }

    void printHistory(bool showHistoryID) const;

  private:
    std::vector<HistoryNode> children_;
    edm::ProcessHistoryID historyId_;
  };

  std::ostream& operator<<(std::ostream& os, HistoryNode const& node) {
    node.print(os);
    return os;
  }

  std::string formatHeader(std::string_view text) {
    constexpr std::string_view decoration = "---------";
    return std::format("{}{}{}", decoration, text, decoration);
  }
}  // namespace

void HistoryChain::printHistory(bool showHistoryID) const {
  constexpr std::string_view indent = "  ";
  std::cout << "History: {" << std::endl;
  if (showHistoryID) {
    std::cout << indent << "History id: " << historyId_ << std::endl;
  }
  for (auto const& item : children_) {
    std::cout << indent << item << std::endl;
    if (showHistoryID) {
      std::cout << indent << indent << "Configuration id: " << item.configurationID() << std::endl;
    }
  }
  std::cout << "}" << std::endl;
}

std::string eventSetupComponent(char const* iType,
                                std::string const& iCompName,
                                edm::ParameterSet const& iProcessConfig,
                                std::string const& iProcessName) {
  std::ostringstream result;
  edm::ParameterSet const& pset = iProcessConfig.getParameterSet(iCompName);
  std::string name(pset.getParameter<std::string>("@module_label"));
  if (name.empty()) {
    name = pset.getParameter<std::string>("@module_type");
  }

  result << iType << ": " << name << " " << iProcessName << "\n"
         << " parameters: ";
  prettyPrint(result, pset, " ", " ");
  return result.str();
}

std::string nonProducerComponent(std::string const& iCompName,
                                 edm::ParameterSet const& iProcessConfig,
                                 std::string const& iProcessName) {
  std::ostringstream result;
  edm::ParameterSet const& pset = iProcessConfig.getParameterSet(iCompName);
  std::string label(pset.getParameter<std::string>("@module_label"));

  result << "Module: " << label << " " << iProcessName << "\n"
         << " parameters: ";
  prettyPrint(result, pset, " ", " ");
  return result.str();
}

static void appendToSet(std::set<std::string>& iSet, std::vector<std::string> const& iFrom) {
  for (auto const& n : iFrom) {
    iSet.insert(n);
  }
}

static std::string topLevelPSet(std::string const& iName,
                                edm::ParameterSet const& iProcessConfig,
                                std::string const& iProcessName) {
  std::ostringstream result;
  edm::ParameterSet const& pset = iProcessConfig.getParameterSet(iName);

  result << "PSet: " << iName << " " << iProcessName << "\n"
         << " parameters: ";
  prettyPrint(result, pset, " ", " ");
  return result.str();
}

namespace {
  std::unique_ptr<TFile> makeTFileWithLookup(std::string const& filename) {
    // See if it is a logical file name.
    auto operate = edm::setupSiteLocalConfig();
    std::string override;
    std::vector<std::string> fileNames;
    fileNames.push_back(filename);
    edm::InputFileCatalog catalog(fileNames, override, true, edm::SciTagCategory::Undefined);
    if (catalog.fileNames(0)[0] == filename) {
      throw cms::Exception("FileNotFound", "RootFile::RootFile()")
          << "File " << filename << " was not found or could not be opened.\n";
    }
    // filename is a valid LFN.
    std::unique_ptr<TFile> result(TFile::Open(catalog.fileNames(0)[0].c_str()));
    if (!result.get()) {
      throw cms::Exception("FileNotFound", "RootFile::RootFile()")
          << "File " << fileNames[0] << " was not found or could not be opened.\n";
    }
    return result;
  }

  // Open the input file, returning the TFile object that represents it.
  // The returned unique_ptr will not be null. The argument must not be null.
  // We first try the file name as a PFN, so that the catalog and related
  // services are not loaded unless needed.
  std::unique_ptr<TFile> makeTFile(std::string const& filename) {
    gErrorIgnoreLevel = kFatal;
    std::unique_ptr<TFile> result(TFile::Open(filename.c_str()));
    gErrorIgnoreLevel = kError;
    if (!result.get()) {
      // Try again with catalog.
      return makeTFileWithLookup(filename);
    }
    return result;
  }
}  // namespace

static std::ostream& prettyPrint(std::ostream& os,
                                 edm::ParameterSetEntry const& psetEntry,
                                 std::string const& iIndent,
                                 std::string const& iIndentDelta) {
  char const* trackiness = (psetEntry.isTracked() ? "tracked" : "untracked");
  os << "PSet " << trackiness << " = (";
  prettyPrint(os, psetEntry.pset(), iIndent + iIndentDelta, iIndentDelta);
  os << ")";
  return os;
}

static std::ostream& prettyPrint(std::ostream& os,
                                 edm::VParameterSetEntry const& vpsetEntry,
                                 std::string const& iIndent,
                                 std::string const& iIndentDelta) {
  std::vector<edm::ParameterSet> const& vps = vpsetEntry.vpset();
  os << "VPSet " << (vpsetEntry.isTracked() ? "tracked" : "untracked") << " = ({" << std::endl;
  std::string newIndent = iIndent + iIndentDelta;
  std::string start;
  std::string const between(",\n");
  for (auto const& item : vps) {
    os << start << newIndent;
    prettyPrint(os, item, newIndent, iIndentDelta);
    start = between;
  }
  if (!vps.empty()) {
    os << std::endl;
  }
  os << iIndent << "})";
  return os;
}

static std::ostream& prettyPrint(std::ostream& oStream,
                                 edm::ParameterSet const& iPSet,
                                 std::string const& iIndent,
                                 std::string const& iIndentDelta) {
  std::string newIndent = iIndent + iIndentDelta;

  oStream << "{" << std::endl;
  for (auto const& item : iPSet.tbl()) {
    // indent a bit
    oStream << newIndent << item.first << ": " << item.second << std::endl;
  }
  for (auto const& item : iPSet.psetTable()) {
    // indent a bit
    edm::ParameterSetEntry const& pe = item.second;
    oStream << newIndent << item.first << ": ";
    prettyPrint(oStream, pe, iIndent, iIndentDelta);
    oStream << std::endl;
  }
  for (auto const& item : iPSet.vpsetTable()) {
    // indent a bit
    edm::VParameterSetEntry const& pe = item.second;
    oStream << newIndent << item.first << ": ";
    prettyPrint(oStream, pe, newIndent, iIndentDelta);
    oStream << std::endl;
  }
  oStream << iIndent << "}";

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
                   bool showHistoryID,
                   std::vector<std::string> const& findMatch,
                   bool dontPrintProducts,
                   std::string const& dumpPSetID,
                   int productIDEntry,
                   bool sort);

  ProvenanceDumper(ProvenanceDumper const&) = delete;             // Disallow copying and moving
  ProvenanceDumper& operator=(ProvenanceDumper const&) = delete;  // Disallow copying and moving

  // Write the provenenace information to the given stream.
  void dump();
  void printErrors(std::ostream& os);
  int exitCode() const;

private:
  void addAncestors(edm::BranchID const& branchID,
                    std::set<edm::BranchID>& ancestorBranchIDs,
                    std::ostringstream& sout,
                    std::map<edm::BranchID, std::set<edm::ParentageID>>& perProductParentage) const;

  void addDescendants(edm::BranchID const& branchID,
                      std::set<edm::BranchID>& descendantBranchIDs,
                      std::ostringstream& sout,
                      std::map<edm::BranchID, std::set<edm::BranchID>>& parentToChildren) const;

  std::string filename_;
  edm::propagate_const<std::unique_ptr<TFile>> inputFile_;
  int exitCode_;
  std::stringstream errorLog_;
  int errorCount_;
  edm::ProductRegistry reg_;
  edm::ProcessConfigurationVector phc_;
  edm::ProcessHistoryVector phv_;
  ParameterSetMap psm_;
  std::vector<HistoryChain> histories_;
  bool showDependencies_;
  bool extendedAncestors_;
  bool extendedDescendants_;
  bool excludeESModules_;
  bool showOtherModules_;
  bool productRegistryPresent_;
  bool showTopLevelPSets_;
  bool showHistoryID_;
  std::vector<std::string> findMatch_;
  bool dontPrintProducts_;
  std::string dumpPSetID_;
  int const productIDEntry_;
  bool const sort_;

  // The "simple ID" unsigned int is a counter for processes that have
  // the same name but with different ProcessConfigurationID. The
  // "simple ID" is used in the process history and configuration
  // printouts for human-friendlier identification of processes than
  // the ProcessConfigurationID.
  using ProcessSimpleIDsType = std::map<edm::ProcessConfigurationID, unsigned int>;

  void work_();
  ProcessSimpleIDsType dumpProcessHistory_();
  void dumpOtherModulesHistory_(ModuleToIdBranches const&);
  void dumpEventSetupHistory_();
  void dumpTopLevelPSetsHistory_();

  bool matchToFindMatch_(std::string_view str) const {
    if (findMatch_.empty())
      return true;
    for (auto const& stringToFind : findMatch_) {
      if (str.find(stringToFind) == std::string::npos) {
        return false;
      }
    }
    return true;
  }

  void dumpEventFilteringParameterSets_(TFile* file, ProcessSimpleIDsType const& processSimpleIDs);
  void dumpEventFilteringParameterSets(edm::EventSelectionIDVector const& ids,
                                       ProcessSimpleIDsType const& processSimpleIDs);
  void dumpParameterSetForID_(edm::ParameterSetID const& id);
  std::optional<std::tuple<edm::BranchIDListHelper, std::vector<edm::ProcessIndex>>> makeBranchIDListHelper();
};

ProvenanceDumper::ProvenanceDumper(std::string const& filename,
                                   bool showDependencies,
                                   bool extendedAncestors,
                                   bool extendedDescendants,
                                   bool excludeESModules,
                                   bool showOtherModules,
                                   bool showTopLevelPSets,
                                   bool showHistoryID,
                                   std::vector<std::string> const& findMatch,
                                   bool dontPrintProducts,
                                   std::string const& dumpPSetID,
                                   int productIDEntry,
                                   bool sort)
    : filename_(filename),
      inputFile_(makeTFile(filename)),
      exitCode_(0),
      errorLog_(),
      errorCount_(0),
      showDependencies_(showDependencies),
      extendedAncestors_(extendedAncestors),
      extendedDescendants_(extendedDescendants),
      excludeESModules_(excludeESModules),
      showOtherModules_(showOtherModules),
      productRegistryPresent_(true),
      showTopLevelPSets_(showTopLevelPSets),
      showHistoryID_(showHistoryID),
      findMatch_(findMatch),
      dontPrintProducts_(dontPrintProducts),
      dumpPSetID_(dumpPSetID),
      productIDEntry_(productIDEntry),
      sort_(sort) {}

void ProvenanceDumper::dump() { work_(); }

void ProvenanceDumper::printErrors(std::ostream& os) {
  if (errorCount_ > 0)
    os << errorLog_.str() << std::endl;
}

int ProvenanceDumper::exitCode() const { return exitCode_; }

void ProvenanceDumper::dumpEventFilteringParameterSets(edm::EventSelectionIDVector const& ids,
                                                       ProcessSimpleIDsType const& processSimpleIDs) {
  edm::EventSelectionIDVector::size_type num_ids = ids.size();
  if (num_ids == 0) {
    std::cout << "No event filtering information is available.\n";
  } else {
    std::cout << "Event filtering information for " << num_ids << " processing steps is available.\n"
              << "The ParameterSets are printed out with the oldest process first.\n";
    std::string const indent = "  ";
    for (edm::EventSelectionIDVector::size_type i = 0; i != num_ids; ++i) {
      auto found = psm_.find(ids[i]);
      if (found == psm_.end()) {
        std::cout << "PSet id " << ids[i] << " not found" << std::endl;
        continue;
      }
      std::cout << "Event filtering:\n" << indent << "PSet id: " << ids[i] << "\n" << indent << "parameters: ";
      edm::ParameterSet ps(found->second.pset());
      prettyPrint(std::cout, ps, indent, indent);
      std::cout << std::endl;
    }
  }
}

void ProvenanceDumper::dumpEventFilteringParameterSets_(TFile* file, ProcessSimpleIDsType const& processSimpleIDs) {
  TTree* history = dynamic_cast<TTree*>(file->Get(edm::poolNames::eventHistoryTreeName().c_str()));
  if (history != nullptr) {
    edm::History h;
    edm::History* ph = &h;

    history->SetBranchAddress(edm::poolNames::eventHistoryBranchName().c_str(), &ph);
    if (history->GetEntry(0) <= 0) {
      std::cout << "No event filtering information is available; the event history tree has no entries\n";
    } else {
      dumpEventFilteringParameterSets(h.eventSelectionIDs(), processSimpleIDs);
    }
  } else {
    TTree* events = dynamic_cast<TTree*>(file->Get(edm::poolNames::eventTreeName().c_str()));
    assert(events != nullptr);
    TBranch* eventSelectionsBranch = events->GetBranch(edm::poolNames::eventSelectionsBranchName().c_str());
    if (eventSelectionsBranch == nullptr)
      return;
    edm::EventSelectionIDVector ids;
    edm::EventSelectionIDVector* pids = &ids;
    eventSelectionsBranch->SetAddress(&pids);
    if (eventSelectionsBranch->GetEntry(0) <= 0) {
      std::cout << "No event filtering information is available; the event selections branch has no entries\n";
    } else {
      dumpEventFilteringParameterSets(ids, processSimpleIDs);
    }
  }
}

void ProvenanceDumper::dumpParameterSetForID_(edm::ParameterSetID const& id) {
  std::cout << "ParameterSetID: " << id << '\n';
  if (id.isValid()) {
    ParameterSetMap::const_iterator i = psm_.find(id);
    if (i == psm_.end()) {
      std::cout << "We are unable to find the corresponding ParameterSet\n";
      edm::ParameterSet empty;
      empty.registerIt();
      if (id == empty.id()) {
        std::cout << "But it would have been empty anyway\n";
      }
    } else {
      edm::ParameterSet ps(i->second.pset());
      prettyPrint(std::cout, ps, " ", " ");
      std::cout << '\n';
    }
  } else {
    std::cout << "This ID is not valid\n";
  }
  std::cout << "     -------------------------\n";
}

ProvenanceDumper::ProcessSimpleIDsType ProvenanceDumper::dumpProcessHistory_() {
  std::cout << formatHeader("Processing histories") << std::endl;

  ProcessSimpleIDsType simpleIDs;
  // expect the outer vector to not grow large so that linear search is still acceptable
  std::vector<std::vector<edm::ProcessConfiguration const*>> processesByName;
  histories_.reserve(phv_.size());
  for (auto const& ph : phv_) {
    // loop over the history entries to find ProcessConfigurations that are the same
    // use a simple count ID for each process that have the same name, but different ProcessConfigurationID
    histories_.emplace_back(ph.id());
    HistoryChain& chain = histories_.back();
    chain.reserve(ph.size());

    for (auto const& pc : ph) {
      unsigned int& id = simpleIDs[pc.id()];
      if (id == 0) {
        // first time seeing this ID
        auto found = std::ranges::find_if(processesByName, [&name = pc.processName()](auto const& pcs) {
          return pcs.front()->processName() == name;
        });
        if (found == processesByName.end()) {
          processesByName.emplace_back(std::initializer_list<edm::ProcessConfiguration const*>({&pc}));
          id = 1;
        } else {
          found->emplace_back(&pc);
          id = found->size();
        }
      }

      chain.addChild(HistoryNode(pc, id));
    }
    chain.printHistory(showHistoryID_);
  }

  std::cout << formatHeader("Processes") << std::endl;
  auto concatenate = [](std::ostream& os, std::vector<std::string> const& vs) {
    if (not vs.empty()) {
      os << vs.front();
      for (auto it = vs.begin() + 1; it != vs.end(); ++it) {
        os << ", " << *it;
      }
    }
  };
  constexpr std::string_view indent = "  ";
  for (auto const& processes : processesByName) {
    for (auto const* pc : processes) {
      auto const& hwresources = pc->hardwareResourcesDescription();
      std::cout << "Process: " << pc->processName() << " [" << simpleIDs.at(pc->id()) << "]\n"
                << indent << "PSet id: " << pc->parameterSetID() << "\n"
                << indent << "version: '" << pc->releaseVersion() << "'\n"
                << indent << "microarchitecture: " << hwresources.microarchitecture << "\n"
                << indent << "accelerators: ";
      concatenate(std::cout, hwresources.selectedAccelerators);
      std::cout << "\n" << indent << "CPU models: ";
      concatenate(std::cout, hwresources.cpuModels);
      if (not hwresources.gpuModels.empty()) {
        std::cout << "\n" << indent << "GPU models: ";
        concatenate(std::cout, hwresources.gpuModels);
      }
      std::cout << "\n" << std::endl;
    }
  }
  return simpleIDs;
}

void ProvenanceDumper::dumpOtherModulesHistory_(ModuleToIdBranches const& iModules) {
  for (auto const& chain : histories_) {
    for (auto const& node : chain) {
      //Get ParameterSet for process
      ParameterSetMap::const_iterator itFind = psm_.find(node.parameterSetID());
      if (itFind == psm_.end()) {
        errorLog_ << "No ParameterSetID for " << node.parameterSetID() << std::endl;
        continue;
      }

      edm::ParameterSet processConfig(itFind->second.pset());
      std::vector<std::string> moduleStrings;
      //get all modules
      std::vector<std::string> modules = processConfig.getParameter<std::vector<std::string>>("@all_modules");
      for (auto& moduleLabel : modules) {
        //if we didn't already handle this from the branches
        if (iModules.end() == iModules.find(std::make_pair(node.processName(), moduleLabel))) {
          std::string retValue(nonProducerComponent(moduleLabel, processConfig, node.processName()));
          if (matchToFindMatch_(retValue)) {
            moduleStrings.push_back(std::move(retValue));
          }
        }
      }
      if (sort_) {
        std::sort(moduleStrings.begin(), moduleStrings.end());
      }
      std::copy(moduleStrings.begin(), moduleStrings.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
    }
  }
}

void ProvenanceDumper::dumpEventSetupHistory_() {
  for (auto const& chain : histories_) {
    for (auto const& node : chain) {
      //Get ParameterSet for process
      ParameterSetMap::const_iterator itFind = psm_.find(node.parameterSetID());
      if (itFind == psm_.end()) {
        errorLog_ << "No ParameterSetID for " << node.parameterSetID() << std::endl;
        continue;
      }

      edm::ParameterSet processConfig(itFind->second.pset());
      std::vector<std::string> sourceStrings, moduleStrings;

      //get the sources
      auto const sources = processConfig.getParameter<std::vector<std::string>>("@all_essources");
      for (auto const& sourceLabel : sources) {
        std::string retValue = eventSetupComponent("ESSource", sourceLabel, processConfig, node.processName());
        if (matchToFindMatch_(retValue)) {
          sourceStrings.push_back(std::move(retValue));
        }
      }

      //get the modules
      auto const modules = processConfig.getParameter<std::vector<std::string>>("@all_esmodules");
      for (auto const& moduleLabel : modules) {
        std::string retValue = eventSetupComponent("ESModule", moduleLabel, processConfig, node.processName());
        if (matchToFindMatch_(retValue)) {
          moduleStrings.push_back(std::move(retValue));
        }
      }

      if (sort_) {
        std::sort(sourceStrings.begin(), sourceStrings.end());
        std::sort(moduleStrings.begin(), moduleStrings.end());
      }

      std::copy(sourceStrings.begin(), sourceStrings.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
      std::copy(moduleStrings.begin(), moduleStrings.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
    }
  }
}

void ProvenanceDumper::dumpTopLevelPSetsHistory_() {
  for (auto const& chain : histories_) {
    for (auto const& node : chain) {
      //Get ParameterSet for process
      ParameterSetMap::const_iterator itFind = psm_.find(node.parameterSetID());
      if (itFind == psm_.end()) {
        errorLog_ << "No ParameterSetID for " << node.parameterSetID() << std::endl;
        continue;
      }

      //Get ParameterSet for process
      edm::ParameterSet processConfig(itFind->second.pset());
      //Need to get the names of PSets which are used by the framework (e.g. names of modules)
      std::set<std::string> namesToExclude;
      appendToSet(namesToExclude, processConfig.getParameter<std::vector<std::string>>("@all_modules"));
      appendToSet(namesToExclude, processConfig.getParameter<std::vector<std::string>>("@all_sources"));
      appendToSet(namesToExclude, processConfig.getParameter<std::vector<std::string>>("@all_loopers"));
      appendToSet(namesToExclude, processConfig.getParameter<std::vector<std::string>>("@all_esmodules"));
      appendToSet(namesToExclude, processConfig.getParameter<std::vector<std::string>>("@all_essources"));
      appendToSet(namesToExclude, processConfig.getParameter<std::vector<std::string>>("@all_esprefers"));
      if (processConfig.existsAs<std::vector<std::string>>("all_aliases")) {
        appendToSet(namesToExclude, processConfig.getParameter<std::vector<std::string>>("@all_aliases"));
      }

      std::vector<std::string> allNames{};
      processConfig.getParameterSetNames(allNames);

      std::vector<std::string> results;
      for (auto const& name : allNames) {
        if (name.empty() || '@' == name[0] || namesToExclude.find(name) != namesToExclude.end()) {
          continue;
        }
        std::string retValue = topLevelPSet(name, processConfig, node.processName());
        if (matchToFindMatch_(retValue)) {
          results.push_back(std::move(retValue));
        }
      }
      if (sort_) {
        std::sort(results.begin(), results.end());
      }
      std::copy(results.begin(), results.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
    }
  }
}

std::optional<std::tuple<edm::BranchIDListHelper, std::vector<edm::ProcessIndex>>>
ProvenanceDumper::makeBranchIDListHelper() {
  // BranchID-to-ProductID mapping disabled
  if (productIDEntry_ < 0) {
    return {};
  }

  TTree* metaTree = dynamic_cast<TTree*>(inputFile_->Get(edm::poolNames::metaDataTreeName().c_str()));
  if (nullptr == metaTree) {
    //std::cerr << "Did not find " << edm::poolNames::metaDataTreeName() << " tree" << std::endl;
    return {};
  }

  TBranch* branchIDListsBranch = metaTree->GetBranch(edm::poolNames::branchIDListBranchName().c_str());
  if (nullptr == branchIDListsBranch) {
    /*
    std::cerr << "Did not find " << edm::poolNames::branchIDListBranchName() << " from "
              << edm::poolNames::metaDataTreeName() << " tree" << std::endl;
    */
    return {};
  }

  edm::BranchIDLists branchIDLists;
  edm::BranchIDLists* branchIDListsPtr = &branchIDLists;
  branchIDListsBranch->SetAddress(&branchIDListsPtr);
  if (branchIDListsBranch->GetEntry(0) <= 0) {
    //std::cerr << "Failed to read an entry from " << edm::poolNames::branchIDListBranchName() << std::endl;
    return {};
  }

  edm::BranchIDListHelper branchIDListHelper;
  branchIDListHelper.updateFromInput(branchIDLists);

  TTree* events = dynamic_cast<TTree*>(inputFile_->Get(edm::poolNames::eventTreeName().c_str()));
  assert(events != nullptr);
  TBranch* branchListIndexesBranch = events->GetBranch(edm::poolNames::branchListIndexesBranchName().c_str());
  if (nullptr == branchListIndexesBranch) {
    /*
    std::cerr << "Did not find " << edm::poolNames::branchListIndexesBranchName() << " from "
              << edm::poolNames::eventTreeName() << " tree" << std::endl;
    */
    return {};
  }
  edm::BranchListIndexes branchListIndexes;
  edm::BranchListIndexes* pbranchListIndexes = &branchListIndexes;
  branchListIndexesBranch->SetAddress(&pbranchListIndexes);
  if (branchListIndexesBranch->GetEntry(productIDEntry_) <= 0 or branchListIndexes.empty()) {
    /*
    std::cerr << "Failed to read entry from " << edm::poolNames::branchListIndexesBranchName() << ", or it is empty"
              << std::endl;
    */
    return {};
  }

  if (not branchIDListHelper.fixBranchListIndexes(branchListIndexes)) {
    //std::cerr << "Call to branchIDListHelper.fixBranchListIndexes() failed" << std::endl;
    return {};
  }

  // Fill in helper map for Branch to ProductID mapping
  auto branchListIndexToProcessIndex = edm::makeBranchListIndexToProcessIndex(branchListIndexes);

  return std::tuple(std::move(branchIDListHelper), std::move(branchListIndexToProcessIndex));
}

void ProvenanceDumper::work_() {
  TTree* meta = dynamic_cast<TTree*>(inputFile_->Get(edm::poolNames::metaDataTreeName().c_str()));
  assert(nullptr != meta);

  edm::ProductRegistry* pReg = &reg_;
  if (meta->FindBranch(edm::poolNames::productDescriptionBranchName().c_str()) != nullptr) {
    meta->SetBranchAddress(edm::poolNames::productDescriptionBranchName().c_str(), &pReg);
  } else {
    productRegistryPresent_ = false;
  }

  ParameterSetMap* pPsm = &psm_;
  if (meta->FindBranch(edm::poolNames::parameterSetMapBranchName().c_str()) != nullptr) {
    meta->SetBranchAddress(edm::poolNames::parameterSetMapBranchName().c_str(), &pPsm);
  } else {
    TTree* psetTree = dynamic_cast<TTree*>(inputFile_->Get(edm::poolNames::parameterSetsTreeName().c_str()));
    assert(nullptr != psetTree);
    typedef std::pair<edm::ParameterSetID, edm::ParameterSetBlob> IdToBlobs;
    IdToBlobs idToBlob;
    IdToBlobs* pIdToBlob = &idToBlob;
    psetTree->SetBranchAddress(edm::poolNames::idToParameterSetBlobsBranchName().c_str(), &pIdToBlob);
    for (long long i = 0; i != psetTree->GetEntries(); ++i) {
      psetTree->GetEntry(i);
      psm_.insert(idToBlob);
    }
  }

  edm::ProcessHistoryVector* pPhv = &phv_;
  if (meta->FindBranch(edm::poolNames::processHistoryBranchName().c_str()) != nullptr) {
    meta->SetBranchAddress(edm::poolNames::processHistoryBranchName().c_str(), &pPhv);
  }

  edm::ProcessHistoryMap phm;
  edm::ProcessHistoryMap* pPhm = &phm;
  if (meta->FindBranch(edm::poolNames::processHistoryMapBranchName().c_str()) != nullptr) {
    meta->SetBranchAddress(edm::poolNames::processHistoryMapBranchName().c_str(), &pPhm);
  }

  if (meta->FindBranch(edm::poolNames::moduleDescriptionMapBranchName().c_str()) != nullptr) {
    if (meta->GetBranch(edm::poolNames::moduleDescriptionMapBranchName().c_str())->GetSplitLevel() != 0) {
      meta->SetBranchStatus((edm::poolNames::moduleDescriptionMapBranchName() + ".*").c_str(), false);
    } else {
      meta->SetBranchStatus(edm::poolNames::moduleDescriptionMapBranchName().c_str(), false);
    }
  }

  meta->GetEntry(0);
  assert(nullptr != pReg);

  edm::pset::Registry& psetRegistry = *edm::pset::Registry::instance();
  for (auto const& item : psm_) {
    edm::ParameterSet pset(item.second.pset());
    pset.setID(item.first);
    psetRegistry.insertMapped(pset);
  }

  if (!phv_.empty()) {
    // (Re-)Sort according to reduced history ID in order to have a
    // stable order with respect to hardware differences
    std::ranges::stable_sort(phv_, {}, [](auto const& history) {
      auto copy = history;
      copy.reduce();
      return copy.id();
    });

    for (auto const& history : phv_) {
      for (auto const& process : history) {
        phc_.push_back(process);
      }
    }
    edm::sort_all(phc_);
    phc_.erase(std::unique(phc_.begin(), phc_.end()), phc_.end());

  }
  // backward compatibility
  else if (!phm.empty()) {
    for (auto const& history : phm) {
      phv_.push_back(history.second);
      for (auto const& process : history.second) {
        phc_.push_back(process);
      }
    }
    edm::sort_all(phc_);
    phc_.erase(std::unique(phc_.begin(), phc_.end()), phc_.end());
  }

  if (!dumpPSetID_.empty()) {
    edm::ParameterSetID psetID;
    try {
      psetID = edm::ParameterSetID(dumpPSetID_);
    } catch (cms::Exception const& x) {
      throw cms::Exception("Command Line Argument")
          << "Illegal ParameterSetID string. It should contain 32 hexadecimal characters";
    }
    dumpParameterSetForID_(psetID);
    return;
  }

  // Helper to map BranchID to ProductID (metadata tree needed also for parentage information)
  auto branchIDListHelperAndToProcessIndex = makeBranchIDListHelper();

  //Prepare the parentage information if requested
  std::map<edm::BranchID, std::set<edm::ParentageID>> perProductParentage;

  if (showDependencies_ || extendedAncestors_ || extendedDescendants_) {
    TTree* parentageTree = dynamic_cast<TTree*>(inputFile_->Get(edm::poolNames::parentageTreeName().c_str()));
    if (nullptr == parentageTree) {
      std::cerr << "ERROR, no Parentage tree available so cannot show dependencies, ancestors, or descendants.\n";
      std::cerr << "Possibly this is not a standard EDM format file. For example, dependency, ancestor, and\n";
      std::cerr << "descendant options to edmProvDump will not work with nanoAOD format files.\n\n";
      showDependencies_ = false;
      extendedAncestors_ = false;
      extendedDescendants_ = false;
    } else {
      edm::ParentageRegistry& registry = *edm::ParentageRegistry::instance();

      std::vector<edm::ParentageID> orderedParentageIDs;
      orderedParentageIDs.reserve(parentageTree->GetEntries());
      for (Long64_t i = 0, numEntries = parentageTree->GetEntries(); i < numEntries; ++i) {
        edm::Parentage parentageBuffer;
        edm::Parentage* pParentageBuffer = &parentageBuffer;
        parentageTree->SetBranchAddress(edm::poolNames::parentageBranchName().c_str(), &pParentageBuffer);
        parentageTree->GetEntry(i);
        registry.insertMapped(parentageBuffer);
        orderedParentageIDs.push_back(parentageBuffer.id());
      }
      parentageTree->SetBranchAddress(edm::poolNames::parentageBranchName().c_str(), nullptr);

      TTree* eventMetaTree =
          dynamic_cast<TTree*>(inputFile_->Get(edm::BranchTypeToMetaDataTreeName(edm::InEvent).c_str()));
      if (nullptr == eventMetaTree) {
        eventMetaTree = dynamic_cast<TTree*>(inputFile_->Get(edm::BranchTypeToProductTreeName(edm::InEvent).c_str()));
      }
      if (nullptr == eventMetaTree) {
        std::cerr << "ERROR, no '" << edm::BranchTypeToProductTreeName(edm::InEvent)
                  << "' Tree in file so can not show dependencies\n";
        showDependencies_ = false;
        extendedAncestors_ = false;
        extendedDescendants_ = false;
      } else {
        TBranch* storedProvBranch =
            eventMetaTree->GetBranch(edm::BranchTypeToProductProvenanceBranchName(edm::InEvent).c_str());

        if (nullptr != storedProvBranch) {
          std::vector<edm::StoredProductProvenance> info;
          std::vector<edm::StoredProductProvenance>* pInfo = &info;
          storedProvBranch->SetAddress(&pInfo);
          for (Long64_t i = 0, numEntries = eventMetaTree->GetEntries(); i < numEntries; ++i) {
            storedProvBranch->GetEntry(i);
            for (auto const& item : info) {
              edm::BranchID bid(item.branchID_);
              perProductParentage[bid].insert(orderedParentageIDs.at(item.parentageIDIndex_));
            }
          }
        } else {
          //backwards compatible check
          TBranch* productProvBranch =
              eventMetaTree->GetBranch(edm::BranchTypeToBranchEntryInfoBranchName(edm::InEvent).c_str());
          if (nullptr != productProvBranch) {
            std::vector<edm::ProductProvenance> info;
            std::vector<edm::ProductProvenance>* pInfo = &info;
            productProvBranch->SetAddress(&pInfo);
            for (Long64_t i = 0, numEntries = eventMetaTree->GetEntries(); i < numEntries; ++i) {
              productProvBranch->GetEntry(i);
              for (auto const& item : info) {
                perProductParentage[item.branchID()].insert(item.parentageID());
              }
            }
          } else {
            std::cerr << " ERROR, could not find provenance information so can not show dependencies\n";
            showDependencies_ = false;
            extendedAncestors_ = false;
            extendedDescendants_ = false;
          }
        }
      }
    }
  }

  std::map<edm::BranchID, std::set<edm::BranchID>> parentToChildren;
  edm::ParentageRegistry& registry = *edm::ParentageRegistry::instance();

  if (extendedDescendants_) {
    for (auto const& itParentageSet : perProductParentage) {
      edm::BranchID childBranchID = itParentageSet.first;
      for (auto const& itParentageID : itParentageSet.second) {
        edm::Parentage const* parentage = registry.getMapped(itParentageID);
        if (nullptr != parentage) {
          for (auto const& branch : parentage->parents()) {
            parentToChildren[branch].insert(childBranchID);
          }
        } else {
          std::cerr << "  ERROR:parentage info not in registry ParentageID=" << itParentageID << std::endl;
        }
      }
    }
  }

  ProcessSimpleIDsType const& processSimpleIDs = dumpProcessHistory_();

  std::cout << formatHeader("Event filtering") << std::endl;
  dumpEventFilteringParameterSets_(inputFile_.get(), processSimpleIDs);

  if (productRegistryPresent_) {
    std::cout << formatHeader("Producers with data in file") << std::endl;
  }

  ModuleToIdBranches moduleToIdBranches;
  std::map<edm::BranchID, std::string> branchIDToBranchName;

  for (auto const& processConfig : phc_) {
    edm::ParameterSet const* processParameterSet =
        edm::pset::Registry::instance()->getMapped(processConfig.parameterSetID());
    if (nullptr == processParameterSet || processParameterSet->empty()) {
      continue;
    }
    for (auto& item : reg_.productListUpdator()) {
      auto& product = item.second;
      if (product.processName() != processConfig.processName()) {
        continue;
      }
      //force it to rebuild the branch name
      product.init();
      setIsMergeable(product);

      if (showDependencies_ || extendedAncestors_ || extendedDescendants_) {
        branchIDToBranchName[product.branchID()] = product.branchName();
      }
      /*
        std::cout << product.branchName()
        << " id " << product.productID() << std::endl;
      */
      std::string moduleLabel = product.moduleLabel();
      if (moduleLabel == source) {
        moduleLabel = input;
      } else if (moduleLabel == triggerResults) {
        moduleLabel = triggerPaths;
      }

      std::stringstream s;

      if (processParameterSet->existsAs<edm::ParameterSet>(moduleLabel)) {
        edm::ParameterSet const& moduleParameterSet = processParameterSet->getParameterSet(moduleLabel);
        if (!moduleParameterSet.isRegistered()) {
          edm::ParameterSet moduleParameterSetCopy = processParameterSet->getParameterSet(moduleLabel);
          moduleParameterSetCopy.registerIt();
          s << moduleParameterSetCopy.id();
        } else {
          s << moduleParameterSet.id();
        }
        auto& moduleInformation =
            moduleToIdBranches[std::make_pair(product.processName(), product.moduleLabel())][s.str()];
        moduleInformation.processSimpleIDs.insert(processSimpleIDs.at(processConfig.id()));
        moduleInformation.products.insert(product);
      }
    }
  }

  for (auto const& item : moduleToIdBranches) {
    std::ostringstream sout;
    auto const& processName = item.first.first;
    auto const& moduleLabel = item.first.second;
    sout << "Module: " << moduleLabel << " " << processName << std::endl;
    std::set<edm::BranchID> allBranchIDsForLabelAndProcess;
    IdToBranches const& idToBranches = item.second;
    for (auto const& idBranch : idToBranches) {
      auto const& psetID = idBranch.first;
      auto const& moduleInformation = idBranch.second;

      sout << " Process: " << processName;
      for (auto const& simpleID : moduleInformation.processSimpleIDs) {
        sout << " [" << simpleID << "]";
      }
      sout << std::endl;
      sout << " PSet id:" << psetID << std::endl;
      if (!dontPrintProducts_) {
        sout << " products: {" << std::endl;
      }
      std::set<edm::BranchID> branchIDs;
      for (auto const& branch : moduleInformation.products) {
        if (!dontPrintProducts_) {
          sout << "  " << branch.branchName();
          edm::ProductID id;
          if (branchIDListHelperAndToProcessIndex) {
            sout << " ProductID "
                 << edm::branchIDToProductID(branch.branchID(),
                                             std::get<0>(*branchIDListHelperAndToProcessIndex),
                                             std::get<1>(*branchIDListHelperAndToProcessIndex));
          } else {
            sout << " BranchID " << branch.branchID();
          }
          sout << std::endl;
        }
        branchIDs.insert(branch.branchID());
        allBranchIDsForLabelAndProcess.insert(branch.branchID());
      }
      sout << " }" << std::endl;
      edm::ParameterSetID psid(idBranch.first);
      ParameterSetMap::const_iterator itpsm = psm_.find(psid);
      if (psm_.end() == itpsm) {
        ++errorCount_;
        errorLog_ << "No ParameterSetID for " << psid << std::endl;
        exitCode_ = 1;
      } else {
        sout << " parameters: ";
        prettyPrint(sout, edm::ParameterSet((*itpsm).second.pset()), " ", " ");
        sout << std::endl;
      }
      if (showDependencies_) {
        sout << " dependencies: {" << std::endl;
        std::set<edm::ParentageID> parentageIDs;
        for (auto const& branch : branchIDs) {
          //Save these BranchIDs
          std::set<edm::ParentageID> const& temp = perProductParentage[branch];
          parentageIDs.insert(temp.begin(), temp.end());
        }
        for (auto const& parentID : parentageIDs) {
          edm::Parentage const* parentage = registry.getMapped(parentID);
          if (nullptr != parentage) {
            for (auto const& branch : parentage->parents()) {
              sout << "  " << branchIDToBranchName[branch] << std::endl;
            }
          } else {
            sout << "  ERROR:parentage info not in registry ParentageID=" << parentID << std::endl;
          }
        }
        if (parentageIDs.empty()) {
          sout << "  no dependencies recorded (event may not contain data from this module)" << std::endl;
        }
        sout << " }" << std::endl;
      }
    }  // end loop over PSetIDs
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
    if (matchToFindMatch_(sout.str())) {
      std::cout << sout.str() << std::endl;
    }
  }  // end loop over module label/process

  if (productRegistryPresent_ && showOtherModules_) {
    std::cout << formatHeader("Other Modules") << std::endl;
    dumpOtherModulesHistory_(moduleToIdBranches);
  } else if (!productRegistryPresent_) {
    std::cout << formatHeader("All Modules") << std::endl;
    dumpOtherModulesHistory_(moduleToIdBranches);
  }

  if (!excludeESModules_) {
    std::cout << formatHeader("EventSetup") << std::endl;
    dumpEventSetupHistory_();
  }

  if (showTopLevelPSets_) {
    std::cout << formatHeader("Top Level PSets") << std::endl;
    dumpTopLevelPSetsHistory_();
  }
  if (errorCount_ != 0) {
    exitCode_ = 1;
  }
}

void ProvenanceDumper::addAncestors(edm::BranchID const& branchID,
                                    std::set<edm::BranchID>& ancestorBranchIDs,
                                    std::ostringstream& sout,
                                    std::map<edm::BranchID, std::set<edm::ParentageID>>& perProductParentage) const {
  edm::ParentageRegistry& registry = *edm::ParentageRegistry::instance();

  std::set<edm::ParentageID> const& parentIDs = perProductParentage[branchID];
  for (auto const& parentageID : parentIDs) {
    edm::Parentage const* parentage = registry.getMapped(parentageID);
    if (nullptr != parentage) {
      for (auto const& branch : parentage->parents()) {
        if (ancestorBranchIDs.insert(branch).second) {
          addAncestors(branch, ancestorBranchIDs, sout, perProductParentage);
        }
      }
    } else {
      sout << "  ERROR:parentage info not in registry ParentageID=" << parentageID << std::endl;
    }
  }
}

void ProvenanceDumper::addDescendants(edm::BranchID const& branchID,
                                      std::set<edm::BranchID>& descendantBranchIDs,
                                      std::ostringstream& sout,
                                      std::map<edm::BranchID, std::set<edm::BranchID>>& parentToChildren) const {
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
static char const* const kShowTopLevelPSetsCommandOpt = "showTopLevelPSets,t";
static char const* const kShowHistoryIDOpt = "showHistoryID";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";
static char const* const kFileNameOpt = "input-file";
static char const* const kDumpPSetIDOpt = "dumpPSetID";
static char const* const kDumpPSetIDCommandOpt = "dumpPSetID,i";
static char const* const kProductIDEntryOpt = "productIDEntry";

int main(int argc, char* argv[]) {
  using namespace boost::program_options;

  std::string descString(argv[0]);
  descString += " [options] <filename>";
  descString += "\nAllowed options";
  options_description desc(descString);

  // clang-format off
  desc.add_options()(kHelpCommandOpt, "show help message")(kSortCommandOpt, "alphabetially sort EventSetup components")(
      kDependenciesCommandOpt, "print what data each EDProducer is directly dependent upon")(
      kExtendedAncestorsCommandOpt, "print what data each EDProducer is dependent upon including indirect dependences")(
      kExtendedDescendantsCommandOpt,
      "print what data depends on the data each EDProducer produces including indirect dependences")(
      kExcludeESModulesCommandOpt, "do not print ES module information")(
      kShowAllModulesCommandOpt, "show all modules (not just those that created data in the file)")(
      kShowTopLevelPSetsCommandOpt, "show all top level PSets")
    (kShowHistoryIDOpt, "show process history and configuration IDs")(
      kFindMatchCommandOpt,
      boost::program_options::value<std::vector<std::string>>(),
      "show only modules whose information contains the matching string (or all the matching strings, this option can "
      "be repeated with different strings)")(kDontPrintProductsCommandOpt, "do not print products produced by module")(
      kDumpPSetIDCommandOpt,
      value<std::string>(),
      "print the parameter set associated with the parameter set ID string (and print nothing else)")(
      kProductIDEntryOpt,
      value<int>(),
      "show ProductID instead of BranchID using the specified entry in the Events tree");
  // clang-format on

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
  } catch (error const& iException) {
    std::cerr << iException.what() << std::endl;
    return 1;
  }

  if (vm.count(kHelpOpt)) {
    std::cout << desc << std::endl;
    return 0;
  }

  bool sort = false;
  if (vm.count(kSortOpt)) {
    sort = true;
  }

  bool showDependencies = false;
  if (vm.count(kDependenciesOpt)) {
    showDependencies = true;
  }

  bool extendedAncestors = false;
  if (vm.count(kExtendedAncestorsOpt)) {
    extendedAncestors = true;
  }

  bool extendedDescendants = false;
  if (vm.count(kExtendedDescendantsOpt)) {
    extendedDescendants = true;
  }

  bool excludeESModules = false;
  if (vm.count(kExcludeESModulesOpt)) {
    excludeESModules = true;
  }

  bool showAllModules = false;
  if (vm.count(kShowAllModulesOpt)) {
    showAllModules = true;
  }

  bool showTopLevelPSets = false;
  if (vm.count(kShowTopLevelPSetsOpt)) {
    showTopLevelPSets = true;
  }

  bool showHistoryID = false;
  if (vm.count(kShowHistoryIDOpt)) {
    showHistoryID = true;
  }

  std::string fileName;
  if (vm.count(kFileNameOpt)) {
    try {
      fileName = vm[kFileNameOpt].as<std::string>();
    } catch (boost::bad_any_cast const& e) {
      std::cout << e.what() << std::endl;
      return 2;
    }
  } else {
    std::cout << "Data file not specified." << std::endl;
    std::cout << desc << std::endl;
    return 2;
  }

  std::string dumpPSetID;
  if (vm.count(kDumpPSetIDOpt)) {
    try {
      dumpPSetID = vm[kDumpPSetIDOpt].as<std::string>();
    } catch (boost::bad_any_cast const& e) {
      std::cout << e.what() << std::endl;
      return 2;
    }
  }

  std::vector<std::string> findMatch;
  if (vm.count(kFindMatchOpt)) {
    try {
      findMatch = vm[kFindMatchOpt].as<std::vector<std::string>>();
    } catch (boost::bad_any_cast const& e) {
      std::cout << e.what() << std::endl;
      return 2;
    }
  }

  bool dontPrintProducts = false;
  if (vm.count(kDontPrintProductsOpt)) {
    dontPrintProducts = true;
  }

  int productIDEntry = -1;
  if (vm.count(kProductIDEntryOpt)) {
    try {
      productIDEntry = vm[kProductIDEntryOpt].as<int>();
    } catch (boost::bad_any_cast const& e) {
      std::cout << e.what() << std::endl;
      return 2;
    }
  }

  //silence ROOT warnings about missing dictionaries
  gErrorIgnoreLevel = kError;

  ProvenanceDumper dumper(fileName,
                          showDependencies,
                          extendedAncestors,
                          extendedDescendants,
                          excludeESModules,
                          showAllModules,
                          showTopLevelPSets,
                          showHistoryID,
                          findMatch,
                          dontPrintProducts,
                          dumpPSetID,
                          productIDEntry,
                          sort);
  int exitCode(0);
  try {
    dumper.dump();
    exitCode = dumper.exitCode();
  } catch (cms::Exception const& x) {
    std::cerr << "cms::Exception caught\n";
    std::cerr << x.what() << '\n';
    exitCode = 2;
  } catch (std::exception& x) {
    std::cerr << "std::exception caught\n";
    std::cerr << x.what() << '\n';
    exitCode = 3;
  } catch (...) {
    std::cerr << "Unknown exception caught\n";
    exitCode = 4;
  }

  dumper.printErrors(std::cerr);
  return exitCode;
}
