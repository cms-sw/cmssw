#include "DataFormats/Provenance/interface/BranchType.h"

namespace edm {
  namespace {
    // Suffixes
    std::string const metaData = "MetaData";
    std::string const auxiliary = "Auxiliary";
    std::string const aux = "Aux"; // backward compatibility
    std::string const productStatus = "ProductStatus";
    std::string const branchEntryInfo = "BranchEntryInfo";

    // Prefixes
    std::string const run = "Run";
    std::string const lumi = "LuminosityBlock";
    std::string const event = "Event";

    // Trees, branches, indices
    std::string const runs = run + 's';
    std::string const lumis = lumi + 's';
    std::string const events = event + 's';

    std::string const runMeta = run + metaData;
    std::string const lumiMeta = lumi + metaData;
    std::string const eventMeta = event + metaData;

    std::string const runInfo = run + "StatusInformation"; // backward compatibility
    std::string const lumiInfo = lumi + "StatusInformation"; // backward compatibility
    std::string const eventInfo = event + "StatusInformation"; // backward compatibility

    std::string const runAuxiliary = run + auxiliary;
    std::string const lumiAuxiliary = lumi + auxiliary;
    std::string const eventAuxiliary = event + auxiliary;

    std::string const runProductStatus = run + productStatus; // backward compatibility
    std::string const lumiProductStatus = lumi + productStatus; // backward compatibility
    std::string const eventProductStatus = event + productStatus; // backward compatibility

    std::string const runEventEntryInfo = run + branchEntryInfo;
    std::string const lumiEventEntryInfo = lumi + branchEntryInfo;
    std::string const eventEventEntryInfo = event + branchEntryInfo;

    std::string const majorIndex = ".id_.run_";
    std::string const runMajorIndex = runAuxiliary + majorIndex;
    std::string const lumiMajorIndex = lumiAuxiliary + majorIndex;
    std::string const eventMajorIndex = eventAuxiliary + majorIndex;

    std::string const runMinorIndex; // empty
    std::string const lumiMinorIndex = lumiAuxiliary + ".id_.luminosityBlock_";
    std::string const eventMinorIndex = eventAuxiliary + ".id_.event_";

    std::string const runAux = run + aux;
    std::string const lumiAux = lumi + aux;
    std::string const eventAux = event + aux;

    std::string const entryDescriptionTree     = "EntryDescription";
    std::string const entryDescriptionIDBranch = "Hash";
    std::string const entryDescriptionBranch   = "Description";

    std::string const parentageTree     = "Parentage";
    std::string const parentageBranch   = "Description";

    std::string const metaDataTree = "MetaData";
    std::string const productRegistry = "ProductRegistry";
    std::string const productDependencies = "ProductDependencies";
    std::string const parameterSetMap = "ParameterSetMap";
    std::string const moduleDescriptionMap = "ModuleDescriptionMap"; // Obsolete
    std::string const processHistoryMap = "ProcessHistoryMap"; // Obsolete
    std::string const processHistory = "ProcessHistory";
    std::string const processConfiguration = "ProcessConfiguration";
    std::string const branchIDLists = "BranchIDLists";
    std::string const fileFormatVersion = "FileFormatVersion";
    std::string const fileIdentifier = "FileIdentifier";
    std::string const fileIndex = "FileIndex";
    std::string const eventHistory = "EventHistory";
    std::string const eventBranchMapper = "EventBranchMapper";
  }

  std::string const& BranchTypeToString(BranchType const& branchType) {
    return ((branchType == InEvent) ? event : ((branchType == InRun) ? run : lumi));
  }

  std::string const& BranchTypeToProductTreeName(BranchType const& branchType) {
    return ((branchType == InEvent) ? events : ((branchType == InRun) ? runs : lumis));
  }

  std::string const& BranchTypeToMetaDataTreeName(BranchType const& branchType) {
    return ((branchType == InEvent) ? eventMeta : ((branchType == InRun) ? runMeta : lumiMeta));
  }

  std::string const& BranchTypeToInfoTreeName(BranchType const& branchType) { // backward compatibility
    return ((branchType == InEvent) ? eventInfo : ((branchType == InRun) ? runInfo : lumiInfo)); // backward compatibility
  } // backward compatibility

  std::string const& BranchTypeToAuxiliaryBranchName(BranchType const& branchType) {
    return ((branchType == InEvent) ? eventAuxiliary : ((branchType == InRun) ? runAuxiliary : lumiAuxiliary));
  }

  std::string const& BranchTypeToAuxBranchName(BranchType const& branchType) { // backward compatibility
    return ((branchType == InEvent) ? eventAux : ((branchType == InRun) ? runAux : lumiAux)); // backward compatibility
  } // backward compatibility

  std::string const& BranchTypeToProductStatusBranchName(BranchType const& branchType) { // backward compatibility
    return ((branchType == InEvent) ? eventProductStatus : ((branchType == InRun) ? runProductStatus : lumiProductStatus)); // backward compatibility
  } // backward compatibility

  std::string const& BranchTypeToBranchEntryInfoBranchName(BranchType const& branchType) {
    return ((branchType == InEvent) ? eventEventEntryInfo : ((branchType == InRun) ? runEventEntryInfo : lumiEventEntryInfo));
  }

  std::string const& BranchTypeToMajorIndexName(BranchType const& branchType) {
    return ((branchType == InEvent) ? eventMajorIndex : ((branchType == InRun) ? runMajorIndex : lumiMajorIndex));
  }

  std::string const& BranchTypeToMinorIndexName(BranchType const& branchType) {
    return ((branchType == InEvent) ? eventMinorIndex : ((branchType == InRun) ? runMinorIndex : lumiMinorIndex));
  }

  namespace poolNames {

    // EntryDescription tree (1 entry per recorded distinct value of EntryDescription)
    std::string const& entryDescriptionTreeName() {
      return entryDescriptionTree;
    }

    std::string const& entryDescriptionIDBranchName() {
      return entryDescriptionIDBranch;
    }

    std::string const& entryDescriptionBranchName() {
      return entryDescriptionBranch;
    }

    // EntryDescription tree (1 entry per recorded distinct value of EntryDescription)
    std::string const& parentageTreeName() {
      return parentageTree;
    }

    std::string const& parentageBranchName() {
      return parentageBranch;
    }

    // MetaData Tree (1 entry per file)
    std::string const& metaDataTreeName() {
      return metaDataTree;
    }

    // Branch on MetaData Tree
    std::string const& productDescriptionBranchName() {
      return productRegistry;
    }

    // Branch on MetaData Tree
    std::string const& productDependenciesBranchName() {
      return productDependencies;
    }

    // Branch on MetaData Tree
    std::string const& parameterSetMapBranchName() {
      return parameterSetMap;
    }

    // Branch on MetaData Tree // Obsolete
    std::string const& moduleDescriptionMapBranchName() {
      return moduleDescriptionMap;
    }

    // Branch on MetaData Tree // Obsolete
    std::string const& processHistoryMapBranchName() {
      return processHistoryMap;
    }

    // Branch on MetaData Tree
    std::string const& processHistoryBranchName() {
      return processHistory;
    }

    // Branch on MetaData Tree
    std::string const& processConfigurationBranchName() {
      return processConfiguration;
    }

    // Branch on MetaData Tree
    std::string const& branchIDListBranchName() {
      return branchIDLists;
    }

    // Branch on MetaData Tree
    std::string const& fileFormatVersionBranchName() {
      return fileFormatVersion;
    }

    // Branch on MetaData Tree
    std::string const& fileIdentifierBranchName() {
      return fileIdentifier;
    }

    // Branch on MetaData Tree
    std::string const& fileIndexBranchName() {
      return fileIndex;
    }

    // Branch on Event History Tree
    std::string const& eventHistoryBranchName() {
      return eventHistory;
    }

    std::string const& eventTreeName() {
      return events;
    }

    std::string const& eventMetaDataTreeName() {
      return eventMeta;
    }

    std::string const& eventHistoryTreeName() {
      return eventHistory;
    }
  }
}
