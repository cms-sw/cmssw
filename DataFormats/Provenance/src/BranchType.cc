#include "DataFormats/Provenance/interface/BranchType.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>

namespace edm {
  std::ostream& operator<<(std::ostream& os, BranchType const& branchType) {
    os << BranchTypeToString(branchType);
    return os;
  }
  namespace {

    // Suffixes
    std::string const metaData = "MetaData";
    std::string const auxiliary = "Auxiliary";
    std::string const aux = "Aux";  // backward compatibility
    std::string const productStatus = "ProductStatus";
    std::string const branchEntryInfo = "BranchEntryInfo";  // backward compatibility
    std::string const productProvenance = "ProductProvenance";

    std::array<std::string, NumBranchTypes> const branchTypeNames{{"Event", "LuminosityBlock", "Run", "ProcessBlock"}};
    std::size_t constexpr eventLumiRunSize = 3;
    using NameArray = std::array<std::string, eventLumiRunSize>;

    NameArray makeNameArray(std::string const& postfix) {
      static_assert(InEvent == 0);
      static_assert(InLumi == 1);
      static_assert(InRun == 2);
      static_assert(InProcess == 3);
      NameArray ret;
      for (auto i = 0U; i != eventLumiRunSize; ++i) {
        ret[i] = branchTypeNames[i] + postfix;
      }
      return ret;
    }

    const NameArray treeNames{makeNameArray(std::string("s"))};

    const NameArray metaTreeNames{makeNameArray(metaData)};

    // backward compatibility
    const NameArray infoNames{makeNameArray(std::string("StatusInformation"))};

    const NameArray auxiliaryNames{makeNameArray(auxiliary)};

    // backward compatibility
    const NameArray productStatusNames{makeNameArray(productStatus)};

    // backward compatibility
    const NameArray eventEntryInfoNames{makeNameArray(branchEntryInfo)};

    const NameArray productProvenanceNames{makeNameArray(productProvenance)};

    // backward compatibility
    const NameArray auxNames{makeNameArray(aux)};

    std::string const entryDescriptionTree = "EntryDescription";
    std::string const entryDescriptionIDBranch = "Hash";
    std::string const entryDescriptionBranch = "Description";

    std::string const parentageTree = "Parentage";
    std::string const parentageBranch = "Description";

    std::string const metaDataTree = "MetaData";
    std::string const productRegistry = "ProductRegistry";
    std::string const productDependencies = "ProductDependencies";
    std::string const parameterSetMap = "ParameterSetMap";
    std::string const moduleDescriptionMap = "ModuleDescriptionMap";  // Obsolete
    std::string const processHistoryMap = "ProcessHistoryMap";        // Obsolete
    std::string const processHistory = "ProcessHistory";
    std::string const processConfiguration = "ProcessConfiguration";
    std::string const branchIDLists = "BranchIDLists";
    std::string const thinnedAssociationsHelper = "ThinnedAssociationsHelper";
    std::string const fileFormatVersion = "FileFormatVersion";
    std::string const fileIdentifier = "FileIdentifier";
    std::string const fileIndex = "FileIndex";
    std::string const indexIntoFile = "IndexIntoFile";
    std::string const mergeableRunProductMetadata = "MergeableRunProductMetadata";
    std::string const processBlockHelper = "ProcessBlockHelper";
    std::string const eventHistory = "EventHistory";
    std::string const eventBranchMapper = "EventBranchMapper";

    std::string const eventSelections = "EventSelections";
    std::string const branchListIndexes = "BranchListIndexes";
    std::string const eventToProcessBlockIndexes = "EventToProcessBlockIndexes";

    std::string const parameterSetsTree = "ParameterSets";
    std::string const idToParameterSetBlobsBranch = "IdToParameterSetsBlobs";
  }  // namespace

  std::string const& BranchTypeToString(BranchType const& branchType) { return branchTypeNames[branchType]; }

  std::string const& BranchTypeToProductTreeName(BranchType const& branchType) {
    assert(branchType < eventLumiRunSize);
    return treeNames[branchType];
  }

  std::string BranchTypeToProductTreeName(BranchType const& branchType, std::string const& processName) {
    assert(branchType == InProcess);
    return branchTypeNames[InProcess] + "s" + processName;
  }

  std::string const& BranchTypeToMetaDataTreeName(BranchType const& branchType) {
    assert(branchType < eventLumiRunSize);
    return metaTreeNames[branchType];
  }

  // backward compatibility
  std::string const& BranchTypeToInfoTreeName(BranchType const& branchType) {
    assert(branchType < eventLumiRunSize);
    return infoNames[branchType];
  }

  std::string const& BranchTypeToAuxiliaryBranchName(BranchType const& branchType) {
    assert(branchType < eventLumiRunSize);
    return auxiliaryNames[branchType];
  }

  // backward compatibility
  std::string const& BranchTypeToAuxBranchName(BranchType const& branchType) {
    assert(branchType < eventLumiRunSize);
    return auxNames[branchType];
  }

  // backward compatibility
  std::string const& BranchTypeToProductStatusBranchName(BranchType const& branchType) {
    assert(branchType < eventLumiRunSize);
    return productStatusNames[branchType];
  }

  // backward compatibility
  std::string const& BranchTypeToBranchEntryInfoBranchName(BranchType const& branchType) {
    assert(branchType < eventLumiRunSize);
    return eventEntryInfoNames[branchType];
  }

  std::string const& BranchTypeToProductProvenanceBranchName(BranchType const& branchType) {
    assert(branchType < eventLumiRunSize);
    return productProvenanceNames[branchType];
  }

  namespace poolNames {

    // EntryDescription tree (1 entry per recorded distinct value of EntryDescription)
    std::string const& entryDescriptionTreeName() { return entryDescriptionTree; }

    std::string const& entryDescriptionIDBranchName() { return entryDescriptionIDBranch; }

    std::string const& entryDescriptionBranchName() { return entryDescriptionBranch; }

    // EntryDescription tree (1 entry per recorded distinct value of EntryDescription)
    std::string const& parentageTreeName() { return parentageTree; }

    std::string const& parentageBranchName() { return parentageBranch; }

    // MetaData Tree (1 entry per file)
    std::string const& metaDataTreeName() { return metaDataTree; }

    // Branch on MetaData Tree
    std::string const& productDescriptionBranchName() { return productRegistry; }

    // Branch on MetaData Tree
    std::string const& productDependenciesBranchName() { return productDependencies; }

    // Branch on MetaData Tree
    std::string const& parameterSetMapBranchName() { return parameterSetMap; }

    // Branch on MetaData Tree // Obsolete
    std::string const& moduleDescriptionMapBranchName() { return moduleDescriptionMap; }

    // Branch on MetaData Tree // Obsolete
    std::string const& processHistoryMapBranchName() { return processHistoryMap; }

    // Branch on MetaData Tree
    std::string const& processHistoryBranchName() { return processHistory; }

    // Branch on MetaData Tree
    std::string const& processConfigurationBranchName() { return processConfiguration; }

    // Branch on MetaData Tree
    std::string const& branchIDListBranchName() { return branchIDLists; }

    // Branch on MetaData Tree
    std::string const& thinnedAssociationsHelperBranchName() { return thinnedAssociationsHelper; }

    // Branch on MetaData Tree
    std::string const& fileFormatVersionBranchName() { return fileFormatVersion; }

    // Branch on MetaData Tree
    std::string const& fileIdentifierBranchName() { return fileIdentifier; }

    // Branch on MetaData Tree
    std::string const& fileIndexBranchName() { return fileIndex; }

    // Branch on MetaData Tree
    std::string const& indexIntoFileBranchName() { return indexIntoFile; }

    // Branch on MetaData Tree
    std::string const& mergeableRunProductMetadataBranchName() { return mergeableRunProductMetadata; }

    // Branch on MetaData Tree
    std::string const& processBlockHelperBranchName() { return processBlockHelper; }

    // Branch on Event History Tree
    std::string const& eventHistoryBranchName() { return eventHistory; }

    // Branches on Events Tree
    std::string const& eventSelectionsBranchName() { return eventSelections; }

    std::string const& branchListIndexesBranchName() { return branchListIndexes; }

    std::string const& eventToProcessBlockIndexesBranchName() { return eventToProcessBlockIndexes; }

    std::string const& parameterSetsTreeName() { return parameterSetsTree; }
    // Branch on ParameterSets Tree
    std::string const& idToParameterSetBlobsBranchName() { return idToParameterSetBlobsBranch; }

    std::string const& eventTreeName() { return treeNames[InEvent]; }

    std::string const& eventMetaDataTreeName() { return metaTreeNames[InEvent]; }

    std::string const& eventHistoryTreeName() { return eventHistory; }
    std::string const& luminosityBlockTreeName() { return treeNames[InLumi]; }
    std::string const& runTreeName() { return treeNames[InRun]; }
  }  // namespace poolNames
}  // namespace edm
