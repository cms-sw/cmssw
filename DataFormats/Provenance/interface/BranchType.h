#ifndef DataFormats_Provenance_BranchType_h
#define DataFormats_Provenance_BranchType_h

#include <string>
/*----------------------------------------------------------------------
  
BranchType: The type of a Branch (Event, LuminosityBlock, or Run)

----------------------------------------------------------------------*/

namespace edm {
  // Note: These enum values are used as subscripts for a fixed size array, so they must not change.
  enum BranchType {
    InEvent = 0,
    InLumi = 1,
    InRun = 2,
    NumBranchTypes
  };

  std::string const& BranchTypeToString(BranchType const& branchType);

  std::string const& BranchTypeToProductTreeName(BranchType const& branchType);

  std::string const& BranchTypeToMetaDataTreeName(BranchType const& branchType);

  std::string const& BranchTypeToInfoTreeName(BranchType const& branchType);

  std::string const& BranchTypeToAuxiliaryBranchName(BranchType const& branchType);

  std::string const& BranchTypeToAuxBranchName(BranchType const& branchType);

  std::string const& BranchTypeToProductStatusBranchName(BranchType const& branchType);

  std::string const& BranchTypeToBranchEntryInfoBranchName(BranchType const& branchType);

  std::string const& BranchTypeToMajorIndexName(BranchType const& branchType);

  std::string const& BranchTypeToMinorIndexName(BranchType const& branchType);

  inline
  std::ostream&
  operator<<(std::ostream& os, BranchType const& branchType) {
    os << BranchTypeToString(branchType);
    return os;
  }

  namespace poolNames {
    //------------------------------------------------------------------
    // EntryDescription Tree
    std::string const& entryDescriptionTreeName();

    // Branches on EntryDescription Tree
    std::string const& entryDescriptionIDBranchName();
    std::string const& entryDescriptionBranchName();

    //------------------------------------------------------------------
    // MetaData Tree (1 entry per file)
    std::string const& metaDataTreeName();

    // Branches on MetaData Tree
    std::string const& productDescriptionBranchName();
    std::string const& productDependenciesBranchName();
    std::string const& parameterSetMapBranchName();
    std::string const& moduleDescriptionMapBranchName();
    std::string const& processHistoryMapBranchName();
    std::string const& processConfigurationMapBranchName();
    std::string const& fileFormatVersionBranchName();
    std::string const& fileIdentifierBranchName();
    std::string const& fileIndexBranchName();

    // Event History Tree
    std::string const& eventHistoryTreeName();

    // Branches on EventHistory Tree
    std::string const& eventHistoryBranchName();

    //------------------------------------------------------------------
    // Other tree names
    std::string const& eventTreeName();
    std::string const& eventMetaDataTreeName();
  }
}
#endif
