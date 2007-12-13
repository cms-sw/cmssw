#ifndef DataFormats_Provenance_BranchType_h
#define DataFormats_Provenance_BranchType_h

#include <string>
/*----------------------------------------------------------------------
  
BranchType: The type of a Branch (Event, LuminosityBlock, or Run)

$Id: BranchType.h,v 1.7 2007/12/07 23:22:56 wmtan Exp $
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

  std::string const& BranchTypeToAuxiliaryBranchName(BranchType const& branchType);

  std::string const& BranchTypeToAuxBranchName(BranchType const& branchType);

  std::string const& BranchTypeToMajorIndexName(BranchType const& branchType);

  std::string const& BranchTypeToMinorIndexName(BranchType const& branchType);

  inline
  std::ostream&
  operator<<(std::ostream& os, BranchType const& branchType) {
    os << BranchTypeToString(branchType);
    return os;
  }

  namespace poolNames {
    // MetaData Tree (1 entry per file)
    std::string const& metaDataTreeName();

    // Branch on MetaData Tree
    std::string const& productDescriptionBranchName();

    // Branch on MetaData Tree
    std::string const& parameterSetMapBranchName();

    // Branch on MetaData Tree
    std::string const& moduleDescriptionMapBranchName();

    // Branch on MetaData Tree
    std::string const& processHistoryMapBranchName();

    // Branch on MetaData Tree
    std::string const& processConfigurationMapBranchName();

    // Branch on MetaData Tree
    std::string const& fileFormatVersionBranchName();

    // Branch on MetaData Tree
    std::string const& fileIdentifierBranchName();

    // Branch on MetaData Tree
    std::string const& fileIndexBranchName();

    // Branch on MetaData Tree
    std::string const& eventHistoryBranchName();

    std::string const& eventTreeName();

    std::string const& eventMetaDataTreeName();
  }
}
#endif
