#ifndef DataFormats_Provenance_BranchType_h
#define DataFormats_Provenance_BranchType_h

#include <string>
#include <iosfwd>
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Utilities/interface/BranchType.h"

namespace edm {
  std::string const& BranchTypeToString(BranchType const& branchType);

  std::string const& BranchTypeToProductTreeName(BranchType const& branchType);

  std::string const& BranchTypeToMetaDataTreeName(BranchType const& branchType);

  std::string const& BranchTypeToInfoTreeName(BranchType const& branchType); // backward compatibility

  std::string const& BranchTypeToAuxiliaryBranchName(BranchType const& branchType);

  std::string const& BranchTypeToAuxBranchName(BranchType const& branchType); // backward compatibility

  std::string const& BranchTypeToProductStatusBranchName(BranchType const& branchType); // backward compatibility

  std::string const& BranchTypeToBranchEntryInfoBranchName(BranchType const& branchType);

  std::string const& BranchTypeToMajorIndexName(BranchType const& branchType);

  std::string const& BranchTypeToMinorIndexName(BranchType const& branchType);

  std::ostream&
  operator<<(std::ostream& os, BranchType const& branchType);

  namespace poolNames {
    //------------------------------------------------------------------
    // EntryDescription Tree // Obsolete
    std::string const& entryDescriptionTreeName();

    // Branches on EntryDescription Tree // Obsolete
    std::string const& entryDescriptionIDBranchName();
    std::string const& entryDescriptionBranchName();

    //------------------------------------------------------------------
    // Parentage Tree
    std::string const& parentageTreeName();

    // Branches on parentage tree
    std::string const& parentageBranchName();

    //------------------------------------------------------------------
    //------------------------------------------------------------------
    // MetaData Tree (1 entry per file)
    std::string const& metaDataTreeName();

    // Branches on MetaData Tree
    std::string const& productDescriptionBranchName();
    std::string const& productDependenciesBranchName();
    std::string const& parameterSetMapBranchName();
    std::string const& moduleDescriptionMapBranchName(); // Obsolete
    std::string const& processHistoryMapBranchName(); // Obsolete
    std::string const& processHistoryBranchName();
    std::string const& processConfigurationBranchName();
    std::string const& branchIDListBranchName();
    std::string const& fileFormatVersionBranchName();
    std::string const& fileIdentifierBranchName();
    std::string const& fileIndexBranchName();

    // Event History Tree
    std::string const& eventHistoryTreeName();

    // Branches on EventHistory Tree
    std::string const& eventHistoryBranchName();

    //------------------------------------------------------------------
    // ParameterSet Tree (1 entry per ParameterSet
    std::string const& parameterSetsTreeName();
    
    std::string const& idToParameterSetBlobsBranchName();
    
    //------------------------------------------------------------------
    // Other tree names
    std::string const& runTreeName();
    std::string const& luminosityBlockTreeName();
    std::string const& eventTreeName();
    std::string const& eventMetaDataTreeName();
  }
}
#endif
