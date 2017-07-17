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
  
  std::string const& BranchTypeToProductProvenanceBranchName(BranchType const& BranchType);

  std::string const& BranchTypeToMajorIndexName(BranchType const& branchType);

  std::string const& BranchTypeToMinorIndexName(BranchType const& branchType);

  std::ostream&
  operator<<(std::ostream& os, BranchType const& branchType);

  namespace poolNames {
    //------------------------------------------------------------------
    // EntryDescription Tree // backward compatibility
    std::string const& entryDescriptionTreeName(); // backward compatibility

    // Branches on EntryDescription Tree // backward compatibility
    std::string const& entryDescriptionIDBranchName(); // backward compatibility
    std::string const& entryDescriptionBranchName(); // backward compatibility

    //------------------------------------------------------------------
    // Parentage Tree
    std::string const& parentageTreeName();

    // Branches on parentage tree
    std::string const& parentageBranchName();

    //------------------------------------------------------------------
    // Other branches on Events Tree
    std::string const& eventSelectionsBranchName();
    std::string const& branchListIndexesBranchName();

    //------------------------------------------------------------------
    //------------------------------------------------------------------
    // MetaData Tree (1 entry per file)
    std::string const& metaDataTreeName();

    // Branches on MetaData Tree
    std::string const& productDescriptionBranchName();
    std::string const& productDependenciesBranchName();
    std::string const& parameterSetMapBranchName(); // backward compatibility
    std::string const& moduleDescriptionMapBranchName(); // backward compatibility
    std::string const& processHistoryMapBranchName(); // backward compatibility
    std::string const& processHistoryBranchName();
    std::string const& processConfigurationBranchName();
    std::string const& branchIDListBranchName();
    std::string const& thinnedAssociationsHelperBranchName();
    std::string const& fileFormatVersionBranchName();
    std::string const& fileIdentifierBranchName();
    std::string const& fileIndexBranchName(); // backward compatibility
    std::string const& indexIntoFileBranchName();

    // Event History Tree // backward compatibility
    std::string const& eventHistoryTreeName(); // backward compatibility

    // Branches on EventHistory Tree // backward compatibility
    std::string const& eventHistoryBranchName(); // backward compatibility

    //------------------------------------------------------------------
    // ParameterSet Tree (1 entry per ParameterSet)
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
