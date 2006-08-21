#ifndef Utilities_PersistentNames_h
#define Utilities_PersistentNames_h
//////////////////////////////////////////////////////////////////////
//
// $Id: PersistentNames.h,v 1.4 2006/07/11 22:40:00 wmtan Exp $
//
// Functions defining tree, branch, and container names.
// Namespace rootNames. Defined names of ROOT trees and branches
//
// Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <string>
namespace edm {

  namespace poolNames {

    inline
    std::string containerName(std::string const& tree, std::string const& branch) {
      return tree + "(" + branch + ")";
    }

    // Event Tree (1 entry per event)
    // One branch per EDProduct
    inline
    std::string
    eventTreeName() { return "Events"; }

    // Extra Branch on Event Tree
    inline
    std::string auxiliaryBranchName() { return "EventAux"; }

    // Event MetaData Tree (1 entry per event)
    // One branch per EDProduct
    inline
    std::string
    eventMetaDataTreeName() { return "EventMetaData"; }

    // MetaData Tree (1 entry per file)
    inline
    std::string
    metaDataTreeName() { return "MetaData"; }

    // Branch on MetaData Tree
    inline
    std::string
    productDescriptionBranchName() { return "ProductRegistry"; }

    // Branch on MetaData Tree
    inline
    std::string
    parameterSetMapBranchName() { return "ParameterSetMap"; }

    // Branch on MetaData Tree
    inline
    std::string
    moduleDescriptionMapBranchName() { return "ModuleDescriptionMap"; }

    // Branch on MetaData Tree
    inline
    std::string
    processHistoryMapBranchName() { return "ProcessHistoryMap"; }

    // Branch on MetaData Tree
    inline
    std::string
    fileFormatVersionBranchName() { return "FileFormatVersion"; }

    // Run Tree (1 entry per run)
    inline
    std::string
    runTreeName() { return "Runs"; }

    // Branch on Run Tree
    inline
    std::string
    runBranchName() { return "RunInformation"; }

    // Luminosity Block Tree (1 entry per luminosity block)
    inline
    std::string
    luminosityBlockTreeName() { return "LuminosityBlocks"; }

    // Branch on Luminosity Block Tree
    inline
    std::string
    luminosityBlockBranchName() { return "LuminosityInformation"; }

// Obsolete.  Kept for backward compatibility and conversion
    inline
    std::string
    parameterSetTreeName() { return "ParameterSets"; }

    inline
    std::string
    parameterSetIDBranchName() { return "ParameterSetID"; }

    inline
    std::string
    parameterSetBranchName() { return "ParameterSet"; }

    inline
    std::string
    provenanceBranchName() { return "Provenance"; }

  }
}
#endif
