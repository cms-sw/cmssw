#ifndef Utilities_PersistentNames_h
#define Utilities_PersistentNames_h
//////////////////////////////////////////////////////////////////////
//
// $Id: PersistentNames.h,v 1.2 2006/03/10 17:57:49 wmtan Exp $
//
// Functions defining tree, branch, and container names.
// Namespace rootNames. Defined names of ROOT trees and branches
//
// Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

namespace edm {

  namespace poolNames {

    inline
    std::string containerName(std::string const& tree, std::string const& branch) {
      return tree + "(" + branch + ")";
    }

    inline
    std::string
    eventTreeName() { return "Events"; }

    inline
    std::string
    provenanceBranchName() { return "Provenance"; }

    inline
    std::string auxiliaryBranchName() { return "EventAux"; }

    inline
    std::string
    metaDataTreeName() { return "MetaData"; }

    inline
    std::string
    productDescriptionBranchName() { return "ProductRegistry"; }

    inline
    std::string
    parameterSetTreeName() { return "ParameterSets"; }

    inline
    std::string
    parameterSetIDBranchName() { return "ParameterSetID"; }

    inline
    std::string
    parameterSetBranchName() { return "ParameterSet"; }

  }
}
#endif
