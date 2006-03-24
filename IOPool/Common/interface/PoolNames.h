#ifndef Common_PoolNames_h
#define Common_PoolNames_h
//////////////////////////////////////////////////////////////////////
//
// $Id: PoolNames.h,v 1.1 2005/11/01 22:42:45 wmtan Exp $
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
