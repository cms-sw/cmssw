// -*- C++ -*-
//
// Package:     Framework
// Class  :     edmodule_mightGet_config
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu Feb  2 14:26:42 CST 2012
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

static const std::string kMightGet("mightGet");
static const char* const kComment=
"List contains the branch names for the EDProducts which might be requested by the module.\n"
"The format for identifying the EDProduct is the same as the one used for OutputModules, "
"except no wild cards are allowed. E.g.\n"
"Foos_foomodule_whichFoo_RECO";

namespace edm {
  void edmodule_mightGet_config(ConfigurationDescriptions& iDesc) {
    //NOTE: by not giving a default, we are intentionally not having 'mightGet' added
    // to any cfi files. This was done intentionally to avoid problems with HLT. If requested,
    // the appropriate default would be an empty vector.
    if(iDesc.defaultDescription()) {
      if (iDesc.defaultDescription()->isLabelUnused(kMightGet)) {
        iDesc.defaultDescription()->addOptionalUntracked<std::vector<std::string> >(kMightGet)
        ->setComment(kComment);
      }
    }
    for(auto& v: iDesc) {
      if (v.second.isLabelUnused(kMightGet)) {
        v.second.addOptionalUntracked<std::vector<std::string> >(kMightGet)->setComment(kComment);
      }
    }
    
  }
}
