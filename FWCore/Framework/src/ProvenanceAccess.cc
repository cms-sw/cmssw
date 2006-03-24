// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProvenanceAccess
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Feb 15 11:49:59 IST 2006
// $Id: ProvenanceAccess.cc,v 1.1 2006/03/05 21:40:22 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/ProvenanceAccess.h"

namespace edm {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ProvenanceAccess::ProvenanceAccess(Provenance* iProv,
                                   const EventProvenanceFiller* iFiller):
prov_(iProv), filler_(iFiller), haveFilled_(false)
{
   assert(0!=prov_);
}

//
// member functions
//

//
// const member functions
//
   const BranchDescription& 
   ProvenanceAccess::product() const
   {
      return prov_->product;
   }
   const BranchEntryDescription& 
      ProvenanceAccess::event() const
   {
      return prov_->event;
   }

//
// static member functions
//
}
