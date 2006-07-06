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
// $Id: ProvenanceAccess.cc,v 1.2.2.1 2006/07/01 06:23:37 wmtan Exp $
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
   const Provenance& 
   ProvenanceAccess::provenance() const
   {
      return *prov_;
   }
//
// static member functions
//
}
