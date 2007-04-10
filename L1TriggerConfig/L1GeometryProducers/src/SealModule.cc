// -*- C++ -*-
//
// Package:     L1GeometryProducers
// Class  :     SealModule
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Werner Sun
//         Created:  Tue Oct 24 00:20:38 EDT 2006
// $Id: SealModule.cc,v 1.1 2006/12/21 01:55:37 wsun Exp $
//

// system include files

// user include files
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1TriggerConfig/L1GeometryProducers/interface/L1CaloGeometryProd.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1CaloGeometryProd);

//DEFINE_ANOTHER_FWK_MODULE(L1ScalesTester)
