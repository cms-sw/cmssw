// Last commit: $Id: $
// Latest tag:  $Name:  $
// Location:    $Source: $

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PluginManager/ModuleDef.h"
DEFINE_SEAL_MODULE();

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripConfigDb);
