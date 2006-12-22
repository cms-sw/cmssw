// Last commit: $Id: SealModule.cc,v 1.7 2006/12/22 12:23:40 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/test/stubs/SealModule.cc,v $

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PluginManager/ModuleDef.h"
DEFINE_SEAL_MODULE();

#include "OnlineDB/SiStripESSources/test/stubs/test_FedCablingBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(test_FedCablingBuilder);

#include "OnlineDB/SiStripESSources/test/stubs/test_PedestalsBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(test_PedestalsBuilder);

#include "OnlineDB/SiStripESSources/test/stubs/test_NoiseBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(test_NoiseBuilder);
