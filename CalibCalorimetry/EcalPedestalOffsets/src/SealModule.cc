/*
 *  \author  P. Govoni Univ Milano Bicocca - INFN Milano
 */
 
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
 
#include "CalibCalorimetry/EcalPedestalOffsets/interface/EBPedOffset.h"

//DEFINE_SEAL_MODULE () ;
DEFINE_FWK_MODULE (EBPedOffset) ;
//DEFINE_ANOTHER_FWK_MODULE (EBPedOffset) ;

#include "CalibCalorimetry/EcalPedestalOffsets/interface/testChannel.h"

DEFINE_ANOTHER_FWK_MODULE (testChannel) ;
 
