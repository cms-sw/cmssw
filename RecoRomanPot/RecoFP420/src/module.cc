#include "RecoRomanPot/RecoFP420/interface/ClusterizerFP420.h"
#include "RecoRomanPot/RecoFP420/interface/TrackerizerFP420.h"
#include "RecoRomanPot/RecoFP420/interface/RecFP420Test.h"

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "PluginManager/ModuleDef.h"
  
DEFINE_SEAL_MODULE ();
DEFINE_SIMWATCHER (ClusterizerFP420); 
DEFINE_SIMWATCHER (TrackerizerFP420); 
DEFINE_SIMWATCHER (RecFP420Test); 
