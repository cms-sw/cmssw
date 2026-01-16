#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

// user include files
#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"
typedef SimpleFlatTableProducer<Run3ScoutingPhoton> HLTPhotonTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPhotonTableProducer);
