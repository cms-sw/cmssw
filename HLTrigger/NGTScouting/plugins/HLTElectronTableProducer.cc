#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

// user include files
#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
typedef SimpleFlatTableProducer<Run3ScoutingElectron> HLTElectronTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTElectronTableProducer);
