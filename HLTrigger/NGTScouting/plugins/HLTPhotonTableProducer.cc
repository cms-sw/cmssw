#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"
typedef SimpleFlatTableProducer<Run3ScoutingPhoton> HLTPhotonTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPhotonTableProducer);
