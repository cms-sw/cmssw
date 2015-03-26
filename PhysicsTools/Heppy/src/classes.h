#include "PhysicsTools/Heppy/interface/BTagSF.h"
#include "PhysicsTools/Heppy/interface/RochCor.h"
#include "PhysicsTools/Heppy/interface/RochCor2012.h"
#include "PhysicsTools/Heppy/interface/FSRWeightAlgo.h"
#include "PhysicsTools/Heppy/interface/CMGMuonCleanerBySegmentsAlgo.h"
#include "PhysicsTools/Heppy/interface/TriggerBitChecker.h"
#include "PhysicsTools/Heppy/interface/MuScleFitCorrector.h"
#include "PhysicsTools/Heppy/interface/EGammaMvaEleEstimatorFWLite.h"
#include "PhysicsTools/Heppy/interface/Davismt2.h"
#include "PhysicsTools/Heppy/interface/mt2w_bisect.h"
#include "PhysicsTools/Heppy/interface/Hemisphere.h"
#include "PhysicsTools/Heppy/interface/AlphaT.h"
#include "PhysicsTools/Heppy/interface/ReclusterJets.h"

#include "EgammaAnalysis/ElectronTools/interface/SimpleElectron.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEPcombinator.h"
//#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyCalibrator.h"
#include <vector>
namespace {
  struct heppy_dictionary {
  };
}
