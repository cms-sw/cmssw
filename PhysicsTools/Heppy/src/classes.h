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
#include "PhysicsTools/Heppy/interface/HemisphereViaKt.h"
#include "PhysicsTools/Heppy/interface/AlphaT.h"
#include "EgammaAnalysis/ElectronTools/interface/SimpleElectron.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEPcombinator.h"
//#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyCalibrator.h"

namespace {
  namespace {
    heppy::BTagSF  bTagSF_; 
    heppy::RochCor rc_;
    heppy::RochCor2012 rc2012_;
    heppy::FSRWeightAlgo walgo_;
    heppy::TriggerBitChecker checker;
    heppy::CMGMuonCleanerBySegmentsAlgo cmgMuonCleanerBySegmentsAlgo;
    heppy::EGammaMvaEleEstimatorFWLite egMVA;
    heppy::Hemisphere hemisphere_;
    heppy::HemisphereViaKt hemisphereViaKt_;
    heppy::Davismt2 mt2;
    heppy::mt2w_bisect::mt2w mt2wlept;
    heppy::AlphaT alphaT;
    //  heppy::SimpleElectron fuffaElectron;
    //  ElectronEnergyCalibrator fuffaElectronCalibrator;
    //  heppy::ElectronEPcombinator fuffaElectronCombinator;

  }
}
