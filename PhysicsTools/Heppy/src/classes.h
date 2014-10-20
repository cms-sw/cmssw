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
#include "PhysicsTools/Heppy/interface/HemisphereViaKt.h"

namespace {
  namespace {
    BTagSF  bTagSF_; 
    RochCor rc_;
    RochCor2012 rc2012_;
    FSRWeightAlgo walgo_;
    TriggerBitChecker checker;
    CMGMuonCleanerBySegmentsAlgo cmgMuonCleanerBySegmentsAlgo;
    EGammaMvaEleEstimatorFWLite egMVA;
    Hemisphere hemisphere(vector<float> px, vector<float> py, vector<float> pz, vector<float> E, int hemi_seed, int hemi_association);
    HemisphereViaKt hemisphere(vector<float> px, vector<float> py, vector<float> pz, vector<float> E, double ktpower);
    Davismt2 mt2;
    mt2w_bisect::mt2w mt2wlept;
    AlphaT alphaT;
  }
}
