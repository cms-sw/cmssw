#include "PhysicsTools/KinFitter/interface/TKinFitter.h"
#include "PhysicsTools/KinFitter/interface/TSLToyGen.h"
#include "TMatrixD.h" 
#include "TH1.h" 
#include "PhysicsTools/KinFitter/interface/TFitConstraintEp.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintMGaus.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleCart.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleECart.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEScaledMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleESpher.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtEtaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtThetaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleMCCart.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleMCMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleMCPInvSpher.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleMCSpher.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleSpher.h"

namespace {
  struct dictionary {
    TKinFitter b;
    TFitParticleEtEtaPhi tpeep;
    TFitConstraintM tcm;
    TFitConstraintEp tcp;
    //    TAbsFitParticle tafp;
  };      
}
