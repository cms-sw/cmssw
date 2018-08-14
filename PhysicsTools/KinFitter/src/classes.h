#include "PhysicsTools/KinFitter/interface/TAbsFitConstraint.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintEp.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintMGaus.h"
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
#include "PhysicsTools/KinFitter/interface/TKinFitter.h"
#include "PhysicsTools/KinFitter/interface/TSLToyGen.h"


namespace PhysicsTools_KinFitter {
  struct dictionary {

     TFitConstraintEp fce;
     TFitConstraintEp::component fce_c;
     TFitConstraintM fcm;
     TFitConstraintMGaus fcmg;
     TFitParticleCart fpc;
     TFitParticleECart fpec;
     TFitParticleEMomDev fpemd;
     TFitParticleEScaledMomDev fpesmd;
     TFitParticleESpher fpes;
     TFitParticleEtEtaPhi fpeep;
     TFitParticleEtThetaPhi fpetp;
     TFitParticleMCCart fpmcc;
     TFitParticleMCMomDev fmmccd;
     TFitParticleMCPInvSpher fpmcpis;
     TFitParticleMCSpher fpmcs;
     TFitParticleMomDev fpmd;
     TFitParticleSpher fps;
     TKinFitter kf;
     TSLToyGen sltg;

  };
}
