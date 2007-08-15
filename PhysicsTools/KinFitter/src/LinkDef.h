// -*- C++ -*-

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
// This linkdef file contains all "pragma link" for CLHEP inclusion
// into root including non-member operators and functions
// of Vector, Matrix, DiagMatrix and SymMatrix:
//#ifdef __CINT__
// ##################################################
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// ################## Functions #####################

#pragma link C++ class TAbsFitConstraint;
#pragma link C++ class TFitConstraintEp;
#pragma link C++ class TFitConstraintM;
#pragma link C++ class TFitConstraintMGaus;
//#pragma link C++ class TFitConstraintE;
#pragma link C++ class TAbsFitParticle;
#pragma link C++ class TFitParticleMomDev;
#pragma link C++ class TFitParticleCart;
#pragma link C++ class TFitParticleSpher;
#pragma link C++ class TFitParticleEMomDev;
#pragma link C++ class TFitParticleECart;
#pragma link C++ class TFitParticleESpher;
#pragma link C++ class TFitParticleMCMomDev;
#pragma link C++ class TFitParticleMCCart;
#pragma link C++ class TFitParticleMCSpher;
#pragma link C++ class TFitParticleEScaledMomDev;
#pragma link C++ class TFitParticleMCPInvSpher;
#pragma link C++ class TFitParticleEtEtaPhi;
#pragma link C++ class TFitParticleEtThetaPhi;
#pragma link C++ class TKinFitter;
#pragma link C++ class TSLToyGen;

//  #pragma link C++ class std::vector<TAbsFitParticle*>+;
//  #pragma link C++ class std::vector<TAbsFitConstraint*>+;

//#endif
