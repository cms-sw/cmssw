
/*
________________________________________________________________________

 MixBoostEvtVtxGenerator

 Smear vertex according to the Beta function on the transverse plane
 and a Gaussian on the z axis. It allows the beam to have a crossing
 angle (slopes dxdz and dydz).

 Based on GaussEvtVtxGenerator
 implemented by Francisco Yumiceva (yumiceva@fnal.gov)

 FERMILAB
 2006
________________________________________________________________________
*/

//lingshan: add beta for z-axis boost

#include "IOMC/EventVertexGenerators/interface/MixBoostEvtVtxGenerator.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "HepMC/SimpleVector.h"
#include "TMatrixD.h"

#include <iostream>

using namespace edm;
using namespace std;
using namespace CLHEP;

class RandGaussQ;
class FourVector;


MixBoostEvtVtxGenerator::MixBoostEvtVtxGenerator(edm::ParameterSet const& pset, edm::ConsumesCollector& iC) :
    BaseEvtVtxGenerator(), fVertex(new HepMC::FourVector()), boost_(), fTimeOffset(0),
    useRecVertex(pset.exists("useRecVertex") ? pset.getParameter<bool>("useRecVertex") : false) {
  vtxOffset.resize(3);
  if(pset.exists("vtxOffset")) vtxOffset=pset.getParameter< std::vector<double> >("vtxOffset");
}

MixBoostEvtVtxGenerator::~MixBoostEvtVtxGenerator() {
}

double MixBoostEvtVtxGenerator::BetaFunction(double z, double z0) {
  return sqrt(femittance*(fbetastar+(((z-z0)*(z-z0))/fbetastar)));
}

void MixBoostEvtVtxGenerator::sigmaZ(double s) {
  if (s>=0) {
  	fSigmaZ=s;
  } else {
  	throw cms::Exception("LogicError")
  		<< "Error in MixBoostEvtVtxGenerator::sigmaZ: "
  		<< "Illegal resolution in Z (negative)";
  }
}

TMatrixD* MixBoostEvtVtxGenerator::GetInvLorentzBoost() {

  //alpha_ = 0;
  //phi_ = 142.e-6;
  if (boost_) return boost_.get();
  
  //boost_.ResizeTo(4,4);
  //boost_ = new TMatrixD(4,4);
  TMatrixD tmpboost(4,4);
        TMatrixD tmpboostZ(4,4);
        TMatrixD tmpboostXYZ(4,4);

  //if ((alpha_ == 0) && (phi_==0)) { boost_->Zero(); return boost_; }
  
  // Lorentz boost to frame where the collision is head-on
  // phi is the half crossing angle in the plane ZS
  // alpha is the angle to the S axis from the X axis in the XY plane
  
  tmpboost(0,0) = 1./cos(phi_);
  tmpboost(0,1) = - cos(alpha_)*sin(phi_);
  tmpboost(0,2) = - tan(phi_)*sin(phi_);
  tmpboost(0,3) = - sin(alpha_)*sin(phi_);
  tmpboost(1,0) = - cos(alpha_)*tan(phi_);
  tmpboost(1,1) = 1.;
  tmpboost(1,2) = cos(alpha_)*tan(phi_);
  tmpboost(1,3) = 0.;
  tmpboost(2,0) = 0.;
  tmpboost(2,1) = - cos(alpha_)*sin(phi_);
  tmpboost(2,2) = cos(phi_);
  tmpboost(2,3) = - sin(alpha_)*sin(phi_);
  tmpboost(3,0) = - sin(alpha_)*tan(phi_);
  tmpboost(3,1) = 0.;
  tmpboost(3,2) = sin(alpha_)*tan(phi_);
  tmpboost(3,3) = 1.;
  //cout<<"beta "<<beta_;
  double gama=1.0/sqrt(1-beta_*beta_);
  tmpboostZ(0,0)=gama;
  tmpboostZ(0,1)=0.;
  tmpboostZ(0,2)=-1.0*beta_*gama;
  tmpboostZ(0,3)=0.;
  tmpboostZ(1,0)=0.;
  tmpboostZ(1,1) = 1.;
  tmpboostZ(1,2)=0.;
  tmpboostZ(1,3)=0.;
  tmpboostZ(2,0)=-1.0*beta_*gama;
  tmpboostZ(2,1) = 0.;
  tmpboostZ(2,2)=gama;
  tmpboostZ(2,3) = 0.;
  tmpboostZ(3,0)=0.;
  tmpboostZ(3,1)=0.;
  tmpboostZ(3,2)=0.;
  tmpboostZ(3,3) = 1.;

  tmpboostXYZ=tmpboost*tmpboostZ;
  tmpboost.Invert();

  boost_.reset(new TMatrixD(tmpboostXYZ));
  boost_->Print();
  
  return boost_.get();
}

void MixBoostEvtVtxGenerator::generateNewVertex_(edm::HepMCProduct& product, CLHEP::HepRandomEngine& engine) {

  HepMC::GenEvent const* inev = product.GetEvent();
  HepMC::GenVertex* genvtx = inev->signal_process_vertex();
  if(!genvtx) {
    cout<<"No Signal Process Vertex!"<<endl;
    HepMC::GenEvent::particle_const_iterator pt=inev->particles_begin();
    HepMC::GenEvent::particle_const_iterator ptend=inev->particles_end();
    while(!genvtx || (genvtx->particles_in_size() == 1 && pt != ptend)) {
      if(!genvtx) cout<<"No Gen Vertex!"<<endl;
      if(pt == ptend) cout<<"End reached!"<<endl;
      genvtx = (*pt)->production_vertex();
      ++pt;
    }
  }

  double aX = genvtx->position().x();
  double aY = genvtx->position().y();
  double aZ = genvtx->position().z();
  double aT = genvtx->position().t();

  fVertex->set(aX,aY,aZ,aT);

  product.applyVtxGen(fVertex.get());
  // product.boostToLab(GetInvLorentzBoost(), "vertex");
  // product.boostToLab(GetInvLorentzBoost(), "momentum");
}
