
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
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
class FourVector ;

class MixBoostEvtVtxGenerator : public edm::EDProducer{
public:
  MixBoostEvtVtxGenerator(const edm::ParameterSet & p);
  virtual ~MixBoostEvtVtxGenerator();

  virtual HepMC::FourVector* newVertex() ;
  virtual void produce( edm::Event&, const edm::EventSetup& );
  virtual TMatrixD* GetInvLorentzBoost();
  virtual HepMC::FourVector* getVertex(edm::Event&);
  virtual HepMC::FourVector* getRecVertex(edm::Event&);

  /// set resolution in Z in cm
  void sigmaZ(double s=1.0);

  /// set mean in X in cm
  void X0(double m=0) { fX0=m; }
  /// set mean in Y in cm
  void Y0(double m=0) { fY0=m; }
  /// set mean in Z in cm
  void Z0(double m=0) { fZ0=m; }

  /// set half crossing angle
  void Phi(double m=0) { phi_=m; }
  /// angle between crossing plane and horizontal plane
  void Alpha(double m=0) { alpha_=m; }
  void Beta(double m=0) { beta_=m; }

  /// set beta_star
  void betastar(double m=0) { fbetastar=m; }
  /// emittance (no the normalized)
  void emittance(double m=0) { femittance=m; }

  /// beta function
  double BetaFunction(double z, double z0);
  //  CLHEP::HepRandomEngine& getEngine();

private:
  /** Copy constructor */
  MixBoostEvtVtxGenerator(const MixBoostEvtVtxGenerator &p);
  /** Copy assignment operator */
  MixBoostEvtVtxGenerator&  operator = (const MixBoostEvtVtxGenerator & rhs );

  double alpha_, phi_;
  //TMatrixD boost_;
  double beta_;
  double fX0, fY0, fZ0;
  double fSigmaZ;
  //double fdxdz, fdydz;
  double fbetastar, femittance;
  double falpha;

  HepMC::FourVector*       fVertex ;
  TMatrixD *boost_;
  double fTimeOffset;
  
  //  CLHEP::HepRandomEngine*  fEngine;
  edm::InputTag            sourceLabel;

  //  CLHEP::RandGaussQ*  fRandom ;

  edm::InputTag            signalLabel;
  edm::InputTag            hiLabel;
  bool                     useRecVertex;
  std::vector<double>      vtxOffset;
  bool verbosity_;

};


MixBoostEvtVtxGenerator::MixBoostEvtVtxGenerator(const edm::ParameterSet & pset ):
  fVertex(0), boost_(0), fTimeOffset(0),
  signalLabel(pset.getParameter<edm::InputTag>("signalLabel")),
  hiLabel(pset.getParameter<edm::InputTag>("heavyIonLabel")),
  useRecVertex(pset.exists("useRecVertex")?pset.getParameter<bool>("useRecVertex"):false),
  verbosity_(pset.getUntrackedParameter<bool>("verbosity",false))
{ 

  vtxOffset.resize(3);
  if(pset.exists("vtxOffset")) vtxOffset=pset.getParameter< std::vector<double> >("vtxOffset"); 
  beta_  =  pset.getParameter<double>("Beta");

  alpha_ = 0;
  phi_ = 0;
  if(pset.exists("Alpha")){
    alpha_ =  pset.getParameter<double>("Alpha")*radian;
    phi_   =  pset.getParameter<double>("Phi")*radian;
  }

  produces<bool>("matchedVertex"); 
  
}

MixBoostEvtVtxGenerator::~MixBoostEvtVtxGenerator() 
{
  delete fVertex ;
  if (boost_ != 0 ) delete boost_;
}


HepMC::FourVector* MixBoostEvtVtxGenerator::newVertex() {
  return 0;
}

double MixBoostEvtVtxGenerator::BetaFunction(double z, double z0)
{
	return sqrt(femittance*(fbetastar+(((z-z0)*(z-z0))/fbetastar)));

}


void MixBoostEvtVtxGenerator::sigmaZ(double s) 
{ 
	if (s>=0 ) {
		fSigmaZ=s; 
	}
	else {
		throw cms::Exception("LogicError")
			<< "Error in MixBoostEvtVtxGenerator::sigmaZ: "
			<< "Illegal resolution in Z (negative)";
	}
}

TMatrixD* MixBoostEvtVtxGenerator::GetInvLorentzBoost() {

	TMatrixD tmpboost(4,4);
        TMatrixD tmpboostZ(4,4);
        TMatrixD tmpboostXYZ(4,4);

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
       tmpboostXYZ.Invert();

       boost_ = new TMatrixD(tmpboostXYZ);
       if ( verbosity_ )boost_->Print();
	
	return boost_;
}

HepMC::FourVector* MixBoostEvtVtxGenerator::getVertex( Event& evt){
  
  Handle<HepMCProduct> input;
  evt.getByLabel(hiLabel,input);

  const HepMC::GenEvent* inev = input->GetEvent();
  HepMC::GenVertex* genvtx = inev->signal_process_vertex();
  if(!genvtx){
    cout<<"No Signal Process Vertex!"<<endl;
    HepMC::GenEvent::particle_const_iterator pt=inev->particles_begin();
    HepMC::GenEvent::particle_const_iterator ptend=inev->particles_end();
    while(!genvtx || ( genvtx->particles_in_size() == 1 && pt != ptend ) ){
      if(!genvtx) cout<<"No Gen Vertex!"<<endl;
      if(pt == ptend) cout<<"End reached!"<<endl;
      genvtx = (*pt)->production_vertex();
      ++pt;
    }
  }

  double aX,aY,aZ,aT;
  
  aX = genvtx->position().x();
  aY = genvtx->position().y();
  aZ = genvtx->position().z();
  aT = genvtx->position().t();
   
  if(!fVertex) fVertex = new HepMC::FourVector();
  fVertex->set(aX,aY,aZ,aT);
  
  return fVertex;
  
}
 
 
HepMC::FourVector* MixBoostEvtVtxGenerator::getRecVertex( Event& evt){
 
  Handle<reco::VertexCollection> input;
  evt.getByLabel(hiLabel,input);

  double aX,aY,aZ;
 
  aX = input->begin()->position().x() + vtxOffset[0];
  aY = input->begin()->position().y() + vtxOffset[1];
  aZ = input->begin()->position().z() + vtxOffset[2];
 
  if(!fVertex) fVertex = new HepMC::FourVector();
  fVertex->set(10.0*aX,10.0*aY,10.0*aZ,0.0); // HepMC positions in mm (RECO in cm)
   
  return fVertex;
 
}


void MixBoostEvtVtxGenerator::produce( Event& evt, const EventSetup& )
{
    
    
  Handle<HepMCProduct> HepMCEvt ;
  
  evt.getByLabel( signalLabel, HepMCEvt ) ;
    
  // generate new vertex & apply the shift 
  //
 
  HepMCEvt->boostToLab( GetInvLorentzBoost(), "vertex" );
  HepMCEvt->boostToLab( GetInvLorentzBoost(), "momentum" );

  HepMCEvt->applyVtxGen( useRecVertex ? getRecVertex(evt) : getVertex(evt) ) ;
  
  // OK, create a (pseudo)product and put in into edm::Event
  //
  auto_ptr<bool> NewProduct(new bool(true)) ;      
  evt.put( NewProduct ,"matchedVertex") ;
       
  return ;
  
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MixBoostEvtVtxGenerator);
