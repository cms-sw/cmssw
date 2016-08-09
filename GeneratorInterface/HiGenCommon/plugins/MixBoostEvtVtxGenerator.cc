
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

//#include "IOMC/EventVertexGenerators/interface/BetafuncEvtVtxGenerator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
//#include "CLHEP/Vector/ThreeVector.h"
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

  /// return a new event vertex
  //virtual CLHEP::Hep3Vector * newVertex();
  virtual HepMC::FourVector* newVertex() ;
  virtual void produce( edm::Event&, const edm::EventSetup& ) override;
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
  
  CLHEP::RandGaussQ*  fRandom ;

  edm::EDGetTokenT<reco::VertexCollection>   vtxLabel;
  edm::EDGetTokenT<HepMCProduct>  signalLabel;
  edm::EDGetTokenT<CrossingFrame<HepMCProduct> >   mixLabel;
  bool                     useRecVertex;
  std::vector<double>      vtxOffset;

};

MixBoostEvtVtxGenerator::MixBoostEvtVtxGenerator(const edm::ParameterSet & pset ):
  fVertex(0), boost_(0), fTimeOffset(0),
  vtxLabel(mayConsume<reco::VertexCollection>(pset.getParameter<edm::InputTag>("vtxLabel"))),
  signalLabel(consumes<HepMCProduct>(pset.getParameter<edm::InputTag>("signalLabel"))),
  mixLabel(consumes<CrossingFrame<HepMCProduct> >(pset.getParameter<edm::InputTag>("mixLabel"))),
  useRecVertex(pset.exists("useRecVertex")?pset.getParameter<bool>("useRecVertex"):false)
{ 
  beta_  =  pset.getParameter<double>("Beta");
  alpha_ = 0;
  phi_ = 0;
  if(pset.exists("Alpha")){
     alpha_ =  pset.getParameter<double>("Alpha")*radian;
     phi_   =  pset.getParameter<double>("Phi")*radian;
  }

  vtxOffset.resize(3);
  if(pset.exists("vtxOffset")) vtxOffset=pset.getParameter< std::vector<double> >("vtxOffset"); 

  produces<edm::HepMCProduct>();
  
}

MixBoostEvtVtxGenerator::~MixBoostEvtVtxGenerator() 
{
  if (fVertex != 0) delete fVertex ;
  if (boost_ != 0 ) delete boost_;
  if (fRandom != 0) delete fRandom; 
}


//Hep3Vector* MixBoostEvtVtxGenerator::newVertex() {
HepMC::FourVector* MixBoostEvtVtxGenerator::newVertex() {

	
	double X,Y,Z;
	
	double tmp_sigz = fRandom->fire(0., fSigmaZ);
	Z = tmp_sigz + fZ0;

	double tmp_sigx = BetaFunction(Z,fZ0); 
	// need sqrt(2) for beamspot width relative to single beam width
	tmp_sigx /= sqrt(2.0);
	X = fRandom->fire(0.,tmp_sigx) + fX0; // + Z*fdxdz ;

	double tmp_sigy = BetaFunction(Z,fZ0);
	// need sqrt(2) for beamspot width relative to single beam width
	tmp_sigy /= sqrt(2.0);
	Y = fRandom->fire(0.,tmp_sigy) + fY0; // + Z*fdydz;

	double tmp_sigt = fRandom->fire(0., fSigmaZ);
	double T = tmp_sigt + fTimeOffset; 

	if ( fVertex == 0 ) fVertex = new HepMC::FourVector();
	fVertex->set(X,Y,Z,T);
		
	return fVertex;
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

	//alpha_ = 0;
	//phi_ = 142.e-6;
//	if (boost_ != 0 ) return boost_;
	
	//boost_.ResizeTo(4,4);
	//boost_ = new TMatrixD(4,4);
	TMatrixD tmpboost(4,4);
        TMatrixD tmpboostZ(4,4);
        TMatrixD tmpboostXYZ(4,4);

	//if ( (alpha_ == 0) && (phi_==0) ) { boost_->Zero(); return boost_; }
	
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

       tmpboostXYZ=tmpboostZ*tmpboost;
       tmpboostXYZ.Invert();

       cout<<"Boosting with beta : "<<beta_<<endl;

       boost_ = new TMatrixD(tmpboostXYZ);
       boost_->Print();
	
	return boost_;
}

HepMC::FourVector* MixBoostEvtVtxGenerator::getVertex( Event& evt){
  
  const HepMC::GenEvent* inev = 0;

  Handle<CrossingFrame<HepMCProduct> > cf;
  evt.getByToken(mixLabel,cf);
  MixCollection<HepMCProduct> mix(cf.product());

  const HepMCProduct& bkg = mix.getObject(1);
  if(!(bkg.isVtxGenApplied())){
    throw cms::Exception("MatchVtx")<<"Input background does not have smeared vertex!"<<endl;
  }else{
    inev = bkg.GetEvent();
  }

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
  evt.getByToken(vtxLabel,input);

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
  Handle<HepMCProduct> HepUnsmearedMCEvt;
  evt.getByToken(signalLabel, HepUnsmearedMCEvt);
    
  // Copy the HepMC::GenEvent
  HepMC::GenEvent* genevt = new HepMC::GenEvent(*HepUnsmearedMCEvt->GetEvent());
  std::unique_ptr<edm::HepMCProduct> HepMCEvt(new edm::HepMCProduct(genevt));
  // generate new vertex & apply the shift 
  //
 
  HepMCEvt->boostToLab( GetInvLorentzBoost(), "vertex" );
  HepMCEvt->boostToLab( GetInvLorentzBoost(), "momentum" );

  HepMCEvt->applyVtxGen( useRecVertex ? getRecVertex(evt) : getVertex(evt) ) ;
  
  evt.put(std::move(HepMCEvt));
  return ;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MixBoostEvtVtxGenerator);
