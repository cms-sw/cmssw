/** \file EmulatedME0SegmentProducer.cc
 *
 * \author David Nash
 */

#include <FastSimulation/Muons/plugins/EmulatedME0SegmentProducer.h>

#include <FWCore/PluginManager/interface/ModuleDef.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include "TRandom3.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"

#include "TLorentzVector.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

EmulatedME0SegmentProducer::EmulatedME0SegmentProducer(const edm::ParameterSet& pas) : iev(0) {
	
    Rand = new TRandom3();
    produces<std::vector<EmulatedME0Segment> >(); 

}

EmulatedME0SegmentProducer::~EmulatedME0SegmentProducer() {}

void EmulatedME0SegmentProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {

    LogDebug("EmulatedME0SegmentProducer") << "start producing segments for " << ++iev << "th event ";
	
    //Getting the objects we'll need
    
    using namespace edm;
    ESHandle<MagneticField> bField;
    setup.get<IdealMagneticFieldRecord>().get(bField);
    ESHandle<Propagator> shProp;
    setup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAlong", shProp);

    
    using namespace reco;

    Handle<GenParticleCollection> genParticles;
    ev.getByLabel<GenParticleCollection>("genParticles", genParticles);


    unsigned int gensize=genParticles->size();
    //std::cout<<"gensize = "<<gensize<<std::endl;

    AlgebraicSymMatrix theCovMatrix(4,0);
    AlgebraicSymMatrix theGlobalCovMatrix(4,0);
    AlgebraicMatrix theRotMatrix(4,4,0);

    //Big loop over all gen particles in the event, to propagate them and make segments

    std::auto_ptr<std::vector<EmulatedME0Segment> > oc( new std::vector<EmulatedME0Segment> ); 

    for(unsigned int i=0; i<gensize; ++i) {
      const reco::GenParticle& CurrentParticle=(*genParticles)[i];
      //Just doing status one muons...
      if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) ){  

	//Setup
	float zSign  = CurrentParticle.pz()/fabs(CurrentParticle.pz());

	float zValue = 560. * zSign;
	Plane *plane = new Plane(Surface::PositionType(0,0,zValue),Surface::RotationType());
	TLorentzVector Momentum;
	Momentum.SetPtEtaPhiM(CurrentParticle.pt()
			      ,CurrentParticle.eta()
			      ,CurrentParticle.phi()
			      ,CurrentParticle.mass());
	GlobalVector p3gen(Momentum.Px(), Momentum.Py(), Momentum.Pz());
	GlobalVector r3gen = GlobalVector(CurrentParticle.vertex().x()
						    ,CurrentParticle.vertex().y()
						    ,CurrentParticle.vertex().z());

	AlgebraicSymMatrix66 covGen = AlgebraicMatrixID(); 
	covGen *= 1e-20; // initialize to sigma=1e-10 .. should get overwhelmed by MULS
	AlgebraicSymMatrix66 covFinal;
	int chargeGen =  CurrentParticle.charge(); 

	//Propagation
	FreeTrajectoryState initstate = getFTS(p3gen, r3gen, chargeGen, covGen, &*bField);
	
	SteppingHelixStateInfo startstate(initstate);
	SteppingHelixStateInfo laststate;

	const SteppingHelixPropagator* ThisshProp = 
	  dynamic_cast<const SteppingHelixPropagator*>(&*shProp);

	laststate = ThisshProp->propagate(startstate, *plane);

	FreeTrajectoryState finalstate;
	laststate.getFreeState(finalstate);
	
	GlobalVector p3Final, r3Final;
	getFromFTS(finalstate, p3Final, r3Final, chargeGen, covFinal);

	//Smearing the position

	Double_t rho = r3Final.perp();
	Double_t phi = r3Final.phi();
	//std::cout<<"rho = "<<rho<<std::endl;
	//std::cout<<"phi = "<<phi<<std::endl;
	Double_t drhodx = r3Final.x()/rho;
	Double_t drhody = r3Final.y()/rho;
	Double_t dphidx = -r3Final.y()/(rho*rho);
	Double_t dphidy = r3Final.x()/(rho*rho);
	
	Double_t sigmarho = sqrt( drhodx*drhodx*covFinal(0,0)+
				  drhody*drhody*covFinal(1,1)+
				  drhodx*drhody*2*covFinal(0,1) );

	Double_t sigmaphi = sqrt( dphidx*dphidx*covFinal(0,0)+
				  dphidy*dphidy*covFinal(1,1)+
				  dphidx*dphidy*2*covFinal(0,1) );
	

	Double_t newrho = rho + Rand->Gaus(0,sigmarho);    //Add smearing here
	Double_t newphi = phi + Rand->Gaus(0,sigmaphi);


	GlobalVector SmearedPosition(newrho*cos(newphi),newrho*sin(newphi),r3Final.z());
	
	Double_t sigma_px = covFinal(3,3);
	Double_t sigma_py = covFinal(4,4);
	Double_t sigma_pz = covFinal(5,5);

	Double_t new_px = p3Final.x() + Rand->Gaus(0,sigma_px);    //Add smearing here
	Double_t new_py = p3Final.y() + Rand->Gaus(0,sigma_py);
	Double_t new_pz = p3Final.z() + Rand->Gaus(0,sigma_pz);

	GlobalVector SmearedDirection(new_px,new_py,new_pz);

	//Filling the EmulatedME0Segment

	LocalPoint Point(SmearedPosition.x(),SmearedPosition.y(),SmearedPosition.z());
	LocalVector Direction(SmearedDirection.x(),SmearedDirection.y(),SmearedDirection.z());

	theCovMatrix[2][2] = 0.01;
	theCovMatrix[3][3] = 2.;

	theCovMatrix[0][0] = 0.00025;
	theCovMatrix[1][1] = 0.07;

	//Do the transformation to global coordinates on the Cov Matrix
	double piover2 = acos(0.);
	theRotMatrix[0][0] = cos(SmearedPosition.phi()+piover2);
	theRotMatrix[1][1] = cos(SmearedPosition.phi()+piover2);
	theRotMatrix[2][2] = cos(SmearedPosition.phi()+piover2);
	theRotMatrix[3][3] = cos(SmearedPosition.phi()+piover2);
	
	theRotMatrix[0][1] = -sin(SmearedPosition.phi()+piover2);
	theRotMatrix[1][0] = sin(SmearedPosition.phi()+piover2);

	theRotMatrix[2][3] = -sin(SmearedPosition.phi()+piover2);
	theRotMatrix[3][2] = sin(SmearedPosition.phi()+piover2);

	RotateCovMatrix(theRotMatrix,theCovMatrix,4,theGlobalCovMatrix);
	if ( (fabs(Point.eta()) < 2.4) || (fabs(Point.eta()) > 4.0) ) continue;         //Currently we only save segments that propagate to our defined disk


	//std::cout<<"new rho = "<<newrho<<std::endl;
	//std::cout<<"new phi = "<<newphi<<std::endl;

	oc->push_back(EmulatedME0Segment(Point, Direction,theGlobalCovMatrix,0.));    


      }
    }
    //std::cout<<"oc.size() = "<<oc->size()<<std::endl;
    // put collection in event
    ev.put(oc);
}

FreeTrajectoryState
EmulatedME0SegmentProducer::getFTS(const GlobalVector& p3, const GlobalVector& r3, 
			   int charge, const AlgebraicSymMatrix55& cov,
			   const MagneticField* field){

  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CurvilinearTrajectoryError tCov(cov);
  
  return cov.kRows == 5 ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars) ;
}

FreeTrajectoryState
EmulatedME0SegmentProducer::getFTS(const GlobalVector& p3, const GlobalVector& r3, 
			   int charge, const AlgebraicSymMatrix66& cov,
			   const MagneticField* field){

  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CartesianTrajectoryError tCov(cov);
  
  return cov.kRows == 6 ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars) ;
}

void EmulatedME0SegmentProducer::getFromFTS(const FreeTrajectoryState& fts,
				    GlobalVector& p3, GlobalVector& r3, 
				    int& charge, AlgebraicSymMatrix66& cov){
  GlobalVector p3GV = fts.momentum();
  GlobalPoint r3GP = fts.position();

  GlobalVector p3T(p3GV.x(), p3GV.y(), p3GV.z());
  GlobalVector r3T(r3GP.x(), r3GP.y(), r3GP.z());
  
  p3 = p3T;
  r3 = r3T;  
  
  charge = fts.charge();
  cov = fts.hasError() ? fts.cartesianError().matrix() : AlgebraicSymMatrix66();

}

void EmulatedME0SegmentProducer::RotateCovMatrix(const AlgebraicMatrix& R, const AlgebraicSymMatrix& M, int size, AlgebraicSymMatrix& Output){
  //Here we start to do RMR^T
  //Here we make (RM)
  AlgebraicMatrix MidPoint(size,size,0);
  for (int i=0; i<size;i++){
    for (int j=0; j<size;j++){
      for (int k=0; k<size;k++){
	MidPoint[i][j] += R[i][k]*M[k][j];
      }
    }
  }
  //Here we write to Output (RM)R^T - note that we transpose R via index inversion
  for (int i=0; i<size;i++){
    for (int j=0; j<size;j++){
      for (int k=0; k<size;k++){
	Output[i][j] += MidPoint[i][k]*R[j][k];
      }
    }
  }
  
}



 DEFINE_FWK_MODULE(EmulatedME0SegmentProducer);
