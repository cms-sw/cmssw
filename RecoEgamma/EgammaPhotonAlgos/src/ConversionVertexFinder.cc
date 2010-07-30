#include <iostream>
#include <vector>
#include <memory>
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionVertexFinder.h"
// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
//Kinematic constraint vertex fitter
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include <RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h>
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/ColinearityKinematicConstraint.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

//
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include <TMath.h>





ConversionVertexFinder::ConversionVertexFinder( ){ 
  
  LogDebug("ConversionVertexFinder") << "ConversionVertexFinder CTOR  " <<  "\n";  

}

ConversionVertexFinder::~ConversionVertexFinder() {

  LogDebug("ConversionVertexFinder") << "ConversionVertexFinder DTOR " <<  "\n";  
    
}

bool  ConversionVertexFinder::run(std::vector<reco::TransientTrack>  pair, reco::Vertex& the_vertex) {
  bool found= false;

  if ( pair.size() < 2) return found;
  
  float sigma = 0.00000000001;
  float chi = 0.;
  float ndf = 0.;
  float mass = 0.000000511;
  
  /*
    edm::ParameterSet pSet;
    pSet.addParameter<double>("maxDistance", maxDistance_);//0.001
    pSet.addParameter<double>("maxOfInitialValue",maxOfInitialValue_) ;//1.4
    pSet.addParameter<int>("maxNbrOfIterations", maxNbrOfIterations_);//40
  */
  
  KinematicParticleFactoryFromTransientTrack pFactory;
  
  std::vector<RefCountedKinematicParticle> particles;
  
  particles.push_back(pFactory.particle (pair[0],mass,chi,ndf,sigma));
  particles.push_back(pFactory.particle (pair[1],mass,chi,ndf,sigma));
  
  MultiTrackKinematicConstraint *  constr = new ColinearityKinematicConstraint(ColinearityKinematicConstraint::PhiTheta);
  
  KinematicConstrainedVertexFitter kcvFitter;
  //kcvFitter.setParameters(pSet);
  RefCountedKinematicTree myTree = kcvFitter.fit(particles, constr);
  if( myTree->isValid() ) {
    myTree->movePointerToTheTop();                                                                                
    RefCountedKinematicParticle the_photon = myTree->currentParticle();                                           
    if (the_photon->currentState().isValid()){                                                                    
      //const ParticleMass photon_mass = the_photon->currentState().mass();                                       
      RefCountedKinematicVertex gamma_dec_vertex;                                                               
      gamma_dec_vertex = myTree->currentDecayVertex();                                                          
      if( gamma_dec_vertex->vertexIsValid() ){                                                                  
	const float chi2Prob = ChiSquaredProbability(gamma_dec_vertex->chiSquared(), gamma_dec_vertex->degreesOfFreedom());
	if (chi2Prob>0.){// no longer cut here, only ask positive probability here 
	  //const math::XYZPoint vtxPos(gamma_dec_vertex->position());                                           
	  the_vertex = *gamma_dec_vertex;
	  found = true;
	}
      }
    }
  }
  delete constr;                                                                                                    
  
  


  return found;
}


TransientVertex  ConversionVertexFinder::run(std::vector<reco::TransientTrack>  pair) {
  LogDebug("ConversionVertexFinder") << "ConversionVertexFinder run pair size " << pair.size() <<  "\n";  
  
  //for ( std::vector<reco::TransientTrack>::const_iterator iTk=pair.begin(); iTk!=pair.end(); ++iTk) {
  // LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder  Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";  
  //}


  reco::Vertex theVertex;  
  KalmanVertexFitter fitter(true);
  TransientVertex transientVtx;

  const std::string metname =  "ConversionVertexFinder| ConversionVertexFinder";
  try{

    transientVtx = fitter.vertex(pair); 

  }  catch ( cms::Exception& e ) {


    edm::LogWarning(metname) << "cms::Exception caught in ConversionVertexFinder::run\n"
			     << e.explainSelf();
    
  }
  

  return transientVtx;

    
    
}

