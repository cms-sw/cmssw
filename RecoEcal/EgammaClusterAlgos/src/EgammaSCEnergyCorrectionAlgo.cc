//
// $Id: EgammaSCEnergyCorrectionAlgo.cc,v 1.48 2012/04/19 13:13:12 argiro Exp $
// Author: David Evans, Bristol
//
#include "RecoEcal/EgammaClusterAlgos/interface/EgammaSCEnergyCorrectionAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/SuperClusterShapeAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <string>
#include <vector>

EgammaSCEnergyCorrectionAlgo::EgammaSCEnergyCorrectionAlgo(float noise, 
							   reco::CaloCluster::AlgoId theAlgo,
							   const edm::ParameterSet& pset)

{
  sigmaElectronicNoise_ = noise;
}


reco::SuperCluster EgammaSCEnergyCorrectionAlgo::applyCorrection(const reco::SuperCluster &cl, 
								 const EcalRecHitCollection &rhc, reco::CaloCluster::AlgoId theAlgo, 
								 const CaloSubdetectorGeometry* geometry,
								 EcalClusterFunctionBaseClass* energyCorrectionFunction,
								 std::string energyCorrectorName_,
								 const int modeEB_,
								 const int modeEE_) {	

	
  // A little bit of trivial info to be sure all is well

  
  {
    LogTrace("EgammaSCEnergyCorrectionAlgo")<< "::applyCorrection" << std::endl;
    LogTrace()<< "   SC has energy " << cl.energy() << std::endl;
    LogTrace()<< "   Will correct now.... " << std::endl;
  }

  // Get the seed cluster  	
  reco::CaloClusterPtr seedC = cl.seed();

  
  LogTrace("EgammaSCEnergyCorrectionAlgo")<< "   Seed cluster energy... " << seedC->energy() << std::endl;
  


  // Find the algorithm used to construct the basic clusters making up the supercluster	
  LogTrace("EgammaSCEnergyCorrectionAlgo")<< "   The seed cluster used algo " << theAlgo;  
  
 
  // Find the detector region of the supercluster
  // where is the seed cluster?
  std::vector<std::pair<DetId, float> > seedHits = seedC->hitsAndFractions();  
  EcalSubdetector theBase = EcalSubdetector(seedHits.at(0).first.subdetId());

  LogTrace("EgammaSCEnergyCorrectionAlgo")<< "   seed cluster location == " << theBase << std::endl;
 
  // Get number of crystals 2sigma above noise in seed basiccluster      
  int nCryGT2Sigma = nCrystalsGT2Sigma(*seedC,rhc);
    
  LogTrace("EgammaSCEnergyCorrectionAlgo")<< "   nCryGT2Sigma " << nCryGT2Sigma << std::endl;
  

  // Supercluster enery - seed basiccluster energy
  float bremsEnergy = cl.energy() - seedC->energy();
   
  LogTrace("EgammaSCEnergyCorrectionAlgo")<< "   bremsEnergy " << bremsEnergy << std::endl;
  

  //Create the pointer ot class SuperClusterShapeAlgo
  //which calculates phiWidth and etaWidth
  SuperClusterShapeAlgo  SCShape(&rhc, geometry);

  double phiWidth = 0.;
  double etaWidth = 0.;
  //Calculate phiWidth & etaWidth for SuperClusters
  SCShape.Calculate_Covariances(cl);
  phiWidth = SCShape.phiWidth();
  etaWidth = SCShape.etaWidth();

  // Calculate the new supercluster energy 
  //as a function of number of crystals in the seed basiccluster for Endcap 
  //or apply new Enegry SCale correction
  float newEnergy = 0;

  reco::SuperCluster tmp = cl;
  tmp.setPhiWidth(phiWidth); 
  tmp.setEtaWidth(etaWidth); 
    

  if ( theAlgo == reco::CaloCluster::hybrid || theAlgo == reco::CaloCluster::dynamicHybrid ) {
    if (energyCorrectorName_=="EcalClusterEnergyCorrection") newEnergy = tmp.rawEnergy() + energyCorrectionFunction->getValue(tmp, modeEB_);
    if (energyCorrectorName_=="EcalClusterEnergyCorrectionObjectSpecific") {
      //std::cout << "newEnergy="<<newEnergy<<std::endl;
      newEnergy = energyCorrectionFunction->getValue(tmp, modeEB_);
    }

  } else if  ( theAlgo == reco::CaloCluster::multi5x5 ) {     
    if (energyCorrectorName_=="EcalClusterEnergyCorrection") newEnergy = tmp.rawEnergy() + tmp.preshowerEnergy() + energyCorrectionFunction->getValue(tmp, modeEE_);
    if (energyCorrectorName_=="EcalClusterEnergyCorrectionObjectSpecific") newEnergy = energyCorrectionFunction->getValue(tmp, modeEE_);

  } else {  
    //Apply f(nCry) correction on island algo and fixedMatrix algo 
    newEnergy = seedC->energy()/fNCrystals(nCryGT2Sigma, theAlgo, theBase)+bremsEnergy;
  } 

  // Create a new supercluster with the corrected energy 

  LogTrace("EgammaSCEnergyCorrectionAlgo")<< "   UNCORRECTED SC has energy... " << cl.energy() << std::endl;
  LogTrace("EgammaSCEnergyCorrectionAlgo")<< "   CORRECTED SC has energy... " << newEnergy << std::endl;

  reco::SuperCluster corrCl =cl;
  
  corrCl.setEnergy(newEnergy);
  corrCl.setPhiWidth(phiWidth);
  corrCl.setEtaWidth(etaWidth);
  
  return corrCl;
}

float EgammaSCEnergyCorrectionAlgo::fNCrystals(int nCry, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase) const {

  float p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;
  float x  =  nCry;
  float result =1.f;
 
  if((theBase == EcalBarrel) && (theAlgo == reco::CaloCluster::hybrid))  {
    if (nCry<=10) 
      {
	p0 =  6.32879e-01f; 
	p1 =  1.14893e-01f; 
	p2 = -2.45705e-02f; 
	p3 =  2.53074e-03f; 
	p4 = -9.29654e-05f; 
      } 
    else if (nCry>10 && nCry<=30) 
      {
	p0 =  6.93196e-01f; 
	p1 =  4.44034e-02f; 
	p2 = -2.82229e-03f; 
	p3 =  8.19495e-05f; 
	p4 = -8.96645e-07f; 
      } 
    else 
      {
	p0 =  5.65474e+00f; 
	p1 = -6.31640e-01f; 
	p2 =  3.14218e-02f; 
	p3 = -6.84256e-04f; 
	p4 =  5.50659e-06f; 
      }
    if (x > 40.f) x = 40.f;
  }
  
  else if((theBase == EcalEndcap) && (theAlgo == reco::CaloCluster::hybrid)) {
     
    LogTrace("EgammaSCEnergyCorrectionAlgo") << "ERROR! HybridEFRYsc called" << std::endl;
      
    return 1.f;
  }
  
  else if((theBase == EcalBarrel) && (theAlgo == reco::CaloCluster::island)) { 
    p0 = 4.69976e-01f;     // extracted from fit to all endcap classes with Ebrem = 0.
    p1 = 1.45900e-01f;
    p2 = -1.61359e-02f;
    p3 = 7.99423e-04f;
    p4 = -1.47873e-05f;
    if (x > 16.f) x = 16.f;
  }
  
  else if((theBase == EcalEndcap) && (theAlgo == reco::CaloCluster::island)) {    
    p0 = 4.69976e-01f;     // extracted from fit to all endcap classes with Ebrem = 0.
    p1 = 1.45900e-01f;
    p2 = -1.61359e-02f;
    p3 = 7.99423e-04f;
    p4 = -1.47873e-05f;
    if (x > 16.f) x = 16.f;
  }
  
  else {
    
    LogTrace("EgammaSCEnergyCorrectionAlgo")<< "trying to correct unknown cluster!!!" << std::endl;
    return 1.f;
  }
  result = p0 + x*(p1 + x*(p2 + x*(p3 + x*p4)));
  
  //Rescale energy scale correction to take into account change in calibrated
  //RecHit definition introduced in CMSSW_1_5_0
  float const ebfact = 1.f/0.965f; 
  float const eefact = 1.f/0.975f; 
    
  if(theBase == EcalBarrel) {
    result*=ebfact;
  } else {
    result*=eefact;
  }
  
  return result;  
}

int EgammaSCEnergyCorrectionAlgo::nCrystalsGT2Sigma(reco::BasicCluster const & seed, EcalRecHitCollection const & rhc) const
{
  // return number of crystals 2Sigma above
  // electronic noise
  
  std::vector<std::pair<DetId,float > > const & hits = seed.hitsAndFractions();


  LogTrace("EgammaSCEnergyCorrectionAlgo") << "      EgammaSCEnergyCorrectionAlgo::nCrystalsGT2Sigma" << std::endl;
  LogTrace("EgammaSCEnergyCorrectionAlgo") << "      Will calculate number of crystals above 2sigma noise" << std::endl;
  LogTrace("EgammaSCEnergyCorrectionAlgo") << "      sigmaElectronicNoise = " << sigmaElectronicNoise_ << std::endl;
  LogTrace("EgammaSCEnergyCorrectionAlgo") << "      There are " << hits.size() << " recHits" << std::endl;

  int nCry = 0;
  std::vector<std::pair<DetId,float > >::const_iterator hit;
  EcalRecHitCollection::const_iterator aHit;
  for(hit = hits.begin(); hit != hits.end(); hit++) {
      // need to get hit by DetID in order to get energy
      aHit = rhc.find((*hit).first);
      if((*aHit).energy()>2.f*sigmaElectronicNoise_) nCry++;
    }

  LogTrace("EgammaSCEnergyCorrectionAlgo") << "         " << nCry << " of these above 2sigma noise" << std::endl;  

 
  return nCry;
}

// apply crack correction

reco::SuperCluster 
EgammaSCEnergyCorrectionAlgo::applyCrackCorrection(const reco::SuperCluster &cl,
						   EcalClusterFunctionBaseClass* crackCorrectionFunction){


  double crackcor = 1.; 

  for(reco::CaloCluster_iterator cIt = cl.clustersBegin(); cIt != cl.clustersEnd(); ++cIt) {

    const reco::CaloClusterPtr cc = *cIt; 
    crackcor *= ( (cl.rawEnergy() +
		   cc->energy()*(crackCorrectionFunction->getValue(*cc)-1.)) / 
		   cl.rawEnergy() );   
  }// loop on BCs
  

  reco::SuperCluster corrCl=cl;
  corrCl.setEnergy(cl.energy()*crackcor);
  

  return corrCl;
}


// apply local containment correction
// Assume that the correction function provides correction for the seed Basic Cluster

reco::SuperCluster 
EgammaSCEnergyCorrectionAlgo::
applyLocalContCorrection(const reco::SuperCluster &cl,
			 EcalClusterFunctionBaseClass* localContCorrectionFunction){


  const EcalRecHitCollection  dummy;

  const reco::CaloClusterPtr & seedBC =  cl.seed();
  float seedBCene = seedBC->energy();  
  float correctedSeedBCene = localContCorrectionFunction->getValue(*seedBC,dummy) * seedBCene;


  reco::SuperCluster correctedSC = cl;
  correctedSC.setEnergy(cl.energy() - seedBCene + correctedSeedBCene);
  
  return correctedSC;



}
