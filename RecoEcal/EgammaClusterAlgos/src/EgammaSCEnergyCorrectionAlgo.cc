//
// $Id: EgammaSCEnergyCorrectionAlgo.cc,v 1.38 2009/05/15 20:49:03 ymaravin Exp $
// Author: David Evans, Bristol
//
#include "RecoEcal/EgammaClusterAlgos/interface/EgammaSCEnergyCorrectionAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/SuperClusterShapeAlgo.h"
#include <iostream>
#include <string>
#include <vector>

EgammaSCEnergyCorrectionAlgo::EgammaSCEnergyCorrectionAlgo(double noise, 
							   reco::CaloCluster::AlgoId theAlgo,
							   const edm::ParameterSet& pset,
							   EgammaSCEnergyCorrectionAlgo::VerbosityLevel verbosity
							   )

{
  sigmaElectronicNoise_ = noise;
  verbosity_ = verbosity;
  recHits_m = new std::map<DetId, EcalRecHit>();
  
}

EgammaSCEnergyCorrectionAlgo::~EgammaSCEnergyCorrectionAlgo()
{
  recHits_m->clear();
  delete recHits_m;
}

reco::SuperCluster EgammaSCEnergyCorrectionAlgo::applyCorrection(const reco::SuperCluster &cl, 
								 const EcalRecHitCollection &rhc, reco::CaloCluster::AlgoId theAlgo, const CaloSubdetectorGeometry* geometry,
								 EcalClusterFunctionBaseClass* EnergyCorrection)
{	

  // Insert the recHits into map	
  // (recHits needed as number of crystals in the seed cluster
  //  with energy above 2sigma noise required)
    EcalRecHitCollection::const_iterator it;
    for (it = rhc.begin(); it != rhc.end(); it++)
    {
      std::pair<DetId, EcalRecHit> map_entry(it->id(), *it);
      recHits_m->insert(map_entry);
    }
	
  // A little bit of trivial info to be sure all is well

  if (verbosity_ <= pINFO)
  {
    std::cout << "   EgammaSCEnergyCorrectionAlgo::applyCorrection" << std::endl;
    std::cout << "   SC has energy " << cl.energy() << std::endl;
    std::cout << "   Will correct now.... " << std::endl;
  }

  // Get the seed cluster  	
  reco::CaloClusterPtr seedC = cl.seed();

  if (verbosity_ <= pINFO)
  {
    std::cout << "   Seed cluster energy... " << seedC->energy() << std::endl;
  }

  // Get the constituent clusters
  reco::CaloClusterPtrVector clusters_v;

  if (verbosity_ <= pINFO) std::cout << "   Constituent cluster energies... ";
  for(reco::CaloCluster_iterator cluster = cl.clustersBegin(); cluster != cl.clustersEnd(); cluster ++)
  {
    clusters_v.push_back(*cluster);
    if (verbosity_ <= pINFO) std::cout << (*cluster)->energy() << ", ";
  }
  if (verbosity_ <= pINFO) std::cout << std::endl;

  // Find the algorithm used to construct the basic clusters making up the supercluster	
  if (verbosity_ <= pINFO) 
  {
    std::cout << "   The seed cluster used algo " << theAlgo;  
  }
 
  // Find the detector region of the supercluster
  // where is the seed cluster?
  std::vector<std::pair<DetId, float> > seedHits = seedC->hitsAndFractions();  
  EcalSubdetector theBase = EcalSubdetector(seedHits.at(0).first.subdetId());
  if (verbosity_ <= pINFO)
  {
    std::cout << "   seed cluster location == " << theBase << std::endl;
  }

  // Get number of crystals 2sigma above noise in seed basiccluster      
  int nCryGT2Sigma = nCrystalsGT2Sigma(*seedC);
  if (verbosity_ <= pINFO)
  {
    std::cout << "   nCryGT2Sigma " << nCryGT2Sigma << std::endl;
  }

  // Supercluster enery - seed basiccluster energy
  float bremsEnergy = cl.energy() - seedC->energy();
  if (verbosity_ <= pINFO)
  {
    std::cout << "   bremsEnergy " << bremsEnergy << std::endl;
  }

  //Create the pointer ot class SuperClusterShapeAlgo
  //which calculates phiWidth and etaWidth
  SuperClusterShapeAlgo* SCShape = new SuperClusterShapeAlgo(&rhc, geometry);

  double phiWidth = 0.;
  double etaWidth = 0.;
  //Calculate phiWidth & etaWidth for SuperClusters
  SCShape->Calculate_Covariances(cl);
  phiWidth = SCShape->phiWidth();
  etaWidth = SCShape->etaWidth();

  // Calculate the new supercluster energy 
  //as a function of number of crystals in the seed basiccluster for Endcap 
  //or apply new Enegry SCale correction
  float newEnergy = 0;

  reco::SuperCluster tmp = cl;
  tmp.setPhiWidth(phiWidth); 
  tmp.setEtaWidth(etaWidth); 
    
  if ( theAlgo == reco::CaloCluster::hybrid || theAlgo == reco::CaloCluster::dynamicHybrid ) {
    newEnergy = tmp.rawEnergy() + EnergyCorrection->getValue(tmp, 3);

  } else if  ( theAlgo == reco::CaloCluster::multi5x5 ) {     
    newEnergy = tmp.rawEnergy() + tmp.preshowerEnergy() + EnergyCorrection->getValue(tmp, 5);

  } else {  
    //Apply f(nCry) correction on island algo and fixedMatrix algo 
    newEnergy = seedC->energy()/fNCrystals(nCryGT2Sigma, theAlgo, theBase)+bremsEnergy;
  } 

  // Create a new supercluster with the corrected energy 
  if (verbosity_ <= pINFO)
    {
      std::cout << "   UNCORRECTED SC has energy... " << cl.energy() << std::endl;
      std::cout << "   CORRECTED SC has energy... " << newEnergy << std::endl;
    }

  reco::SuperCluster corrCl(newEnergy, 
    math::XYZPoint(cl.position().X(), cl.position().Y(), cl.position().Z()),
    cl.seed(), clusters_v, cl.preshowerEnergy());

  //set the flags, although we should implement a ctor in SuperCluster
  corrCl.setFlags(cl.flags());
  corrCl.setPhiWidth(phiWidth);
  corrCl.setEtaWidth(etaWidth);

  // Return the corrected cluster
  recHits_m->clear();
 
  delete SCShape;
  return corrCl;
}

float EgammaSCEnergyCorrectionAlgo::fNCrystals(int nCry, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase)
{

  float p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;
  float x  = (float) nCry;
  float result =1.;
 
  if((theBase == EcalBarrel) && (theAlgo == reco::CaloCluster::hybrid)) 
    {
      if (nCry<=10) 
	{
	  p0 =  6.32879e-01; 
	  p1 =  1.14893e-01; 
	  p2 = -2.45705e-02; 
	  p3 =  2.53074e-03; 
	  p4 = -9.29654e-05; 
	} 
      else if (nCry>10 && nCry<=30) 
        {
          p0 =  6.93196e-01; 
          p1 =  4.44034e-02; 
          p2 = -2.82229e-03; 
          p3 =  8.19495e-05; 
          p4 = -8.96645e-07; 
        } 
      else 
        {
          p0 =  5.65474e+00; 
          p1 = -6.31640e-01; 
          p2 =  3.14218e-02; 
          p3 = -6.84256e-04; 
          p4 =  5.50659e-06; 
        }
      if (x > 40.) x = 40.;
      result = p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;
    }
          
    else if((theBase == EcalEndcap) && (theAlgo == reco::CaloCluster::hybrid)) {
        if (verbosity_ <= pERROR)
        {
          std::cout << "ERROR! HybridEFRYsc called" << std::endl;
        } 
        result = 1.;
      }
        
    else if((theBase == EcalBarrel) && (theAlgo == reco::CaloCluster::island)) { 
        p0 = 4.69976e-01;     // extracted from fit to all endcap classes with Ebrem = 0.
        p1 = 1.45900e-01;
        p2 = -1.61359e-02;
        p3 = 7.99423e-04;
        p4 = -1.47873e-05;
        if (x > 16.) x = 16.;
        result = p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;
      }
        
    else if((theBase == EcalEndcap) && (theAlgo == reco::CaloCluster::island)) {    
        p0 = 4.69976e-01;     // extracted from fit to all endcap classes with Ebrem = 0.
        p1 = 1.45900e-01;
        p2 = -1.61359e-02;
        p3 = 7.99423e-04;
        p4 = -1.47873e-05;
        if (x > 16.) x = 16.;
        result = p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;
      }
      
    else {
      if (verbosity_ <= pINFO)
      {
        std::cout << "trying to correct unknown cluster!!!" << std::endl;
      }
    }

  //Rescale energy scale correction to take into account change in calibrated
  //RecHit definition introduced in CMSSW_1_5_0
  if(theBase == EcalBarrel) {
    result/=0.965;
  } else {
    result/=0.975;
  }
  
  return result;  
}

int EgammaSCEnergyCorrectionAlgo::nCrystalsGT2Sigma(const reco::BasicCluster &seed)
{
  // return number of crystals 2Sigma above
  // electronic noise
  
  std::vector<std::pair<DetId,float > > hits = seed.hitsAndFractions();

  if (verbosity_ <= pINFO)
  {
    std::cout << "      EgammaSCEnergyCorrectionAlgo::nCrystalsGT2Sigma" << std::endl;
    std::cout << "      Will calculate number of crystals above 2sigma noise" << std::endl;
    std::cout << "      sigmaElectronicNoise = " << sigmaElectronicNoise_ << std::endl;
    std::cout << "      There are " << hits.size() << " recHits" << std::endl;
  }

  int nCry = 0;
  std::vector<std::pair<DetId,float > >::iterator hit;
  std::map<DetId, EcalRecHit>::iterator aHit;
  for(hit = hits.begin(); hit != hits.end(); hit++)
    {
      // need to get hit by DetID in order to get energy
      aHit = recHits_m->find((*hit).first);
      if(aHit->second.energy()>2.*sigmaElectronicNoise_) nCry++;
    }

  if (verbosity_ <= pINFO)
  {
    std::cout << "         " << nCry << " of these above 2sigma noise" << std::endl;  
  }
 
  return nCry;
}

