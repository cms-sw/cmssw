//
// $Id: HiEgammaSCEnergyCorrectionAlgo.cc,v 1.1 2010/10/21 22:43:51 yjlee Exp $
// Author: David Evans, Bristol
//
#include "RecoHI/HiEgammaAlgos/interface/HiEgammaSCEnergyCorrectionAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/SuperClusterShapeAlgo.h"
#include <iostream>
#include <string>
#include <vector>

HiEgammaSCEnergyCorrectionAlgo::HiEgammaSCEnergyCorrectionAlgo(double noise, 
							   reco::CaloCluster::AlgoId theAlgo,
							   const edm::ParameterSet& pset,
							   HiEgammaSCEnergyCorrectionAlgo::VerbosityLevel verbosity
							   )

{
  sigmaElectronicNoise_ = noise;
  verbosity_ = verbosity;
  recHits_m = new std::map<DetId, EcalRecHit>();
  
}

HiEgammaSCEnergyCorrectionAlgo::~HiEgammaSCEnergyCorrectionAlgo()
{
  recHits_m->clear();
  delete recHits_m;
}

reco::SuperCluster HiEgammaSCEnergyCorrectionAlgo::applyCorrection(const reco::SuperCluster &cl, 
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
    std::cout << "   HiEgammaSCEnergyCorrectionAlgo::applyCorrection" << std::endl;
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
  
  std::cout <<cl.rawEnergy()<< "Correction: "<<phiWidth/etaWidth<<" "<<fWidth(phiWidth/etaWidth, theAlgo, theBase)<<" "<<fNCrystals(cl.size(), theAlgo, theBase)<<" "<<fClustersSize(cl.clustersSize(), theAlgo, theBase)<<std::endl;
     
  newEnergy = (cl.rawEnergy())/fWidth(phiWidth/etaWidth, theAlgo, theBase)/fNCrystals(cl.size(), theAlgo, theBase)/fClustersSize(cl.clustersSize(), theAlgo, theBase)/fEta(cl.eta(), theAlgo, theBase);
  

  // Create a new supercluster with the corrected energy 
  if (verbosity_ <= pINFO)
    {
      std::cout << "   UNCORRECTED SC has energy... " << cl.energy() << std::endl;
      std::cout << "   CORRECTED SC has energy... " << newEnergy << std::endl;
      std::cout << "   Size..." <<cl.size() << std::endl;
      std::cout << "   Seed nCryGT2Sigma Size..." <<nCryGT2Sigma << std::endl;
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

float HiEgammaSCEnergyCorrectionAlgo::fClustersSize(float clustersSize, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase)
{
  float ncl = clustersSize;
  if((theBase == EcalBarrel) && (theAlgo == reco::CaloCluster::island)) { 
      if (ncl>4) ncl=4;
      return 1.01092-0.009828*(ncl);
  } else if((theBase == EcalEndcap) && (theAlgo == reco::CaloCluster::island)) { 
      if (ncl>2) ncl=2;
      return 1.03483-0.027541*(ncl);   // not yet
  }
  return 1;
}

float HiEgammaSCEnergyCorrectionAlgo::fEta(float eta, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase)
{
  if((theBase == EcalBarrel) && (theAlgo == reco::CaloCluster::island)) { 
      return 1.00971+0.00585804*fabs(eta)-0.017339*fabs(eta)*fabs(eta);
  } else if((theBase == EcalEndcap) && (theAlgo == reco::CaloCluster::island)) { 
      return 0.87454+0.0316459*fabs(eta);   
  }
  return 1;
}

float HiEgammaSCEnergyCorrectionAlgo::fWidth(float widthRatio, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase)
{
  if((theBase == EcalBarrel) && (theAlgo == reco::CaloCluster::island)) { 
      return (1.022-0.02812*widthRatio+0.001637*widthRatio*widthRatio);
  } else if((theBase == EcalEndcap) && (theAlgo == reco::CaloCluster::island)) { 
      return (1.07219-0.0722*widthRatio+0.0067396*widthRatio*widthRatio);   
  }
  return 1;
}

//   char *var ="rawEnergy/cosh(genMatchedEta)/(1.01606-0.0162668*abs(eta))/genMatchedPt/(1.022-0.02812*phiWidth/etaWidth+0.001637*phiWidth*phiWidth/etaWidth/etaWidth)/((0.682554+0.0253013*scSize-(0.0007907)*scSize*scSize+(1.166e-5)*scSize*scSize*scSize-(6.7387e-8)*scSize*scSize*scSize*scSize)*(scSize<40)+(scSize>=40))/((1.016-0.009877*((clustersSize<=4)*clustersSize+(clustersSize>4)*4)))";

float HiEgammaSCEnergyCorrectionAlgo::fNCrystals(int nCry, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase)
{

  float p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;
  float x  = (float) nCry;
  float result =1.;
  
  if((theBase == EcalBarrel) && (theAlgo == reco::CaloCluster::island)) { 
        p0 = 0.682554;     
        p1 = 0.0253013;
        p2 = -0.0007907;
        p3 = 1.166e-5;
        p4 = -6.7387e-8;
        if (x < 10.) x = 10.;
        if (x < 40.) result = p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x; else result = 1;
      }
        
    else if((theBase == EcalEndcap) && (theAlgo == reco::CaloCluster::island)) {    
        
        p0 = 0.712185;     
        p1 = 0.0273609;
        p2 = -0.00103818;
        p3 = 2.01828e-05;
        p4 = -1.71438e-07;
        if (x < 10.) x = 10.;
        result = p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;
        if (x < 40.) result = p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x; else result = 1;   
      }
      
    else {
      if (verbosity_ <= pINFO)
      {
        std::cout << "trying to correct unknown cluster!!!" << std::endl;
      }
    }
  
  return result;  
}

int HiEgammaSCEnergyCorrectionAlgo::nCrystalsGT2Sigma(const reco::BasicCluster &seed)
{
  // return number of crystals 2Sigma above
  // electronic noise
  
  std::vector<std::pair<DetId,float > > hits = seed.hitsAndFractions();

  if (verbosity_ <= pINFO)
  {
    std::cout << "      HiEgammaSCEnergyCorrectionAlgo::nCrystalsGT2Sigma" << std::endl;
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

