//
// $Id: HiEgammaSCEnergyCorrectionAlgo.cc,v 1.9 2011/04/25 15:17:04 yjlee Exp $
// Author: David Evans, Bristol
//
#include "RecoHI/HiEgammaAlgos/interface/HiEgammaSCEnergyCorrectionAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/SuperClusterShapeAlgo.h"
#include <iostream>
#include <string>
#include <vector>

HiEgammaSCEnergyCorrectionAlgo::HiEgammaSCEnergyCorrectionAlgo(float noise, 
							   reco::CaloCluster::AlgoId theAlgo,
							   const edm::ParameterSet& pSet,
							   HiEgammaSCEnergyCorrectionAlgo::VerbosityLevel verbosity
							   )
{
  sigmaElectronicNoise_ = noise;
  verbosity_ = verbosity;
  
  // Parameters
  p_fBrem_ = pSet.getParameter< std::vector <double> > ("fBremVect");
  p_fBremTh_ = pSet.getParameter< std::vector <double> > ("fBremThVect");
  p_fEta_ = pSet.getParameter< std::vector <double> > ("fEtaVect");
  p_fEtEta_ = pSet.getParameter< std::vector <double> > ("fEtEtaVect");

  // Min R9 
  minR9Barrel_ = pSet.getParameter<double>("minR9Barrel");
  minR9Endcap_ = pSet.getParameter<double>("minR9Endcap");

  // Max R9
  maxR9_ = pSet.getParameter<double>("maxR9");

}



reco::SuperCluster 
HiEgammaSCEnergyCorrectionAlgo::applyCorrection(const reco::SuperCluster &cl, 
						const EcalRecHitCollection &rhc, reco::CaloCluster::AlgoId theAlgo, 
						const CaloSubdetectorGeometry* geometry,
						const CaloTopology *topology, EcalClusterFunctionBaseClass* EnergyCorrection)
{	
	
   // Print out a little bit of trivial info to be sure all is well
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
   std::vector<std::pair<DetId, float> > const & seedHits = seedC->hitsAndFractions();  
   EcalSubdetector theBase = EcalSubdetector(seedHits.at(0).first.subdetId());
   if (verbosity_ <= pINFO)
   {
      std::cout << "   seed cluster location == " << theBase << std::endl;
   }

   // Get number of crystals 2sigma above noise in seed basiccluster      
   int nCryGT2Sigma = nCrystalsGT2Sigma(*seedC,rhc);
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

   // Create a SuperClusterShapeAlgo
   // which calculates phiWidth and etaWidth
   SuperClusterShapeAlgo SCShape(&rhc, geometry);

   double phiWidth = 0.;
   double etaWidth = 0.;
 
   // Calculate phiWidth & etaWidth for SuperClusters
   SCShape.Calculate_Covariances(cl);
   phiWidth = SCShape.phiWidth();
   etaWidth = SCShape.etaWidth();

   // Calculate r9 and 5x5 energy
   float e3x3    =   EcalClusterTools::e3x3(  *(cl.seed()), &rhc, &(*topology));
   float e5x5    =   EcalClusterTools::e5x5(  *(cl.seed()), &rhc, &(*topology));
   float r9 = e3x3 / cl.rawEnergy();

   // Calculate the new supercluster energy 
   // as a function of number of crystals in the seed basiccluster for Endcap 
   // lslslsor apply new Enegry SCale correction
   float newEnergy = 0;

   // if r9 > maxR9_ -> uncaptured brem.
   if ((r9 < minR9Barrel_&&theBase == EcalBarrel) || (r9 < minR9Endcap_&&theBase == EcalEndcap)) {     
      // if r9 is greater than threshold, then use the SC raw energy
      newEnergy = (cl.rawEnergy())/ fEta(cl.eta(), theAlgo, theBase) / 
	fBrem(phiWidth/etaWidth, theAlgo, theBase)
	/fEtEta(cl.energy()/cosh(cl.eta()), cl.eta(), theAlgo, theBase);

   }  else {
      if (r9 < maxR9_) {
         // use 5x5 energy if r9 < threshold
         newEnergy = e5x5 / fEta(cl.eta(), theAlgo, theBase);
      } else {
         // it comes from a uncaptured brem, doesn't correct
         newEnergy = cl.rawEnergy();
      }
   }
   
   if (newEnergy > 2* cl.rawEnergy()) newEnergy = cl.rawEnergy();  // avoid very large correction due to the uncaptured brem

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
   
   return corrCl;
}

float HiEgammaSCEnergyCorrectionAlgo::fEtEta(float et, float eta, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase) const
{
  int offset = 0;
  float factor;
  if((theBase == EcalBarrel) && (theAlgo == reco::CaloCluster::island)) { 
      offset = 0;
  } else if((theBase == EcalEndcap) && (theAlgo == reco::CaloCluster::island)) { 
      offset = 7;
  }
  
  // Et dependent correction
  factor = (p_fEtEta_[0+offset] + p_fEtEta_[1+offset]*sqrt(et));
  // eta dependent correction
  factor *= (p_fEtEta_[2+offset] + p_fEtEta_[3+offset]*fabs(eta) +
	     p_fEtEta_[4+offset]*eta*eta + p_fEtEta_[5+offset]*eta*eta*fabs(eta) + + p_fEtEta_[6+offset]*eta*eta*eta*eta);

  // Constraint correction factor
  if (factor< 0.66f ) factor = 0.66f;
  if (factor> 1.5f  ) factor = 1.5f;

  return factor;

}

float HiEgammaSCEnergyCorrectionAlgo::fEta(float eta, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase) const
{
  int offset = 0;
  float factor;
  if((theBase == EcalBarrel) && (theAlgo == reco::CaloCluster::island)) { 
      offset = 0;
  } else if((theBase == EcalEndcap) && (theAlgo == reco::CaloCluster::island)) { 
      offset = 3;
  }

  factor = (p_fEta_[0+offset] + p_fEta_[1+offset]*fabs(eta) + p_fEta_[2+offset]*eta*eta);

  // Constraint correction factor
  if (factor< 0.66f ) factor = 0.66f;
  if (factor> 1.5f  ) factor = 1.5f;

  return factor;
}

float HiEgammaSCEnergyCorrectionAlgo::fBrem(float brem, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase) const
{
  int det = 0;
  int offset = 0;
  float factor;
  if((theBase == EcalBarrel) && (theAlgo == reco::CaloCluster::island)) { 
      det = 0;
      offset = 0;
  } else if((theBase == EcalEndcap) && (theAlgo == reco::CaloCluster::island)) { 
      det = 1;
      offset = 6;
  }
  
  if (brem < p_fBremTh_[det]) {
     factor = p_fBrem_[0+offset] + p_fBrem_[1+offset]*brem + p_fBrem_[2+offset]*brem*brem;
  } else {
     factor = p_fBrem_[3+offset] + p_fBrem_[4+offset]*brem + p_fBrem_[5+offset]*brem*brem;
  };

  // Constraint correction factor
  if (factor< 0.66f ) factor = 0.66f;
  if (factor> 1.5f  ) factor = 1.5f;
  
  return factor;
}

//   char *var ="rawEnergy/cosh(genMatchedEta)/(1.01606-0.0162668*abs(eta))/genMatchedPt/(1.022-0.02812*phiWidth/etaWidth+0.001637*phiWidth*phiWidth/etaWidth/etaWidth)/((0.682554+0.0253013*scSize-(0.0007907)*scSize*scSize+(1.166e-5)*scSize*scSize*scSize-(6.7387e-8)*scSize*scSize*scSize*scSize)*(scSize<40)+(scSize>=40))/((1.016-0.009877*((clustersSize<=4)*clustersSize+(clustersSize>4)*4)))";

float HiEgammaSCEnergyCorrectionAlgo::fNCrystals(int nCry, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase) const
{

  float x  = (float) nCry;
  float result =1.f;
  
  if((theBase == EcalBarrel) && (theAlgo == reco::CaloCluster::island)) { 
        float const p0 = 0.682554f;     
        float const p1 = 0.0253013f;
        float const p2 = -0.0007907f;
        float const p3 = 1.166e-5f;
        float const p4 = -6.7387e-8f;
        if (x < 10.f) x = 10.f;
        if (x < 40.f) result = p0 + x*(p1 + x*(p2 + x*(p3 + x*p4))); else result = 1.f;
      }
        
    else if((theBase == EcalEndcap) && (theAlgo == reco::CaloCluster::island)) {    
        
        float const p0 = 0.712185f;     
        float const p1 = 0.0273609f;
        float const p2 = -0.00103818f;
        float const p3 = 2.01828e-05f;
        float const p4 = -1.71438e-07f;
        if (x < 10.f) x = 10.f;
	if (x < 40.f) result = p0 + x*(p1 + x*(p2 + x*(p3 + x*p4))); else result = 1.f;   
      }
      
    else {
      if (verbosity_ <= pINFO)
      {
        std::cout << "trying to correct unknown cluster!!!" << std::endl;
      }
    }
  
  if (result > 1.5f) result = 1.5f;
  if (result < 0.5f) result = 0.5f;

  return result;  
}

int HiEgammaSCEnergyCorrectionAlgo::nCrystalsGT2Sigma(reco::BasicCluster const & seed, EcalRecHitCollection const &rhc) const {
  // return number of crystals 2Sigma above
  // electronic noise
  
  std::vector<std::pair<DetId,float > > const & hits = seed.hitsAndFractions();

  if (verbosity_ <= pINFO)
  {
    std::cout << "      HiEgammaSCEnergyCorrectionAlgo::nCrystalsGT2Sigma" << std::endl;
    std::cout << "      Will calculate number of crystals above 2sigma noise" << std::endl;
    std::cout << "      sigmaElectronicNoise = " << sigmaElectronicNoise_ << std::endl;
    std::cout << "      There are " << hits.size() << " recHits" << std::endl;
  }

  int nCry = 0;
  for(std::vector<std::pair<DetId,float > >::const_iterator hit = hits.begin(); hit != hits.end(); ++hit)
    {
      // need to get hit by DetID in order to get energy
      EcalRecHitCollection::const_iterator aHit = rhc.find((*hit).first);
      // better the hit to exists....
      if(aHit->energy()>2.f*sigmaElectronicNoise_) ++nCry;
    }

  if (verbosity_ <= pINFO)
  {
    std::cout << "         " << nCry << " of these above 2sigma noise" << std::endl;  
  }
 
  return nCry;
}

