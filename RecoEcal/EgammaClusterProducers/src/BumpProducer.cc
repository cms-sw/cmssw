// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/BumpProducer.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Common/interface/Handle.h"

// C/C++ headers
#include <iostream>
#include <cmath>
#include <vector>
#include <time.h>

//

BumpProducer::BumpProducer(const edm::ParameterSet& ps)
{
  tPhi = ps.getParameter<double>("T_phi");
  tEta = ps.getParameter<double>("T_eta");
  maxE  = ps.getParameter< std::vector<double> >("maxEnergies");
  etas  = ps.getParameter< std::vector<int> >("etas");
  phis  = ps.getParameter< std::vector<int> >("phis");

  meanBackground = ps.getParameter<double>("mean_background");
  backgroundFluctuation = ps.getParameter<double>("background_fluctuation");

  hitCollection_  = ps.getParameter<std::string>("hitCollection_");
  produces< EcalRecHitCollection >(hitCollection_);
  nEvt_ = 0; // reset local event counter
}

BumpProducer::~BumpProducer() 
{
}

// must implement the produce method:
void BumpProducer::produce(edm::Event& evt, const edm::EventSetup& es) 
{
  static const int MIN_IPHI = 1;
  static const int MAX_IETA = 85;
  static const int MAX_IPHI = 360;

  srand(time(NULL));
  
  nEvt_++;
  
  // create the collection of reconstructed hits:
  std::auto_ptr<EcalRecHitCollection> rechits(new EcalRecHitCollection);


 for (int etaIndex = - MAX_IETA; etaIndex <= MAX_IETA; etaIndex++)
    {
      for (int phiIndex = MIN_IPHI; phiIndex <= MAX_IPHI; phiIndex++)
	{
	  // the eta index can't be 0:
	  if (etaIndex == 0) continue;

	  double Energy = 0;
	  double time = 0.01;

	  // Add bump energy:
	  for (unsigned int i = 0; i < maxE.size(); i++)
	    {
	      Energy += getBumpEnergy(maxE[i], phis[i], etas[i], phiIndex, etaIndex);
	    }
	  // Add background energy:
	  Energy += getBackgroundEnergy();

	  // Hit Energy Threshold:
	  if (Energy > 0.5)
	    {
	      // Create the RecHit:
	      EBDetId detID(etaIndex, phiIndex, 0);
	      EcalRecHit hit = EcalRecHit(detID, Energy, time);
	      rechits->push_back(hit);
	    }
	}
    }

  // put the collection of reconstructed hits in the event
  evt.put(rechits, hitCollection_);
}

double BumpProducer::getBumpEnergy(double maxE, int bumpPhi, int bumpEta, int phiIndex, int etaIndex)
{
  // difference in phi:
  int dphi = abs(bumpPhi - phiIndex);
  if (dphi <= 180) dphi = dphi;
  else dphi = 360 - dphi;

  // difference in eta:
  int deta = abs(bumpEta - etaIndex);
  if (bumpEta * etaIndex < 0) deta -= 1; // there is no eta index = 0

  double bumpEnergy = maxE * exp( - dphi / tPhi) * exp( - deta / tEta);
  return bumpEnergy;
}

double BumpProducer::getBackgroundEnergy()
{
  double multiplier = (rand()% 10000 - 5000) / 5000.0; // a number n, for which: -1 <= n <  1
  double bgEnergy = meanBackground + multiplier * backgroundFluctuation;
  return bgEnergy;
}
