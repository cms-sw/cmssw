#include "Calibration/Tools/interface/CalibElectron.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "Calibration/Tools/interface/EcalRingCalibrationTools.h"
#include "Calibration/Tools/interface/EcalIndexingTools.h"
#include <iostream>

using namespace calib;
using namespace std;


CalibElectron::CalibElectron() : theElectron_(0), theHits_(0), theEEHits_(0)
{
}


std::vector< std::pair<int,float> > CalibElectron::getCalibModulesWeights(TString calibtype)
{
  std::vector< std::pair<int,float> > theWeights;

  if (calibtype == "RING")
    {
      float w_ring[EcalRingCalibrationTools::N_RING_TOTAL];

      for (int i=0;i<EcalRingCalibrationTools::N_RING_TOTAL;++i)
	w_ring[i]=0.;
      
      std::vector<DetId> scDetIds = theElectron_->superCluster()->getHitsByDetId();

      
      for(std::vector<DetId>::const_iterator idIt=scDetIds.begin(); idIt!=scDetIds.end(); ++idIt){
    
	const EcalRecHit* rh=0;
	if ( (*idIt).subdetId() == EcalBarrel) 
	  rh = &*(theHits_->find(*idIt));
	else if ( (*idIt).subdetId() == EcalEndcap) 
	  rh = &*(theEEHits_->find(*idIt));
	if (!rh)
	  std::cout << "CalibElectron::BIG ERROR::RecHit NOT FOUND" << std::endl;  
	w_ring[EcalRingCalibrationTools::getRingIndex(*idIt)]+=rh->energy();
	
      }

      for (int i=0;i<EcalRingCalibrationTools::N_RING_TOTAL;++i)
	if (w_ring[i]!=0.) 
	  theWeights.push_back(std::pair<int,float>(i,w_ring[i]));
	  // cout << " ring " << i << " - energy sum " << w_ring[i] << endl;
    }

  else if(calibtype == "MODULE")
    {

      float w_ring[EcalRingCalibrationTools::N_MODULES_BARREL];

      for (int i=0;i<EcalRingCalibrationTools::N_MODULES_BARREL;++i)
        w_ring[i]=0.;

      std::vector<DetId> scDetIds = theElectron_->superCluster()->getHitsByDetId();


      for(std::vector<DetId>::const_iterator idIt=scDetIds.begin(); idIt!=scDetIds.end(); ++idIt){

        const EcalRecHit* rh=0;
        if ( (*idIt).subdetId() == EcalBarrel)
          rh = &*(theHits_->find(*idIt));
        else if ( (*idIt).subdetId() == EcalEndcap)
          rh = &*(theEEHits_->find(*idIt));
        if (!rh)
          std::cout << "CalibElectron::BIG ERROR::RecHit NOT FOUND" << std::endl;
        w_ring[EcalRingCalibrationTools::getModuleIndex(*idIt)]+=rh->energy();

      }

      for (int i=0;i<EcalRingCalibrationTools::N_MODULES_BARREL;++i)
        if (w_ring[i]!=0.)
          theWeights.push_back(std::pair<int,float>(i,w_ring[i]));
      // cout << " ring " << i << " - energy sum " << w_ring[i] << endl;                            

    }

  else if(calibtype == "ABS_SCALE")
    {
      std::cout<< "ENTERING CalibElectron, ABS SCALE mode"<<std::endl;

      float w_ring(0.);
      
      std::vector<DetId> scDetIds = theElectron_->superCluster()->getHitsByDetId();
      
      
      for(std::vector<DetId>::const_iterator idIt=scDetIds.begin(); idIt!=scDetIds.end(); ++idIt){
	
        const EcalRecHit* rh=0;
        if ( (*idIt).subdetId() == EcalBarrel)
          rh = &*(theHits_->find(*idIt));
        else if ( (*idIt).subdetId() == EcalEndcap)
          rh = &*(theEEHits_->find(*idIt));
        if (!rh)
          std::cout << "CalibElectron::BIG ERROR::RecHit NOT FOUND" << std::endl;
	
        w_ring += rh->energy();
	
      }
      
      if(w_ring != 0.)
	theWeights.push_back(std::pair<int,float>(0,w_ring));
      cout << " ABS SCALE  - energy sum " << w_ring << endl;                            
      
    }
  
  else if(calibtype == "ETA_ET_MODE")
    {

      float w_ring[ 200 ];
      
      for (int i=0; i< EcalIndexingTools::getInstance()->getNumberOfChannels(); ++i)
        w_ring[i]=0.;

      std::vector<DetId> scDetIds = theElectron_->superCluster()->getHitsByDetId();


      for(std::vector<DetId>::const_iterator idIt=scDetIds.begin(); idIt!=scDetIds.end(); ++idIt){

        const EcalRecHit* rh=0;
        if ( (*idIt).subdetId() == EcalBarrel)
          rh = &*(theHits_->find(*idIt));
        else if ( (*idIt).subdetId() == EcalEndcap)
          rh = &*(theEEHits_->find(*idIt));
        if (!rh)
          std::cout << "CalibElectron::BIG ERROR::RecHit NOT FOUND" << std::endl;
	
	float eta = fabs( theElectron_->eta() );
	float theta = 2. * atan( exp(- eta) );
	float et = theElectron_->superCluster()->energy() * sin(theta);
	
	int in = EcalIndexingTools::getInstance()->getProgressiveIndex(eta, et);
	
	w_ring[in]+=rh->energy();
	//w_ring[in]+=theElectron_->superCluster()->energy();

	std::cout << "CalibElectron::filling channel " << in << " with value " << theElectron_->superCluster()->energy() << std::endl;
      }
      
      for (int i=0; i< EcalIndexingTools::getInstance()->getNumberOfChannels(); ++i){
        if (w_ring[i]!=0.){
          theWeights.push_back(std::pair<int,float>(i,w_ring[i]));
	  cout << " ring " << i << " - energy sum " << w_ring[i] << endl;                            
	}
	
      }
      
    }
  
  else 
    {
      cout << "CalibType not yet implemented" << endl;
      
    }
  
  return theWeights;

}

