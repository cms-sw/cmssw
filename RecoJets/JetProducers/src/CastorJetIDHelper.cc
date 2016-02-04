#include "RecoJets/JetProducers/interface/CastorJetIDHelper.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"

#include "TMath.h"
#include <vector>
#include <numeric>
#include <iostream>


reco::helper::CastorJetIDHelper::CastorJetIDHelper()
{
 
  initValues();
}
  
void reco::helper::CastorJetIDHelper::initValues()
{
      emEnergy_ = 0.0;
      hadEnergy_ = 0.0; 
      fem_ = 0.0;
      width_ = 0.0;
      depth_ = 0.0;
      fhot_ = 0.0;
      sigmaz_ = 0.0;
      nTowers_ = 0;
}


void reco::helper::CastorJetIDHelper::calculate( const edm::Event& event, const reco::BasicJet &jet )
{
  initValues();
  
  // calculate Castor JetID properties
 
		double zmean = 0.;
		double z2mean = 0.;
	
		std::vector<CandidatePtr> ccp = jet.getJetConstituents();
		std::vector<CandidatePtr>::const_iterator itParticle;
   		for (itParticle=ccp.begin();itParticle!=ccp.end();++itParticle){	    
        		const CastorTower* castorcand = dynamic_cast<const CastorTower*>(itParticle->get());
			emEnergy_ += castorcand->emEnergy();
			hadEnergy_ += castorcand->hadEnergy();
			depth_ += castorcand->depth()*castorcand->energy();
			width_ += pow(phiangle(castorcand->phi() - jet.phi()),2)*castorcand->energy();
      			fhot_ += castorcand->fhot()*castorcand->energy();
			
			// loop over rechits
      			for (edm::RefVector<edm::SortedCollection<CastorRecHit> >::iterator it = castorcand->rechitsBegin(); it != castorcand->rechitsEnd(); it++) {
	                         edm::Ref<edm::SortedCollection<CastorRecHit> > rechit_p = *it;	                        
	                         double Erechit = rechit_p->energy();
	                         HcalCastorDetId id = rechit_p->id();
	                         int module = id.module();	                                
                                 double zrechit = 0;	 
                                 if (module < 3) zrechit = -14390 - 24.75 - 49.5*(module-1);	 
                                 if (module > 2) zrechit = -14390 - 99 - 49.5 - 99*(module-3);	 
                                 zmean += Erechit*zrechit;	 
                                 z2mean += Erechit*zrechit*zrechit;
      			} // end loop over rechits
			
			nTowers_++;
		}
		//cout << "" << endl;
		
		depth_ = depth_/jet.energy();
		width_ = sqrt(width_/jet.energy());
		fhot_ = fhot_/jet.energy();
		fem_ = emEnergy_/jet.energy();
		
		zmean = zmean/jet.energy();
    		z2mean = z2mean/jet.energy();
    		double sigmaz2 = z2mean - zmean*zmean;
    		if(sigmaz2 > 0) sigmaz_ = sqrt(sigmaz2);

  
}

// help function to calculate phi within [-pi,+pi]
double reco::helper::CastorJetIDHelper::phiangle (double testphi) {
  double phi = testphi;
  while (phi>M_PI) phi -= (2*M_PI);
  while (phi<-M_PI) phi += (2*M_PI);
  return phi;
}



