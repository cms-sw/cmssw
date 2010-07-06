#include "RecoJets/JetProducers/interface/CastorJetIDHelper.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorCell.h"

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
			
			// loop over cells
                        for (CastorCell_iterator it = castorcand->cellsBegin(); it != castorcand->cellsEnd(); it++) {
                                CastorCellRef cell_p = *it;
                                math::XYZPointD rcell = cell_p->position();
                                double Ecell = cell_p->energy();
                                zmean += Ecell*cell_p->z();
                                z2mean += Ecell*cell_p->z()*cell_p->z();
                        } // end loop over cells
			
			nTowers_++;
		}
		//cout << "" << endl;
		
		depth_ = depth_/jet.energy();
		width_ = sqrt(width_/jet.energy());
		fhot_ = fhot_/jet.energy();
		
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



