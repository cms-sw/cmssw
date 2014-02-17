/* Implementation of class EcalCleaningAlgo
   \author Stefano Argiro
   \version $Id: EcalCleaningAlgo.cc,v 1.9 2011/05/17 12:07:23 argiro Exp $
   \date 20 Dec 2010
*/    


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h" 

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalCleaningAlgo.h"

EcalCleaningAlgo::EcalCleaningAlgo(const edm::ParameterSet&  p){
  
  cThreshold_barrel_   = p.getParameter<double>("cThreshold_barrel");;
  cThreshold_endcap_   = p.getParameter<double>("cThreshold_endcap");;  
  e4e1_a_barrel_       = p.getParameter<double>("e4e1_a_barrel");
  e4e1_b_barrel_       = p.getParameter<double>("e4e1_b_barrel");
  e4e1_a_endcap_       = p.getParameter<double>("e4e1_a_endcap");
  e4e1_b_endcap_       = p.getParameter<double>("e4e1_b_endcap");
  e4e1Treshold_barrel_ = p.getParameter<double>("e4e1Threshold_barrel");
  e4e1Treshold_endcap_ = p.getParameter<double>("e4e1Threshold_endcap");

  cThreshold_double_   = p.getParameter<double>("cThreshold_double"); 

  ignoreOutOfTimeThresh_   =p.getParameter<double>("ignoreOutOfTimeThresh");  
  tightenCrack_e1_single_  =p.getParameter<double>("tightenCrack_e1_single");
  tightenCrack_e4e1_single_=p.getParameter<double>("tightenCrack_e4e1_single");
  tightenCrack_e1_double_  =p.getParameter<double>("tightenCrack_e1_double");
  tightenCrack_e6e2_double_=p.getParameter<double>("tightenCrack_e6e2_double");
  e6e2thresh_=              p.getParameter<double>("e6e2thresh");


  
}




/**
   Flag spikey channels
   
   Mark single spikes. Spike definition:
 
      Barrel: e> cThreshold_barrel_  &&
              e4e1 > e4e1_a_barrel_ * log10(e) + e4e1_b_barrel_

      Near cracks: energy threshold is multiplied by tightenCrack_e1_single
                   e4e1 threshold is divided by tightenCrack_e4e1_single

      Endcap : e> cThreshold_endcap_ &&
               e4e1>   e4e1_a_endcap_ * log10(e) + e4e1_b_endcap_

   Mark double spikes    (barrel only)
      e> cThreshold_double_ &&
      e6e2 >  e6e2thresh_;

   Near cracks:
          energy threshold multiplied by   tightenCrack_e1_double    
          e6e2 threshold divided by tightenCrack_e6e2_double


   Out of time hits above e4e1_IgnoreOutOfTimeThresh_  are 
   ignored in topological quantities   
 */

EcalRecHit::Flags 
EcalCleaningAlgo::checkTopology(const DetId& id,
				const EcalRecHitCollection& rhs){


  float a=0,b=0,e4e1thresh=0,ethresh=0;


  if( id.subdetId() == EcalBarrel) {
    a= e4e1_a_barrel_;
    b= e4e1_b_barrel_; 
    ethresh=cThreshold_barrel_;
   
  }
  else if( id.subdetId() == EcalEndcap){
    a= e4e1_a_endcap_;
    b= e4e1_b_endcap_;
    ethresh=cThreshold_endcap_;
   }
  
  

  // for energies below threshold, we don't apply e4e1 cut
  float energy = recHitE(id,rhs,false);
 
  if (energy< ethresh) return EcalRecHit::kGood;
  if (isNearCrack(id) && energy < ethresh*tightenCrack_e1_single_) 
    return EcalRecHit::kGood;


  float e4e1value = e4e1(id,rhs);
  e4e1thresh = a* log10(energy) + b;

  // near cracks the cut is tighter by a factor 
  if (isNearCrack(id)) {
    e4e1thresh/=tightenCrack_e4e1_single_;
  }

  // identify spike
  if (e4e1value < e4e1thresh) return EcalRecHit::kWeird; 
  


  // now for double spikes
 
  // no checking for double spikes in EE
  if( id.subdetId() == EcalEndcap) return EcalRecHit::kGood;

  float e6e2value = e6e2(id,rhs);
  float e6e2thresh = e6e2thresh_ ;
  if (isNearCrack(id) && energy < cThreshold_double_ *tightenCrack_e1_double_ )
    return EcalRecHit::kGood;

  if  (energy <  cThreshold_double_) return EcalRecHit::kGood;
  
  // near cracks the cut is tighter by a factor 
  if (id.subdetId() == EcalBarrel && isNearCrack(id)) 
    e6e2thresh/=tightenCrack_e6e2_double_;

  // identify double spike
  if (e6e2value < e6e2thresh) return EcalRecHit::kDiWeird; 

  return EcalRecHit::kGood;

}






float EcalCleaningAlgo::e4e1(const DetId& id, 
			     const EcalRecHitCollection& rhs){

 
  float s4 = 0;
  float e1 = recHitE( id, rhs, false );
  
  
  if ( e1 == 0 ) return 0;
  const std::vector<DetId>& neighs =  neighbours(id);
  for (size_t i=0; i<neighs.size(); ++i)
    // avoid hits out of time when making s4
    s4+=recHitE(neighs[i],rhs, true);
  
  return s4 / e1;
  
 
}




float EcalCleaningAlgo::e6e2(const DetId& id, 
			     const EcalRecHitCollection& rhs){

    float s4_1 = 0;
    float s4_2 = 0;
    float e1 = recHitE( id, rhs , false );


    float maxene=0;
    DetId maxid;

    if ( e1 == 0 ) return 0;

    const std::vector<DetId>& neighs =  neighbours(id);

    // find the most energetic neighbour ignoring time info
    for (size_t i=0; i<neighs.size(); ++i){
      float ene = recHitE(neighs[i],rhs,false);
      if (ene>maxene)  {
	maxene=ene;
	maxid = neighs[i];
      }
    }

    float e2=maxene;

    s4_1 = e4e1(id,rhs)* e1;
    s4_2 = e4e1(maxid,rhs)* e2;

    return (s4_1 + s4_2) / (e1+e2) -1. ;

}




float EcalCleaningAlgo::recHitE( const DetId id, 
				 const EcalRecHitCollection &recHits,
                                 bool useTimingInfo )
{
  if ( id.rawId() == 0 ) return 0;
  

  float threshold = e4e1Treshold_barrel_;
  if ( id.subdetId() == EcalEndcap) threshold = e4e1Treshold_endcap_; 

  EcalRecHitCollection::const_iterator it = recHits.find( id );
  if ( it != recHits.end() ){
    float ene= (*it).energy();

    // ignore out of time in EB when making e4e1 if so configured
    if (useTimingInfo){

      if (id.subdetId()==EcalBarrel &&
	  it->checkFlag(EcalRecHit::kOutOfTime) 
	  && ene>ignoreOutOfTimeThresh_) return 0;
    }

    // ignore hits below threshold
    if (ene < threshold) return 0;

    // else return the energy of this hit
    return ene;
  }
  return 0;
}

/// four neighbours in the swiss cross around id
const std::vector<DetId> EcalCleaningAlgo::neighbours(const DetId& id){
 
  std::vector<DetId> ret;

  if ( id.subdetId() == EcalBarrel) {

    ret.push_back( EBDetId::offsetBy( id,  1, 0 ));
    ret.push_back( EBDetId::offsetBy( id, -1, 0 ));
    ret.push_back( EBDetId::offsetBy( id,  0, 1 ));
    ret.push_back( EBDetId::offsetBy( id,  0,-1 ));
  }
  // nobody understands what polymorphism is for, sgrunt !
  else  if (id.subdetId() == EcalEndcap) {
    ret.push_back( EEDetId::offsetBy( id,  1, 0 ));
    ret.push_back( EEDetId::offsetBy( id, -1, 0 ));
    ret.push_back( EEDetId::offsetBy( id,  0, 1 ));
    ret.push_back( EEDetId::offsetBy( id,  0,-1 ));

  }


  return ret;

} 


bool EcalCleaningAlgo::isNearCrack(const DetId& id){

  if (id.subdetId() == EcalEndcap) { 
    return EEDetId::isNextToRingBoundary(id);
  } else {
    return EBDetId::isNextToBoundary(id);
  }
}



void EcalCleaningAlgo::setFlags(EcalRecHitCollection& rhs){
  EcalRecHitCollection::iterator rh;
  //changing the collection on place
  for (rh=rhs.begin(); rh!=rhs.end(); ++rh){
    EcalRecHit::Flags state=checkTopology(rh->id(),rhs);
    if (state!=EcalRecHit::kGood) { 
      rh->unsetFlag(EcalRecHit::kGood);
      rh->setFlag(state);
    }
  }
}

