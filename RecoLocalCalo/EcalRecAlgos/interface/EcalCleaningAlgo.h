/**
   \file
   Declaration of class EcalCleaningAlgo

   \author Stefano Argiro
   \version $Id: EcalCleaningAlgo.h,v 1.5 2011/05/17 12:07:25 argiro Exp $
   \date 20 Dec 2010
*/

#ifndef __EcalCleaningAlgo_h_
#define __EcalCleaningAlgo_h_


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h" 
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h" 
#include <vector>

class DetId;


class EcalCleaningAlgo{


public:
  
  EcalCleaningAlgo(const edm::ParameterSet& p);
  
  /** check topology, return :
   *  kGood    : not anomalous
      kWeird   : spike
      kDiWeird : dispike */
  EcalRecHit::Flags checkTopology(const DetId& id,
				  const EcalRecHitCollection& rhs);

  void setFlags(EcalRecHitCollection& rhs);
  
private:
  
  /// yet another function to calculate swiss cross
  float e4e1(const DetId& id, const EcalRecHitCollection& rhs);

  /**  Compute e6 over e2 around xtal 1, where 2 is the most energetic in 
       the swiss cross around 1
  
             | | |   
           +-+-+-+-+
           | |1|2| |
           +-+-+-+-+
             | | |               */
  float e6e2 (const DetId& id, const EcalRecHitCollection& rhs);

  float recHitE( const DetId id, 
		 const EcalRecHitCollection &recHits,
		 bool  useTimingInfo);
  
  /// in EB, check if we are near a crack
  bool isNearCrack(const DetId& detid);

  /// return the id of the  4 neighbours in the swiss cross
  const std::vector<DetId> neighbours(const DetId& id);

  ///ignore kOutOfTime above threshold when calculating e4e1
  float ignoreOutOfTimeThresh_;

  // Parameters for tolopogical cut 
  // mark anomalous if e> cThreshold &&  e4e1> a*log10(e1e1)+b
  float cThreshold_barrel_;
  float cThreshold_endcap_;         
  float e4e1_a_barrel_;
  float e4e1_b_barrel_;
  float e4e1_a_endcap_;
  float e4e1_b_endcap_;
  // when calculating e4/e1, ignore hits below this threshold
  float e4e1Treshold_barrel_;
  float e4e1Treshold_endcap_;
  float tightenCrack_e1_single_;
  float tightenCrack_e4e1_single_;
  float cThreshold_double_;
  float tightenCrack_e1_double_;
  float tightenCrack_e6e2_double_;
  float e6e2thresh_;

};

#endif // __EcalCleaningAlgo_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
