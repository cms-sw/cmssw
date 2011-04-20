/**
   \file
   Declaration of class EcalTools

   \author Stefano Argiro
   \version $Id: EcalTools.h,v 1.1 2011/01/12 14:46:08 argiro Exp $
   \date 11 Jan 2011
*/

#ifndef __EcalTools_h_
#define __EcalTools_h_

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

class EcalTools{
  
public:
  
  /// the good old 1-e4/e1. Ignore hits below recHitThreshold
  static float swissCross( const DetId& id, 
			   const EcalRecHitCollection & recHits, 
			   float recHitThreshold , 
			   bool avoidIeta85=true);

  /// true if the channel is near a dead one
  static bool isNextToDead( const DetId& id, const edm::EventSetup& es);


  /// true if near a crack or ecal border
  static bool isNextToBoundary (const DetId& id);

  /// return true if the channel at offsets dx,dy is dead 
  static bool deadNeighbour(const DetId& id, const edm::EventSetup& es, 
			    int dx, int dy); 

private:

  static float recHitE( const DetId id, const EcalRecHitCollection &recHits );
  static float recHitE( const DetId id, const EcalRecHitCollection & recHits, 
			int dEta, int dPhi );
  static float recHitApproxEt( const DetId id, 
			       const EcalRecHitCollection &recHits );
  static uint16_t getChannelStatus(const DetId& id, const edm::EventSetup& es);


};


#endif // __EcalTools_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "scram b -k"
// End:
