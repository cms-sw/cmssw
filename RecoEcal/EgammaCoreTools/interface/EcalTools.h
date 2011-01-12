/**
   \file
   Declaration of class EcalTools

   \author Stefano Argiro
   \version $Id$
   \date 11 Jan 2011
*/

#ifndef __EcalTools_h_
#define __EcalTools_h_

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class EcalTools{
  
public:
  
  static float swissCross( const DetId& id, 
			   const EcalRecHitCollection & recHits, 
			   float recHitThreshold , 
			   bool avoidIeta85=true);

private:

  static float recHitE( const DetId id, const EcalRecHitCollection &recHits );
  static float recHitE( const DetId id, const EcalRecHitCollection & recHits, 
			int dEta, int dPhi );
  static float recHitApproxEt( const DetId id, 
			       const EcalRecHitCollection &recHits );
};


#endif // __EcalTools_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "scram b -k"
// End:
