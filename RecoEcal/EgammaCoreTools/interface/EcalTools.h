/**
   \file
   Declaration of class EcalTools

   \author Stefano Argiro
   \version $Id: EcalTools.h,v 1.5 2011/05/19 14:41:38 argiro Exp $
   \date 11 Jan 2011
*/

#ifndef __EcalTools_h_
#define __EcalTools_h_

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

class EcalTools{
  
public:
  
  /// the good old 1-e4/e1. Ignore hits below recHitThreshold
  static float swissCross( const DetId& id, 
			   const EcalRecHitCollection & recHits, 
			   float recHitThreshold , 
			   bool avoidIeta85=true);

  /// true if the channel is near a dead one (in the 3x3)
  /** This function will use the ChannelStatus record to determine
      if the channel is next to a dead one, using bit 12 of the
      channel status word
   */
  static bool isNextToDead( const DetId& id, const edm::EventSetup& es);

  /// same as isNextToDead, but will use information from the neighbour
  /** Looks at neighbours in 3x3 and returns true if one of them has
      chStatus above chStatusThreshold.
      Use sparingly, slow. Normally isNextToDead() should be used instead.
        
   */

  static bool isNextToDeadFromNeighbours( const DetId& id, 
					  const EcalChannelStatus& chs,
					  int chStatusThreshold) ; 

  /// true if near a crack or ecal border
  static bool isNextToBoundary (const DetId& id);

  /// return true if the channel at offsets dx,dy is dead 
  static bool deadNeighbour(const DetId& id, const EcalChannelStatus& chs, 
			    int chStatusThreshold, 
			    int dx, int dy); 

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
