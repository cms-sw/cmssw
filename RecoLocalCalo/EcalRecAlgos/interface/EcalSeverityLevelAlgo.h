/**
   \file
   Declaration of class EcalSeverityLevelAlgo

   \author Stefano Argiro
   \version $Id$
   \date 10 Jan 2011
*/

#ifndef __EcalSeverityLevelAlgo_h_
#define __EcalSeverityLevelAlgo_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"


#include <vector>

/**
     Combine information from EcalRecHit::Flag and
     DBStatus into a simpler EcalSeverityLevel flag 
     defined here.
 
     
 */

class EcalRecHit;
class DetId;


class  EcalSeverityLevelAlgo{


public:

  /** Levels of severity:
   * - kGood        --> good channel
   * - kProblematic --> problematic (e.g. noisy)
   * - kRecovered   --> recovered (e.g. an originally dead or saturated)
   * - kTime        --> the channel is out of time (e.g. spike)
   * - kWeird       --> weird (e.g. spike)
   * - kBad         --> bad, not suitable to be used in the reconstruction 
   */
  enum EcalSeverityLevel { kGood=0, kProblematic, kRecovered, kTime, kWeird, kBad };

  explicit EcalSeverityLevelAlgo(const edm::ParameterSet& p);

  /// Evaluate status from id
  /** If the id is in the collection, use the EcalRecHit::Flag
      else use the channelStatus from DB 
   */
  EcalSeverityLevel severityLevel(const DetId& id, 
				  const EcalRecHitCollection& rhs, 
				  const edm::EventSetup& es) const;

  /// same as above but the client will have to retrieve the chstatus record
  EcalSeverityLevel severityLevel(const DetId& id, 
				  const EcalRecHitCollection& rhs, 
				  const EcalChannelStatus& chs) const;
  
  

  /// Evaluate status from rechit, using its EcalRecHit::Flag
  EcalSeverityLevel severityLevel(const EcalRecHit& rh) const;

private:

 
  /// Configure which EcalRecHit::Flag is mapped into which EcalSeverityLevel
  /** The position in the vector is the EcalSeverityLevel
      The content defines which EcalRecHit::Flag should be mapped into that EcalSeverityLevel
      in a bit-wise way */
  std::vector<uint32_t> flagMask_;
  

  /// Configure which DBStatus::Flag is mapped into which EcalSeverityLevel
  /** The position in the vector is the EcalSeverityLevel
      The content defines which EcalRecHit::Flag should be mapped into that EcalSeverityLevel 
      in a bit-wise way*/
  std::vector<uint32_t> dbstatusMask_;

};


#endif // __EcalSeverityLevelAlgo_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
