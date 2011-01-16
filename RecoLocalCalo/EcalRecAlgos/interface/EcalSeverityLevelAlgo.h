/**
   \file
   Declaration of class EcalSeverityLevelAlgo

   \author Stefano Argiro
   \version $Id: EcalSeverityLevelAlgo.h,v 1.27 2011/01/12 13:40:31 argiro Exp $
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

     Instances of this class are supposed to be retrieved from the EventSetup
     via the EcalSeverityLevelESProducer.
     Do not cache the algorithm, or the channel status will not be updated.
     
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
				  const EcalRecHitCollection& rhs) const;


  /// Evaluate status from rechit, using its EcalRecHit::Flag
  EcalSeverityLevel severityLevel(const EcalRecHit& rh) const;


  /// Set the ChannelStatus record. 
  void setChannelStatus(const EcalChannelStatus& chs){chStatus_=&chs;}

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

  const EcalChannelStatus * chStatus_;
};


#endif // __EcalSeverityLevelAlgo_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; scram b -k"
// End:
