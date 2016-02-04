/**
   \file
   Declaration of class EcalSeverityLevelAlgo

   \author Stefano Argiro
   \version $Id: EcalSeverityLevelAlgo.h,v 1.30 2011/04/12 08:04:25 argiro Exp $
   \date 10 Jan 2011
*/

#ifndef __EcalSeverityLevelAlgo_h_
#define __EcalSeverityLevelAlgo_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalSeverityLevel.h"
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

 
  explicit EcalSeverityLevelAlgo(const edm::ParameterSet& p);


  /// Evaluate status from id
  /** If the id is in the collection, use the EcalRecHit::Flag
      else use the channelStatus from DB 
   */
  EcalSeverityLevel::SeverityLevel severityLevel(const DetId& id, 
				  const EcalRecHitCollection& rhs) const;


  /// Evaluate status from rechit, using its EcalRecHit::Flag
  EcalSeverityLevel::SeverityLevel severityLevel(const EcalRecHit& rh) const;


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


  /// Return kTime only if the rechit is flagged kOutOfTime and E>timeThresh_
  float timeThresh_;

  const EcalChannelStatus * chStatus_;
};


#endif // __EcalSeverityLevelAlgo_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; scram b -k"
// End:
