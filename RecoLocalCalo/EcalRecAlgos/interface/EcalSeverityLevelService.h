/**
   \file
   Implementation of class EcalSeverityLevelService

   \author Stefano Argiro
   \version $Id: EcalSeverityLevelService.h,v 1.1 2011/01/12 21:55:53 argiro Exp $
   \date 11 Jan 2011
*/

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"


/// A service to retrieve to provide a hook to EcalSeverityLevelAlgo
class EcalSeverityLevelService {
  
public:
  EcalSeverityLevelService(const edm::ParameterSet& p, 
			   edm::ActivityRegistry& r){
    algo_ = new EcalSeverityLevelAlgo(p);
  }

  ~EcalSeverityLevelService(){delete algo_;}
  

  EcalSeverityLevelAlgo::EcalSeverityLevel 
  severityLevel(const EcalRecHit& rh) const {
    return algo_->severityLevel(rh);
  }

  EcalSeverityLevelAlgo::EcalSeverityLevel 
  severityLevel(const DetId& id, 
		const EcalRecHitCollection& rhs, 
		const edm::EventSetup& es) const {
    return algo_->severityLevel(id,rhs,es);
  }
  
  
  EcalSeverityLevelAlgo::EcalSeverityLevel 
  severityLevel(const DetId& id, 
		const EcalRecHitCollection& rhs, 
		const EcalChannelStatus& chs) const {
    return algo_->severityLevel(id,rhs,chs);
  }
  
  const EcalSeverityLevelAlgo* getAlgorithm() const {return algo_;}

private:
  EcalSeverityLevelAlgo * algo_;

};


// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; scram b"
// End:
