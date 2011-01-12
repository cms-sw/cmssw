/**
   \file
   Implementation of class EcalSeverityLevelAlgo

   \author Stefano Argiro
   \version $Id$
   \date 10 Jan 2011
*/



#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


EcalSeverityLevelAlgo::EcalSeverityLevelAlgo(const edm::ParameterSet& p){
  flagMask_      = p.getParameter< std::vector<uint32_t> >("flagMask");
  dbstatusMask_  = p.getParameter< std::vector<uint32_t> >("dbstatusMask");
	    
}


EcalSeverityLevelAlgo::EcalSeverityLevel 
EcalSeverityLevelAlgo::severityLevel(const DetId& id, 	   
				     const EcalRecHitCollection& rhs, 
				     const edm::EventSetup& es) const{

  // if the detid is within our rechits, evaluate from flag
  EcalRecHitCollection::const_iterator rh;
  if ( (rh=rhs.find(id))  != rhs.end()  )
    return severityLevel(*rh);
  
  // else evaluate from dbstatus
  
  edm::ESHandle<EcalChannelStatus> pChannelStatus;
  es.get<EcalChannelStatusRcd>().get(pChannelStatus);
  const EcalChannelStatus* chStatus = pChannelStatus.product();

  return severityLevel(id,rhs,*chStatus);

}

 

EcalSeverityLevelAlgo::EcalSeverityLevel 
EcalSeverityLevelAlgo::severityLevel(const DetId& id, 	   
				     const EcalRecHitCollection& rhs, 
				     const EcalChannelStatus& chs) const{


  EcalChannelStatus::const_iterator chIt = chs.find( id );
  uint16_t dbStatus = 0;
  if ( chIt != chs.end() ) {
    dbStatus = chIt->getStatusCode();
  } else {
    edm::LogError("ObjectNotFound") << "No channel status found for xtal " 
	 << id.rawId() 
	 << "! something wrong with EcalChannelStatus in your DB? ";
  }
  // check if the bit corresponding to that dbStatus is set in the mask
  // This implementation implies that the statuses have a priority
  for (size_t i=0; i< dbstatusMask_.size();++i){
     if ((dbstatusMask_[i] & 0x1<<dbStatus)) return EcalSeverityLevel(i);
  }

  // no matching
  edm::LogWarning("EcalSeverityLevelAlgo")<< "Unmatched DB status, returning kBad";
  return kBad;
}


EcalSeverityLevelAlgo::EcalSeverityLevel 
EcalSeverityLevelAlgo::severityLevel(const EcalRecHit& rh) const{
   
  // check if the bit corresponding to that dbStatus is set in the mask
  // This implementation implies that the severity have a priority... 
  for (size_t i=0; i< flagMask_.size();++i){
    for (int flag=0; flag<EcalRecHit::kUnknown; ++flag){
 
      if (flagMask_[i] & (rh.checkFlag(flag)<<flag) ) 
	return EcalSeverityLevel(i);
      
    }
  }

  // no matching
  edm::LogWarning("EcalSeverityLevelAlgo")<< "Unmatched Flag , returning kBad";
  return kBad;
}




// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "scram b -k"
// End:
