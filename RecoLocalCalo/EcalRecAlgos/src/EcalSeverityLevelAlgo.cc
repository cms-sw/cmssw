/**
   \file
   Implementation of class EcalSeverityLevelAlgo

   \author Stefano Argiro
   \version $Id: EcalSeverityLevelAlgo.cc,v 1.33 2011/01/12 13:40:32 argiro Exp $
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
  
  edm::ESHandle<EcalChannelStatus> pChannelStatus;
  es.get<EcalChannelStatusRcd>().get(pChannelStatus);
  const EcalChannelStatus* chStatus = pChannelStatus.product();

  return severityLevel(id,rhs,*chStatus);

}

 

EcalSeverityLevelAlgo::EcalSeverityLevel 
EcalSeverityLevelAlgo::severityLevel(const DetId& id, 	   
				     const EcalRecHitCollection& rhs, 
				     const EcalChannelStatus& chs) const{


  // if the detid is within our rechits, evaluate from flag
 EcalRecHitCollection::const_iterator rh= rhs.find(id);
  if ( rh != rhs.end()  )
    return severityLevel(*rh);


  // else evaluate from dbstatus
 

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
    uint32_t tmp = 0x1<<dbStatus;
    if (dbstatusMask_[i] & tmp) return EcalSeverityLevel(i);
  }

  // no matching
  LogDebug("EcalSeverityLevelAlgo")<< 
    "Unmatched DB status, returning kGood";
  return kGood;
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
  LogDebug("EcalSeverityLevelAlgo")<< "Unmatched Flag , returning kGood";
  return kGood;
}




// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "scram b -k"
// End:
