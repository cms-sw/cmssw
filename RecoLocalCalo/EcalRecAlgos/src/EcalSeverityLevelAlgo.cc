/**
   \file
   Implementation of class EcalSeverityLevelAlgo

   \author Stefano Argiro
   \version $Id: EcalSeverityLevelAlgo.cc,v 1.44 2011/07/14 13:49:10 zafar Exp $
   \date 10 Jan 2011
*/



#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/Utils/interface/StringToEnumValue.h"

EcalSeverityLevelAlgo::EcalSeverityLevelAlgo(const edm::ParameterSet& p){
  
  
  timeThresh_    = p.getParameter< double> ("timeThresh");
  chStatus_ =0;	    

  const edm::ParameterSet & ps=p.getParameter< edm::ParameterSet >("flagMask");
  std::vector<std::string> severities = ps.getParameterNames();
  std::vector<std::string> flags;
 
  flagMask_.resize(severities.size());

  // read configuration of severities

  for (unsigned int is=0;is!=severities.size();++is){

    EcalSeverityLevel::SeverityLevel snum=
      (EcalSeverityLevel::SeverityLevel) StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severities[is]);
    flags=ps.getParameter<std::vector<std::string> >(severities[is]);
    uint32_t mask=0;
    for (unsigned int ifi=0;ifi!=flags.size();++ifi){
      EcalRecHit::Flags f=
	(EcalRecHit::Flags)StringToEnumValue<EcalRecHit::Flags>(flags[ifi]);
      //manipulate the mask
      mask|=(0x1<<f);
    }
    flagMask_[snum]=mask;
  }
  // read configuration of dbstatus

  const edm::ParameterSet & dbps=
    p.getParameter< edm::ParameterSet >("dbstatusMask");
  std::vector<std::string> dbseverities = dbps.getParameterNames();
  std::vector<uint32_t>    dbflags;
 
  dbstatusMask_.resize(dbseverities.size());

  for (unsigned int is=0;is!=dbseverities.size();++is){

    EcalSeverityLevel::SeverityLevel snum=
      (EcalSeverityLevel::SeverityLevel) StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severities[is]);
    
    dbflags=dbps.getParameter<std::vector<uint32_t> >(severities[is]);
    uint32_t mask=0;
    for (unsigned int ifi=0;ifi!=dbflags.size();++ifi){
      int f= dbflags[ifi];
      
      //manipulate the mask
      mask|=(0x1<<f);
    }
    dbstatusMask_[snum]=mask;
  }

  

}


EcalSeverityLevel::SeverityLevel 
EcalSeverityLevelAlgo::severityLevel(const DetId& id, 	   
				     const EcalRecHitCollection& rhs) const{
  
  using namespace EcalSeverityLevel;

  // if the detid is within our rechits, evaluate from flag
 EcalRecHitCollection::const_iterator rh= rhs.find(id);
  if ( rh != rhs.end()  )
    return severityLevel(*rh);


  // else evaluate from dbstatus
 
  if (!chStatus_)     
    edm::LogError("ObjectNotFound") << "Channel Status not set for EcalSeverityLevelAlgo"; 
	

  EcalChannelStatus::const_iterator chIt = chStatus_->find( id );
  uint16_t dbStatus = 0;
  if ( chIt != chStatus_->end() ) {
    dbStatus = chIt->getStatusCode() & 0x1F;
  } else {
    edm::LogError("ObjectNotFound") << "No channel status found for xtal " 
	 << id.rawId() 
	 << "! something wrong with EcalChannelStatus in your DB? ";
  }
 
  // kGood==0 we know!
  if (0==dbStatus)  return kGood;
 
  // check if the bit corresponding to that dbStatus is set in the mask
  // This implementation implies that the statuses have a priority
  for (size_t i=0; i< dbstatusMask_.size();++i){
    uint32_t tmp = 0x1<<dbStatus;
    if (dbstatusMask_[i] & tmp) return SeverityLevel(i);
  }

  // no matching
  LogDebug("EcalSeverityLevelAlgo")<< 
    "Unmatched DB status, returning kGood";
  return kGood;
}


EcalSeverityLevel::SeverityLevel 
EcalSeverityLevelAlgo::severityLevel(const EcalRecHit& rh) const{

  using namespace EcalSeverityLevel;

  //if marked good, do not do any further test
  if (rh.checkFlag(kGood)) return kGood;

  // check if the bit corresponding to that flag is set in the mask
  // This implementation implies that  severities have a priority... 
  for (int sev=kBad;sev>=0;--sev){
    if(sev==kTime && rh.energy() < timeThresh_ ) continue;
    if (rh.checkFlagMask(flagMask_[sev])) return SeverityLevel(sev);
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
