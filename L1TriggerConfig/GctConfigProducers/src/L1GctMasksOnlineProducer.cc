//-------------------------------------------------
//
//   \class L1GctMasksOnlineProducer
//
//   Description:  A class to produce the L1 GMT emulator Parameters record in the event setup
//                 by reading them from the online database.
//
//   $Date: 2008/11/24 19:00:38 $
//   $Revision: 1.1 $
//
//   Author :
//   Thomas Themel
//
//--------------------------------------------------
#include "L1TriggerConfig/GctConfigProducers/interface/L1GctMasksOnlineProducer.h"

#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondTools/L1Trigger/interface/Exception.h"

using namespace std;
using coral::AttributeList;

//RH_ASSIGN_GROUP(L1GctChannelMasks, TGlobalTriggerGroup)

/** ------------ method called to produce the data  ------------
 *  Query the CMS_GCT.GMT_SOFTWARE_CONFIG table with a key determined by the "master" config table
 *  and return the matching record.
 */
boost::shared_ptr<L1GctChannelMask> L1GctMasksOnlineProducer::newObject( const std::string& objectKey )
{
  using namespace edm::es;


  // Copy data members from L1GctChannelMasks,
  // and  M-x replace-regexp RET .*m_\([a-z:_]*\) RET ADD_FIELD(helper, L1GctChannelMasks, \1) RET

  // new object in smart pointer
  boost::shared_ptr<L1GctChannelMask> ptrResult(new L1GctChannelMask);

  // make SQL query
  l1t::OMDSReader::QueryResults resultLines = 
    m_omdsReader.basicQuery(
          // SELECTed columns
	  "",
	  // schema name
	  "CMS_GMT",
	  // table name
          "GMT_SOFTWARE_CONFIG",
	  // WHERE lhs
	  "GMT_SOFTWARE_CONFIG.KEY",
	  // WHERE rhs
	  m_omdsReader.singleAttribute(objectKey) );

  if(resultLines.numberRows() == 1) {
    const AttributeList&  resultRecord = resultLines.attributeLists().front();
    checkCMSSWVersion(resultRecord);

    // convert sql results to object here!

    return ptrResult;
  }
     
  throw cond::Exception("Couldn't find GMT_SOFTWARE_CONFIG record for GCT key `" + objectKey + "'") ;
}

void L1GctMasksOnlineProducer::checkCMSSWVersion(const coral::AttributeList& configRecord) 
{
  const coral::Attribute& version = configRecord["CMSSW_VERSION"];

  /* If the DB field is unset, take any. */
  if(version.isNull()) {
    edm::LogInfo("No CMSSW version set in database, accepting " PROJECT_VERSION);
    return;
  }

  /* Else make sure we have the correct  version. */
  const string& versionString = version.data<string>();

  /* PROJECT_VERSION is passed as a -D #define from scramv1 (eg CMSSW_2_1_0) */
  if(versionString != PROJECT_VERSION) { 
    string errMsg = "CMSSW version mismatch: Configuration requires " + 
      versionString + ", but this is " PROJECT_VERSION "!";
   
    if(ignoreVersionMismatch_) { 
      edm::LogWarning(errMsg + " (will continue because ignoreVersionMismatch is set)");
    } else { 
      throw cond::Exception(errMsg);	    
    }
  }
}

L1GctMasksOnlineProducer::L1GctMasksOnlineProducer(const edm::ParameterSet& ps) : 
  L1ConfigOnlineProdBase<L1GctChannelMaskRcd, L1GctChannelMask>(ps)
{
  ignoreVersionMismatch_  = ps.getParameter<bool>("ignoreVersionMismatch");
}


L1GctMasksOnlineProducer::~L1GctMasksOnlineProducer() {}

DEFINE_FWK_EVENTSETUP_MODULE(L1GctMasksOnlineProducer);
