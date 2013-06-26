//-------------------------------------------------
//
//   \class L1MuGMTParametersOnlineProducer
//
//   Description:  A class to produce the L1 GMT emulator Parameters record in the event setup
//                 by reading them from the online database.
//
//   $Date: 2012/02/10 14:20:06 $
//   $Revision: 1.3 $
//
//   Author :
//   Thomas Themel
//
//--------------------------------------------------
#include "L1TriggerConfig/GMTConfigProducers/interface/L1MuGMTParametersOnlineProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

/* Define this to see debug output from the record parsing layer. */
//#define RECORDHELPER_DEBUG

#include "L1TriggerConfig/GMTConfigProducers/interface/RecordHelper.h"
#include "L1TriggerConfig/GMTConfigProducers/interface/GTRecordGroup.h"
#include "CondTools/L1Trigger/interface/Exception.h"

using namespace std;
using coral::AttributeList;

RH_ASSIGN_GROUP(L1MuGMTParameters, TGlobalTriggerGroup)

/** ------------ method called to produce the data  ------------
 *  Query the CMS_GMT.GMT_SOFTWARE_CONFIG table with a key determined by the "master" config table
 *  and return the matching record.
 */
boost::shared_ptr<L1MuGMTParameters> L1MuGMTParametersOnlineProducer::newObject( const std::string& objectKey )
{
  using namespace edm::es;

  RecordHelper<L1MuGMTParameters> helper;

  // Copy data members from L1MuGMTParameters,
  // and  M-x replace-regexp RET .*m_\([a-z:_]*\) RET ADD_FIELD(helper, L1MuGMTParameters, \1) RET

  ADD_FIELD(helper, L1MuGMTParameters, EtaWeight_barrel);
  ADD_FIELD(helper, L1MuGMTParameters, PhiWeight_barrel);
  ADD_FIELD(helper, L1MuGMTParameters, EtaPhiThreshold_barrel);
  ADD_FIELD(helper, L1MuGMTParameters, EtaWeight_endcap);
  ADD_FIELD(helper, L1MuGMTParameters, PhiWeight_endcap);
  ADD_FIELD(helper, L1MuGMTParameters, EtaPhiThreshold_endcap);
  ADD_FIELD(helper, L1MuGMTParameters, EtaWeight_COU);
  ADD_FIELD(helper, L1MuGMTParameters, PhiWeight_COU);
  ADD_FIELD(helper, L1MuGMTParameters, EtaPhiThreshold_COU);
  ADD_FIELD(helper, L1MuGMTParameters, CaloTrigger);
  ADD_FIELD(helper, L1MuGMTParameters, IsolationCellSizeEta);
  ADD_FIELD(helper, L1MuGMTParameters, IsolationCellSizePhi);
  ADD_FIELD(helper, L1MuGMTParameters, DoOvlRpcAnd);
  ADD_FIELD(helper, L1MuGMTParameters, PropagatePhi);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodPhiBrl);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodPhiFwd);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodEtaBrl);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodEtaFwd);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodPtBrl);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodPtFwd);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodChargeBrl);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodChargeFwd);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodMIPBrl);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodMIPFwd);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodMIPSpecialUseANDBrl);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodMIPSpecialUseANDFwd);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodISOBrl);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodISOFwd);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodISOSpecialUseANDBrl);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodISOSpecialUseANDFwd);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodSRKBrl);
  ADD_FIELD(helper, L1MuGMTParameters, MergeMethodSRKFwd);
  ADD_FIELD(helper, L1MuGMTParameters, HaloOverwritesMatchedBrl);
  ADD_FIELD(helper, L1MuGMTParameters, HaloOverwritesMatchedFwd);
  ADD_FIELD(helper, L1MuGMTParameters, SortRankOffsetBrl);
  ADD_FIELD(helper, L1MuGMTParameters, SortRankOffsetFwd);
  ADD_FIELD(helper, L1MuGMTParameters, CDLConfigWordDTCSC);
  ADD_FIELD(helper, L1MuGMTParameters, CDLConfigWordCSCDT);
  ADD_FIELD(helper, L1MuGMTParameters, CDLConfigWordbRPCCSC);
  ADD_FIELD(helper, L1MuGMTParameters, CDLConfigWordfRPCDT);
  ADD_FIELD(helper, L1MuGMTParameters, VersionSortRankEtaQLUT);
  ADD_FIELD(helper, L1MuGMTParameters, VersionLUTs);

  boost::shared_ptr<L1MuGMTParameters> ptrResult(new L1MuGMTParameters);

  std::vector<std::string> resultColumns = helper.getColumnList();
  resultColumns.push_back("CMSSW_VERSION");

  l1t::OMDSReader::QueryResults resultLines = 
    m_omdsReader.basicQuery(
          // SELECTed columns
	  resultColumns,
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
    helper.extractRecord(resultRecord, *ptrResult);
    return ptrResult;
  }
     
  throw cond::Exception("Couldn't find GMT_SOFTWARE_CONFIG record for GMT key `" + objectKey + "'") ;
}

void L1MuGMTParametersOnlineProducer::checkCMSSWVersion(const coral::AttributeList& configRecord) 
{
  const coral::Attribute& version = configRecord["CMSSW_VERSION"];

  /* If the DB field is unset, take any. */
  if(version.isNull()) {
    edm::LogInfo("No CMSSW version set in database, accepting " PROJECT_VERSION);
    return;
  }

  /* Else make sure we have the correct  version. */
  const std::string& versionString = version.data<string>();

  /* PROJECT_VERSION is passed as a -D #define from scramv1 (eg CMSSW_2_1_0) */
  if(versionString != PROJECT_VERSION) { 
    std::string errMsg = "CMSSW version mismatch: Configuration requires " + 
      versionString + ", but this is " PROJECT_VERSION "!";
   
    if(ignoreVersionMismatch_) { 
      edm::LogWarning(errMsg + " (will continue because ignoreVersionMismatch is set)");
    } else { 
      throw cond::Exception(errMsg);	    
    }
  }
}

L1MuGMTParametersOnlineProducer::L1MuGMTParametersOnlineProducer(const edm::ParameterSet& ps) : 
  L1ConfigOnlineProdBase<L1MuGMTParametersRcd, L1MuGMTParameters>(ps)
{
  ignoreVersionMismatch_  = ps.getParameter<bool>("ignoreVersionMismatch");
}


L1MuGMTParametersOnlineProducer::~L1MuGMTParametersOnlineProducer() {}

DEFINE_FWK_EVENTSETUP_MODULE(L1MuGMTParametersOnlineProducer);
