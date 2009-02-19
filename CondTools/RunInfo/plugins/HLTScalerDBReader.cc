#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/src/CoralConnectionProxy.h"
#include "CondFormats/RunInfo/interface/LuminosityInfo.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Exception.h"
#include "CoralBase/TimeStamp.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ISchema.h"
#include "HLTScalerDBReader.h"
#include "CondTools/RunInfo/interface/LumiReaderFactory.h"
//#include <iostream>
lumi::HLTScalerDBReader::HLTScalerReader(const edm::ParameterSet&pset):lumi::LumiReaderBase(pset),m_session(new cond::DBSession ){
  m_constr=pset.getParameter<std::string>("connect");
  std::string authPath=pset.getParameter<std::string>("authenticationPath");
  int messageLevel=pset.getUntrackedParameter<int>("messageLevel",0);
  switch (messageLevel) {
  case 0 :
    m_session->configuration().setMessageLevel( cond::Error );
    break;    
  case 1:
    m_session->configuration().setMessageLevel( cond::Warning );
    break;
  case 2:
    m_session->configuration().setMessageLevel( cond::Info );
    break;
  case 3:
    m_session->configuration().setMessageLevel( cond::Debug );
    break;  
  default:
    m_session->configuration().setMessageLevel( cond::Error );
  }
  m_session->configuration().setMessageLevel(cond::Debug);
  m_session->configuration().setAuthenticationMethod(cond::XML);
  m_session->configuration().setAuthenticationPath(authPath);
}
lumi::HLTScalerDBReader::~HLTScalerDBReader(){
  delete m_session;
}
void lumi::HLTScalerDBReader::fill(int startRun,
				int numberOfRuns,
				std::vector< std::pair<lumi::HLTScaler*,cond::Time_t> >& result){
  //fill hlt registry hlt is empty. nothing to do
  //select hlt.INPUT_COUNT, hlt.ACCEPT_COUNT, hlt.PRESCALE_FACTOR from CMS_LUMI.HLTS hlt , CMS_LUMI.LUMI_SECTIONS sect WHERE sect.SECTION_ID=hlt.SECTION_ID AND sect.RUN_NUMBER=runnumber AND sect.LUMI_SECTION_NUMBER=lumisectionid;
}

DEFINE_EDM_PLUGIN(lumi::LumiReaderFactory,lumi::HLTScalerDBReader,"hltdb");
