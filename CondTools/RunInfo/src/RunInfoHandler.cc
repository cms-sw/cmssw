#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/RunInfo/interface/RunInfoHandler.h"
#include "CondTools/RunInfo/interface/RunInfoRead.h"
#include<iostream>
#include<vector>

RunInfoHandler::RunInfoHandler(const edm::ParameterSet& pset) :
   m_name( pset.getUntrackedParameter<std::string>( "name", "RunInfoHandler") )
  ,m_since( pset.getParameter<unsigned long long>( "runNumber" ) )
  ,m_connectionString( pset.getUntrackedParameter<std::string>( "connectionString", "oracle://cms_omds_adg/CMS_RUNINFO") )
  ,m_authpath( pset.getUntrackedParameter<std::string>( "authenticationPath", "." ) )
  ,m_user( pset.getUntrackedParameter<std::string>( "OnlineDBUser", "CMS_RUNINFO_R" ) )
  ,m_pass( pset.getUntrackedParameter<std::string>( "OnlineDBPass", "PASSWORD") ) {
}

RunInfoHandler::~RunInfoHandler() {}

void RunInfoHandler::getNewObjects() {
  //check whats already inside of database
  edm::LogInfo( "RunInfoHandler" ) << "------- " << m_name
                                   << " - > getNewObjects\n"
                                   << "got offlineInfo " << tagInfo().name
                                   << ", size " << tagInfo().size
                                   << ", last object valid since " << tagInfo().lastInterval.first
                                   << " token " << tagInfo().lastPayloadToken << std::endl;
  edm::LogInfo( "RunInfoHandler" ) << "runnumber/first since = " << m_since << std::endl;
  RunInfo* r = new RunInfo();
  
  //fill with null runinfo if empty run are found beetween the two last valid ones 
  size_t n_empty_run = 0;
  if( tagInfo().size > 0  && (tagInfo().lastInterval.first + 1) < m_since ) {
    n_empty_run = m_since - tagInfo().lastInterval.first - 1;
    edm::LogInfo( "RunInfoHandler" ) << "------- " << "entering fake run from "
                                     << tagInfo().lastInterval.first + 1
                                     <<  "to " << m_since - 1 << "- > getNewObjects"
                                     << std::endl;
  } 
  // transfer fake run for 1 to since for the first time
  if( tagInfo().size == 0 && m_since != 1 ) {
    m_to_transfer.push_back( std::make_pair( (RunInfo*)(r->Fake_RunInfo()), 1 ) );
  }
  if ( n_empty_run != 0 ) {
    m_to_transfer.push_back(std::make_pair( (RunInfo*)(r->Fake_RunInfo()), tagInfo().lastInterval.first + 1 ) );
  }
  
  //reading from omds
  RunInfoRead rn( m_connectionString, m_user, m_pass );
  *r = rn.readData( "RUNSESSION_PARAMETER", "STRING_VALUE",(int)m_since );
  m_to_transfer.push_back( std::make_pair( (RunInfo*)r, m_since) );
  std::ostringstream ss;
  ss << "since =" << m_since;
  m_userTextLog = ss.str() + ";";
  edm::LogInfo( "RunInfoHandler" ) << "------- " << m_name << " - > getNewObjects" << std::endl;
}
