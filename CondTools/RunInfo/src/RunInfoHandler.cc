#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondTools/RunInfo/interface/RunInfoHandler.h"
#include "CondTools/RunInfo/interface/RunInfoRead.h"
#include<iostream>
#include<vector>

RunInfoHandler::RunInfoHandler( const edm::ParameterSet& pset ) :
   m_since( pset.getParameter<unsigned long long>( "runNumber" ) )
  ,m_name( pset.getUntrackedParameter<std::string>( "name", "RunInfoHandler" ) )
  ,m_runinfo_schema( pset.getUntrackedParameter<std::string>( "RunInfoSchema", "CMS_RUNINFO" ) )
  ,m_dcsenv_schema( pset.getUntrackedParameter<std::string>( "DCSEnvSchema", "CMS_DCS_ENV_PVSS_COND") )
  ,m_connectionString( pset.getParameter<std::string>( "connect" ) )
  ,m_connectionPset( pset.getParameter<edm::ParameterSet>( "DBParameters" ) ) {
}

RunInfoHandler::~RunInfoHandler() {}

void RunInfoHandler::getNewObjects() {
  //check whats already inside of database
  edm::LogInfo( "RunInfoHandler" ) << "[" << "RunInfoHandler::" << __func__ << "]:" << m_name << ": "
                                   << "Destination Tag Info: name " << tagInfo().name
                                   << ", size " << tagInfo().size
                                   << ", last object valid since " << tagInfo().lastInterval.first
                                   << ", hash " << tagInfo().lastPayloadToken << std::endl;
  edm::LogInfo( "RunInfoHandler" ) << "[" << "RunInfoHandler::" << __func__ << "]:" << m_name << ": runnumber/first since = " << m_since << std::endl;

  //check if a transfer is needed:
  //if the new run number is smaller than or equal to the latest IOV, exit.
  //This is needed as now the IOV Editor does not always protect for insertions:
  //ANY and VALIDATION sychronizations are allowed to write in the past.
  if( tagInfo().size > 0  && tagInfo().lastInterval.first >= m_since ) {
    edm::LogWarning( "RunInfoHandler" ) << "[" << "RunInfoHandler::" << __func__ << "]:" << m_name << ": "
                                        << "last IOV " << tagInfo().lastInterval.first
                                        << ( tagInfo().lastInterval.first == m_since ? " is equal to" : " is larger than" )
                                        << " the run proposed for insertion " << m_since
                                        << ". No transfer needed." << std::endl;
    return;
  }

  RunInfo* r = new RunInfo();
  
  //fill with null runinfo if empty run are found beetween the two last valid ones 
  size_t n_empty_run = 0;
  if( tagInfo().size > 0  && (tagInfo().lastInterval.first + 1) < m_since ) {
    n_empty_run = m_since - tagInfo().lastInterval.first - 1;
    edm::LogInfo( "RunInfoHandler" ) << "[" << "RunInfoHandler::" << __func__ << "]:" << m_name << ": "
                                     << "entering fake run from "
                                     << tagInfo().lastInterval.first + 1
                                     <<  " to " << m_since - 1
                                     << std::endl;
  } 
  std::ostringstream ss;
  // transfer fake run for 1 to since for the first time
  if( tagInfo().size == 0 && m_since != 1 ) {
    m_to_transfer.push_back( std::make_pair( (RunInfo*)(r->Fake_RunInfo()), 1 ) );
    ss << "fake run number: " << 1 << ", ";
  }
  if ( n_empty_run != 0 ) {
    m_to_transfer.push_back(std::make_pair( (RunInfo*)(r->Fake_RunInfo()), tagInfo().lastInterval.first + 1 ) );
    ss << "fake run number: " << tagInfo().lastInterval.first + 1 << ", ";
  }
  
  //reading from omds
  RunInfoRead rn( m_connectionString, m_connectionPset );
  *r = rn.readData( m_runinfo_schema, m_dcsenv_schema, (int)m_since );
  m_to_transfer.push_back( std::make_pair( (RunInfo*)r, m_since) );
  ss << "run number: " << m_since << ";";
  m_userTextLog = ss.str();
  edm::LogInfo( "RunInfoHandler" ) << "[" << "RunInfoHandler::" << __func__ << "]:" << m_name << ": END." << std::endl;
}
