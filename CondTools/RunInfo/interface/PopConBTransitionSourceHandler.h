#ifndef PopConBTransitionSourceHandler_H
#define PopConBTransitionSourceHandler_H

#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/IOVProxy.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

namespace popcon {
  template <class T>
  class PopConBTransitionSourceHandler: public PopConSourceHandler<T> {
  public:
    PopConBTransitionSourceHandler( edm::ParameterSet const & pset ):
      m_run( pset.getParameter<edm::ParameterSet>( "BTransition" ).getParameter<unsigned long long>( "runNumber" ) ),
      m_currentThreshold( pset.getParameter<edm::ParameterSet>( "BTransition" ).getUntrackedParameter<double>( "currentThreshold", 18000. ) ),
      m_tagForRunInfo( pset.getParameter<edm::ParameterSet>( "BTransition" ).getParameter<std::string>( "tagForRunInfo" ) ),
      m_tagForBOff( pset.getParameter<edm::ParameterSet>( "BTransition" ).getParameter<std::string>( "tagForBOff" ) ),
      m_tagForBOn( pset.getParameter<edm::ParameterSet>( "BTransition" ).getParameter<std::string>( "tagForBOn" ) ),
      m_connectionString( pset.getParameter<edm::ParameterSet>( "BTransition" ).getParameter<std::string>( "connect" ) ),
      m_connectionPset( pset.getParameter<edm::ParameterSet>( "BTransition" ).getParameter<edm::ParameterSet>( "DBParameters" ) ) {
      edm::LogInfo( "PopConBTransitionSourceHandler" ) << "[" << "PopConBTransitionSourceHandler:" << __func__ << "]: "
                                                       << "Initialising Connection Pool" << std::endl;
      m_connection.setParameters( m_connectionPset );
      m_connection.configure();
  }

    virtual ~PopConBTransitionSourceHandler() {}

    virtual std::string id() const override final { return std::string( "PopConBTransitionSourceHandler" ); }

    bool checkBOn() {
      //the output boolean is set to true as default
      bool isBOn = true;
      //reading RunInfo from Conditions
      edm::LogInfo( "PopConBTransitionSourceHandler" ) << "[" << "PopConBTransitionSourceHandler::" << __func__ << "]: "
                                                       << "Initialising CondDB read-only session to " << m_connectionString << std::endl;
      cond::persistency::Session session = m_connection.createReadOnlySession( m_connectionString, "" );
      session.transaction().start();
      edm::LogInfo( "PopConBTransitionSourceHandler" ) << "[" << "PopConBTransitionSourceHandler::" << __func__ << "]: "
                                                       << "Loading tag for RunInfo " << m_tagForRunInfo
                                                       << " and IOV valid for run number: " << m_run << std::endl;
      cond::persistency::IOVProxy iov = session.readIov( m_tagForRunInfo );
      cond::Iov_t currentIov = *(iov.find( m_run ));
      LogDebug( "PopConBTransitionSourceHandler" ) << "Loaded IOV sequence from tag " << m_tagForRunInfo
                                                   << " with size: "<< iov.loadedSize()
                                                   << ", IOV valid for run number " << m_run << " starting from: " << currentIov.since
                                                   << ", with corresponding payload hash: " << currentIov.payloadId
                                                   << std::endl;
      //accessing the average magnet current for the run
      double current_default = -1;
      double avg_current = current_default;
      avg_current = session.fetchPayload<RunInfo>( currentIov.payloadId )->m_avg_current;
      LogDebug( "PopConBTransitionSourceHandler" ) << "Comparing value of magnet current: " << avg_current << " A for run: " << m_run
                                                   << " with the corresponding threshold: "<< m_currentThreshold << " A." << std::endl;
      //comparing the magnet current with the user defined threshold
      if( avg_current != current_default && avg_current <= m_currentThreshold ) isBOn = false;
      edm::LogInfo( "PopConBTransitionSourceHandler" ) << "[" << "PopConBTransitionSourceHandler::" << __func__ << "]: "
                                                       << "The magnet was " << ( isBOn ? "ON" : "OFF" )
                                                       << " during run " << m_run << std::endl;
      session.transaction().commit();
      return isBOn;
    }

    virtual void getObjectsForBTransition( bool isBOn ) {
      //reading payloads for 0T and 3.8T from Conditions
      edm::LogInfo( "PopConBTransitionSourceHandler" ) << "[" << "PopConBTransitionSourceHandler::" << __func__ << "]: "
                                                       << "Initialising CondDB read-only session to " << m_connectionString << std::endl;
      cond::persistency::Session session = m_connection.createReadOnlySession( m_connectionString, "" );
      session.transaction().start();
      edm::LogInfo( "PopConBTransitionSourceHandler" ) << "[" << "PopConBTransitionSourceHandler::" << __func__ << "]: "
                                                       << "Loading tag for B " << ( isBOn ? "ON" : "OFF" ) << ": "
                                                       << ( isBOn ? m_tagForBOn : m_tagForBOff )
                                                       << " and IOV valid for run number: " << m_run << std::endl;
      cond::persistency::IOVProxy iov = session.readIov( isBOn ? m_tagForBOn : m_tagForBOff, true );
      cond::Iov_t currentIov = *(iov.find( m_run ));
      LogDebug( "PopConBTransitionSourceHandler" ) << "Loaded IOV sequence from tag " << ( isBOn ? m_tagForBOn : m_tagForBOff )
                                                   << " with size: "<< iov.loadedSize()
                                                   << ", IOV valid for run number " << m_run << " starting from: " << currentIov.since
                                                   << ", with corresponding payload hash: " << currentIov.payloadId
                                                   << std::endl;
      if( currentIov.payloadId != this->tagInfo().lastPayloadToken ) {
        std::ostringstream ss;
        ss << "Exporting payload with hash " << currentIov.payloadId
           << " corresponding to the calibrations for magnetic field "
           << ( isBOn ? "ON" : "OFF" )
           << " starting from run number: " << m_run;
        edm::LogInfo( "PopConBTransitionSourceHandler" ) << "[" << "PopConBTransitionSourceHandler::" << __func__ << "]: "
                                                         << ss.str() << std::endl;
        T* pp = new T( *(session.fetchPayload<T>( currentIov.payloadId )) );
        this->m_to_transfer.push_back( std::make_pair( pp, m_run ) );
        this->m_userTextLog = ss.str();
      } else {
        edm::LogInfo( "PopConBTransitionSourceHandler" ) << "[" << "PopConBTransitionSourceHandler::" << __func__ << "]: "
                                                         << "The payload with hash " << currentIov.payloadId
                                                         << " corresponding to the calibrations for magnetic field "
                                                         << ( isBOn ? "ON" : "OFF" )
                                                         << " is still valid for run " << m_run
                                                         << " in the destination tag " << this->tagInfo().name
                                                         << ".\nNo transfer needed." <<std::endl;
      }
      session.transaction().commit();
    }

    virtual void getNewObjects() override final {
      //check whats already inside of database
       edm::LogInfo( "PopConBTransitionSourceHandler" ) << "[" << "PopConBTransitionSourceHandler::" << __func__ << "]: "
                                                        << "Destination Tag Info: name " << this->tagInfo().name
                                                        << ", size " << this->tagInfo().size
                                                        << ", last object valid since " << this->tagInfo().lastInterval.first
                                                        << ", hash " << this->tagInfo().lastPayloadToken << std::endl;
      //check if a transfer is needed:
      //if the new run number is smaller than or equal to the latest IOV, exit.
      //This is needed as now the IOV Editor does not always protect for insertions:
      //ANY and VALIDATION sychronizations are allowed to write in the past.
      if( this->tagInfo().size > 0  && this->tagInfo().lastInterval.first >= m_run ) {
        edm::LogInfo( "PopConBTransitionSourceHandler" ) << "[" << "PopConBTransitionSourceHandler::" << __func__ << "]: "
                                                         << "last IOV " << this->tagInfo().lastInterval.first
                                                         << ( this->tagInfo().lastInterval.first == m_run ? " is equal to" : " is larger than" )
                                                         << " the run proposed for insertion " << m_run
                                                         << ". No transfer needed." << std::endl;
        return;
      }
      bool isBOn = checkBOn();
      getObjectsForBTransition( isBOn );
      edm::LogInfo( "PopConBTransitionSourceHandler" ) << "[" << "PopConBTransitionSourceHandler::" << __func__ << "]: "
                                                       << "END." << std::endl;
    }

  private:
    unsigned long long m_run;
    double m_currentThreshold;
    // for reading from CondDB the current from RunInfo
    std::string m_tagForRunInfo;
    // for reading from CondDB the 0T and 3.8T payloads
    std::string m_tagForBOff;
    std::string m_tagForBOn;
    std::string m_connectionString;
    edm::ParameterSet m_connectionPset;
    cond::persistency::ConnectionPool m_connection;
  };
} //namespace popcon

#endif //PopConBTransitionSourceHandler_H
