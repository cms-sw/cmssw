// $Id: ConsumerUtils.cc,v 1.7 2009/12/01 13:58:08 mommsen Exp $
/// @file: ConsumerUtils.cc

#include "EventFilter/StorageManager/interface/ConsumerID.h"
#include "EventFilter/StorageManager/interface/ConsumerUtils.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"

#include "IOPool/Streamer/interface/ConsRegMessage.h"
#include "IOPool/Streamer/interface/OtherMessage.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "xgi/Input.h"
#include "xgi/Output.h"
#include "xgi/exception/Exception.h"

#include <string>
#include <vector>
#include <memory>

using namespace stor;

////////////////////////////////////////////
//// Create consumer registration info: ////
////////////////////////////////////////////
ConsRegPtr stor::parseEventConsumerRegistration( xgi::Input* in,
                                                 int queueSize,
                                                 enquing_policy::PolicyTag queuePolicy,
                                                 utils::duration_t secondsToStale )
{

  if( in == 0 )
    {
      XCEPT_RAISE( xgi::exception::Exception,
                   "Null xgi::Input* in parseEventConsumerRegistration" );
    }

  std::string name = "unknown";
  std::string pset_str = "<>";

  const std::string l_str = in->getenv( "CONTENT_LENGTH" );
  unsigned long l = std::atol( l_str.c_str() );

  const std::string remote_host = in->getenv( "REMOTE_HOST" );

  if( l > 0 )
    {
      std::auto_ptr< std::vector<char> > buf( new std::vector<char>(l) );
      in->read( &(*buf)[0], l );
      ConsRegRequestView req( &(*buf)[0] );
      name = req.getConsumerName();
      pset_str = req.getRequestParameterSet();
    }
  else
    {
      XCEPT_RAISE( stor::exception::ConsumerRegistration,
                   "Bad request length" );
    }

  const edm::ParameterSet pset( pset_str );

  //
  //// Check if HLT output module is there: ////
  //

  std::string sel_hlt_out = "";

  try
    {
      // new-style consumer
      sel_hlt_out = pset.getParameter<std::string>( "TrackedHLTOutMod" );
    }
  catch( edm::Exception& e )
    {
      // old-style consumer or param not specified
      sel_hlt_out =
        pset.getUntrackedParameter<std::string>( "SelectHLTOutput", "" );
    }

  if( sel_hlt_out == "" )
    {
      XCEPT_RAISE( stor::exception::ConsumerRegistration,
                   "No HLT output module specified" );
    }

  // Event filters:
  std::string sel_events_new = std::string();
  try
    {
      sel_events_new = pset.getParameter<std::string>("TriggerSelector");
    }
  catch (edm::Exception& e) {}

  EventConsumerRegistrationInfo::FilterList sel_events;
  try
    {
      sel_events = pset.getParameter<Strings>( "TrackedEventSelection" );
    }
  catch( edm::Exception& e )
    {
      edm::ParameterSet tmpPSet1 =
        pset.getUntrackedParameter<edm::ParameterSet>( "SelectEvents",
                                                       edm::ParameterSet() );
      if ( ! tmpPSet1.empty() )
        {
          sel_events = tmpPSet1.getParameter<Strings>( "SelectEvents" );
        }
    }

  // Number of retries:
  unsigned int max_conn_retr = 5;
  try
    {
      max_conn_retr = pset.getParameter<int>( "TrackedMaxConnectTries" );
    }
  catch( edm::Exception& e )
    {
      pset.getUntrackedParameter<int>( "maxConnectTries", 5 );
    }

  // Retry interval:
  unsigned int conn_retr_interval = 10;
  try
    {
      conn_retr_interval =
        pset.getParameter<int>( "TrackedConnectTrySleepTime" );
    }
  catch( edm::Exception& e )
    {
      conn_retr_interval =
        pset.getUntrackedParameter<int>( "connectTrySleepTime", 10 );
    }

  ConsRegPtr cr( new EventConsumerRegistrationInfo( max_conn_retr,
                                                    conn_retr_interval,
                                                    name,
                                                    sel_events_new,
                                                    sel_events,
                                                    sel_hlt_out,
                                                    queueSize,
                                                    queuePolicy,
                                                    secondsToStale,
                                                    remote_host ) );
  return cr;

}

////////////////////////////////////////////////
//// Create DQM consumer registration info: ////
////////////////////////////////////////////////
DQMEventConsRegPtr stor::parseDQMEventConsumerRegistration( xgi::Input* in,
                                                            int queueSize,
                                                            enquing_policy::PolicyTag queuePolicy,
                                                            utils::duration_t secondsToStale )
{

  if( in == 0 )
    {
      XCEPT_RAISE( xgi::exception::Exception,
                   "Null xgi::Input* in parseDQMEventConsumerRegistration" );
    }
  
  std::string consumerName = "None provided";
  std::string consumerTopFolderName = "*";

  // read the consumer registration message from the http input stream
  std::string lengthString = in->getenv("CONTENT_LENGTH");
  unsigned int contentLength = std::atol(lengthString.c_str());
  if (contentLength > 0)
  {
    std::auto_ptr< std::vector<char> > bufPtr(new std::vector<char>(contentLength));
    in->read(&(*bufPtr)[0], contentLength);
    ConsRegRequestView requestMessage(&(*bufPtr)[0]);
    consumerName = requestMessage.getConsumerName();
    // for DQM consumers top folder name is stored in the "parameteSet"
    std::string reqFolder = requestMessage.getRequestParameterSet();
    if (reqFolder.size() >= 1) consumerTopFolderName = reqFolder;
  }

  const std::string remote_host = in->getenv( "REMOTE_HOST" );

  DQMEventConsRegPtr cr( new DQMEventConsumerRegistrationInfo( consumerName,
                                                               consumerTopFolderName, 
                                                               queueSize,
                                                               queuePolicy,
                                                               secondsToStale,
                                                               remote_host ) );
  return cr;

}

/////////////////////////////
//// Write HTTP headers: ////
/////////////////////////////
void stor::writeHTTPHeaders( xgi::Output* out )
{
  out->getHTTPResponseHeader().addHeader( "Content-Type",
                                          "application/octet-stream" );
  out->getHTTPResponseHeader().addHeader( "Content-Transfer-Encoding",
                                          "binary" );
}

//////////////////////////////
//// Send ID to consumer: ////
//////////////////////////////
void stor::writeConsumerRegistration( xgi::Output* out, ConsumerID cid )
{

  const int buff_size = 1000;
  std::vector<unsigned char> buff( buff_size );

  ConsRegResponseBuilder rb( &buff[0], buff.capacity(), 0, cid.value );
  ConsRegResponseView rv( &buff[0] );
  const unsigned int len = rv.size();

  writeHTTPHeaders( out );
  out->write( (char*)(&buff[0]), len );

}

////////////////////////////////////////
//// Tell consumer we're not ready: ////
////////////////////////////////////////
void stor::writeNotReady( xgi::Output* out )
{

  const int buff_size = 1000;
  std::vector<unsigned char> buff( buff_size );

  ConsRegResponseBuilder rb( &buff[0], buff.capacity(),
                             ConsRegResponseBuilder::ES_NOT_READY, 0 );
  ConsRegResponseView rv( &buff[0] );
  const unsigned int len = rv.size();

  writeHTTPHeaders( out );
  out->write( (char*)(&buff[0]), len );

}

////////////////////////////////////////
//// Send empty buffer to consumer: ////
////////////////////////////////////////
void stor::writeEmptyBuffer( xgi::Output* out )
{
  char buff;
  writeHTTPHeaders( out );
  out->write( &buff, 0 );
}

///////////////////////////////////////////
//// Tell consumer that run has ended: ////
///////////////////////////////////////////
void stor::writeDone( xgi::Output* out )
{

  const int buff_size = 1000;
  std::vector<unsigned char> buff( buff_size );

  OtherMessageBuilder omb( &buff[0], Header::DONE );
  const unsigned int len = omb.size();

  writeHTTPHeaders( out );
  out->write( (char*)(&buff[0]), len );

}

/////////////////////////////////////////
//// Send error message to consumer: ////
/////////////////////////////////////////
void stor::writeErrorString( xgi::Output* out, std::string errorString )
{

  const int buff_size = errorString.size();
  std::vector<unsigned char> buff( buff_size );

  const char *errorBytes = errorString.c_str();
  for (int i=0; i<buff_size; ++i) buff[i]=errorBytes[i];

  writeHTTPHeaders( out );
  out->write( (char*)(&buff[0]), buff_size );

}


//////////////////////////////////////////////////
//// Extract consumer ID from header request: ////
//////////////////////////////////////////////////
ConsumerID stor::getConsumerID( xgi::Input* in )
{

  if( in == 0 )
    {          
      XCEPT_RAISE( xgi::exception::Exception,
                   "Null xgi::Input* in getConsumerID" );
    }

  const std::string l_str = in->getenv( "CONTENT_LENGTH" );
  unsigned long l = std::atol( l_str.c_str() );

  unsigned int cid_int = 0;

  if( l > 0 )
    {
      std::auto_ptr< std::vector<char> > buf( new std::vector<char>(l) );
      in->read( &(*buf)[0], l );
      OtherMessageView req( &(*buf)[0] );
      if( req.code() == Header::HEADER_REQUEST ||
          req.code() == Header::EVENT_REQUEST ||
          req.code() == Header::DQMEVENT_REQUEST )
        {
          uint8* ptr = req.msgBody();
          cid_int = convert32( ptr );
        }
      else
        {
          XCEPT_RAISE( stor::exception::Exception,
                       "Bad request code in getConsumerID" );
        }
    }
  else
    {
      XCEPT_RAISE( stor::exception::Exception,
                   "Bad request length in getConsumerID" );
    }

  return ConsumerID( cid_int );

}

///////////////////////
//// Write header: ////
///////////////////////
void stor::writeConsumerHeader( xgi::Output* out, InitMsgSharedPtr ptr )
{
  const unsigned int len = ptr->size();
  std::vector<unsigned char> buff( len );
  for( unsigned int i = 0; i < len; ++i )
    {
      buff[i] = (*ptr)[i];
    }
  writeHTTPHeaders( out );
  out->write( (char*)(&buff[0]), len );
}

//////////////////////
//// Write event: ////
//////////////////////
void stor::writeConsumerEvent( xgi::Output* out, const I2OChain& evt )
{

  writeHTTPHeaders( out );

  const unsigned int nfrags = evt.fragmentCount();
  for ( unsigned int i = 0; i < nfrags; ++i )
   {
     const unsigned int len = evt.dataSize( i );
     unsigned char* location = evt.dataLocation( i );
     out->write( (char*)location, len );
   } 

}

//////////////////////////
//// Write DQM event: ////
//////////////////////////
void stor::writeDQMConsumerEvent( xgi::Output* out, const DQMEventMsgView& view )
{

  writeHTTPHeaders( out );

  const unsigned int len = view.size();
  unsigned char* location = view.startAddress();
  out->write( (char*)location, len );

}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
