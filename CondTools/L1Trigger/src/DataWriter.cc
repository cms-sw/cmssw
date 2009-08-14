#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1Trigger/interface/DataWriter.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"

#include <utility>

namespace l1t
{

std::string
DataWriter::writePayload( const edm::EventSetup& setup,
			  const std::string& recordType )
{
  WriterFactory* factory = WriterFactory::get();
  WriterProxy* writer = factory->create( recordType + "@Writer" ) ;
  if( writer == 0 )
    {
      throw cond::Exception( "DataWriter: could not create WriterProxy with name "
			     + recordType + "@Writer" ) ;
    }

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (!poolDb.isAvailable())
    {
      throw cond::Exception( "DataWriter: PoolDBOutputService not available."
			     ) ;
    }
  cond::PoolTransaction& pool = poolDb->connection().poolTransaction() ;
  pool.start( false ) ;

  // update key to have new payload registered for record-type pair.
  std::string payloadToken = writer->save( setup, pool ) ;
  edm::LogVerbatim( "L1-O2O" ) << recordType << " PAYLOAD TOKEN "
			       << payloadToken ;

  delete writer;
  pool.commit ();

  return payloadToken ;
}

void
DataWriter::writeKeyList( L1TriggerKeyList* keyList,
			  edm::RunNumber_t sinceRun,
			  bool logTransactions )
{
  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if( !poolDb.isAvailable() )
    {
      throw cond::Exception( "DataWriter: PoolDBOutputService not available."
			     ) ;
    }

   cond::PoolTransaction& pool = poolDb->connection().poolTransaction() ;
   pool.start( false ) ;

   // Write L1TriggerKeyList payload
   cond::TypedRef< L1TriggerKeyList > ref( pool, keyList ) ;
   //   ref.markWrite( "L1TriggerKeyListRcd" ) ;
   ref.markWrite( ref.className() ) ;

   // Save payload token before committing.
   std::string payloadToken = ref.token() ;

   // Commit before calling updateIOV(), otherwise PoolDBOutputService gets
   // confused.
   pool.commit ();

   // Set L1TriggerKeyList IOV
   updateIOV( "L1TriggerKeyListRcd",
	      payloadToken,
	      sinceRun,
	      logTransactions ) ;
}

bool
DataWriter::updateIOV( const std::string& esRecordName,
		       const std::string& payloadToken,
		       edm::RunNumber_t sinceRun,
		       bool logTransactions )
{
  edm::LogVerbatim( "L1-O2O" ) << esRecordName
			       << " PAYLOAD TOKEN " << payloadToken ;

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (!poolDb.isAvailable())
    {
      throw cond::Exception( "DataWriter: PoolDBOutputService not available."
			     ) ;
    }

  bool iovUpdated = true ;

  if( poolDb->isNewTagRequest( esRecordName ) )
    {
      sinceRun = poolDb->beginOfTime() ;
      poolDb->createNewIOV( payloadToken,
			    sinceRun,
			    poolDb->endOfTime(),
			    esRecordName,
			    logTransactions ) ;
    }
  else
    {	
      cond::TagInfo tagInfo ;
      poolDb->tagInfo( esRecordName, tagInfo ) ;

      if( sinceRun == 0 ) // find last since and add 1
	{
	  sinceRun = tagInfo.lastInterval.first ;
	  ++sinceRun ;
	}

      if( tagInfo.lastPayloadToken != payloadToken )
	{
	  poolDb->appendSinceTime( payloadToken,
				   sinceRun,
				   esRecordName,
				   logTransactions ) ;
	}
      else
	{
	  iovUpdated = false ;
	  edm::LogVerbatim( "L1-O2O" ) << "IOV already up to date." ;
	}
    }

  if( iovUpdated )
    {
      edm::LogVerbatim( "L1-O2O" ) << esRecordName
				   << " SINCE " << sinceRun ;
    }

  return iovUpdated ;
}

std::string
DataWriter::payloadToken( const std::string& recordName,
			  edm::RunNumber_t runNumber )
{
  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if( !poolDb.isAvailable() )
    {
      throw cond::Exception( "DataWriter: PoolDBOutputService not available."
			     ) ;
    }

  // Get tag corresponding to EventSetup record name.
  std::string iovTag = poolDb->tag( recordName ) ;

  // Get IOV token for tag.
  cond::CoralTransaction& coral = poolDb->connection().coralTransaction() ;
  coral.start( false ) ;
  cond::MetaData metadata( coral ) ;
  std::string iovToken ;
  if( metadata.hasTag( iovTag ) )
    {
      iovToken = metadata.getToken( iovTag ) ;
    }
  coral.commit() ;
  if( iovToken.empty() )
    {
      return std::string() ;
    }

  // Get payload token for run number.
  cond::PoolTransaction& pool = poolDb->connection().poolTransaction() ;
  pool.start( false ) ;
  cond::IOVService iovService( pool ) ;
  std::string payloadToken = iovService.payloadToken( iovToken, runNumber ) ;
  pool.commit() ;
  return payloadToken ;
}

} // ns
