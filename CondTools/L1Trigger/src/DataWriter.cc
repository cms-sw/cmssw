#include "CondTools/L1Trigger/interface/DataWriter.h"

#include "CondCore/DBCommon/interface/CoralTransaction.h"


#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"

#include <utility>

namespace l1t
{

/* Data writting functions */
std::string DataWriter::findTokenForTag (const std::string & tag)
{
    // fast check in cashe
    TagToToken::const_iterator found = tagToToken.find (tag);
    if (found != tagToToken.end ())
        return found->second;

    // else slow way, also we may need to create db
//     coral->connect (cond::ReadWriteCreate);
//     coral->startTransaction (false);
    // connection->connect ( session );
  cond::CoralTransaction& coral = connection->coralTransaction() ;
    coral.start (false);

    std::string tagToken;
    if (metadata->hasTag (tag))
        tagToken = metadata->getToken (tag);

    coral.commit ();
//     coral->disconnect ();
    //connection->disconnect ();

    // if not found empty string is returned
    return tagToken;
}

void DataWriter::writeKey (L1TriggerKey * key,
			   const std::string & tag,
			   const edm::RunNumber_t sinceRun)
{
    // writting key as bit more complicated. At this time we have to worry
    // about such things if the key already exists or not
    // Also we need to get IOVToken for given tag if key exist
    // if it does not, then we need to addMapping in the end

    // Bad part - get TagTokent for given tag.
    // We use layzy cash to save all tag adn tokens, in case we save key with same tag
    std::string tagToken = findTokenForTag (tag);
    bool requireMapping = tagToken.empty ();

    //    pool->connect ();
    //connection->connect ( session );
    cond::PoolTransaction& pool = connection->poolTransaction() ;
    pool.start (false);

    cond::TypedRef<L1TriggerKey> ref (pool, key);
    ref.markWrite ("L1TriggerKeyRcd");

    cond::IOVService iov (pool);

    // Create editor, with or wothoug TagToken
    cond::IOVEditor * editor;
    editor = iov.newIOVEditor (tagToken);

    // finally insert new IOV
    cond::TimeType timetype = cond::runnumber; 
    cond::Time_t globalSince = cond::timeTypeSpecs[timetype].beginValue; 

    if( requireMapping )
      {
	editor->create( globalSince, timetype ) ;
      }

    if( sinceRun == globalSince )
      {
	cond::Time_t globalTill = cond::timeTypeSpecs[timetype].endValue; 
	editor->insert (globalTill, ref.token ());
      }
    else
      {
	editor->append(sinceRun, ref.token ());
      }
    tagToken = editor->token ();
    delete editor;

    pool.commit ();
    //    pool->disconnect ();
    //connection->disconnect ();

    if (tagToToken.find (tag) != tagToToken.end ())
        tagToToken.insert (std::make_pair (tag, tagToken));

    // Assign payload token with IOV value
    if (requireMapping)
        addMappings (tag, tagToken);

    std::cout << "L1TriggerKey TOKEN " << tagToken << std::endl ;
}

void DataWriter::addMappings (const std::string tag, const std::string iovToken)
{
  //    coral->connect (cond::ReadWriteCreate);
  //connection->connect( session ) ;
  cond::CoralTransaction& coral = connection->coralTransaction() ;
  //    coral->startTransaction (false);
    coral.start (false);

    metadata->addMapping (tag, iovToken); // default timetype = run number

    coral.commit ();
    //    coral->disconnect ();
    //connection->disconnect ();
}

static std::string buildName( const std::string& iRecordName, const std::string& iTypeName )
{
    return iRecordName+"@"+iTypeName+"@Writer";
}

void DataWriter::writePayload (L1TriggerKey & key, const edm::EventSetup & setup,
        const std::string & record, const std::string & type)
{
    WriterFactory * factory = WriterFactory::get();
    const std::string name = buildName(record, type);
    WriterProxy * writer = factory->create(name);
    if( writer == 0 )
      {
	throw cond::Exception( "DataWriter: could not create WriterProxy with name "
			       + name ) ;
      }
    //assert (writer != 0);

    //    pool->connect ();
    //connection->connect ( session );
    cond::PoolTransaction& pool = connection->poolTransaction() ;
    //    pool->startTransaction (false);
    pool.start (false);

    // update key to have new payload registered for record-type pair.
    std::string payloadToken = writer->save(setup, pool);
    std::cout << "TOKEN " << payloadToken << std::endl ;
    if( payloadToken.empty() )
      {
	throw cond::Exception( "DataWriter: failure to write payload in "
			       + name ) ;
      }
    //assert (!payloadToken.empty ());

    key.add (record, type, payloadToken);

    delete writer;

    pool.commit ();
    // pool->disconnect ();
  //connection->disconnect ();
}

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
    //assert( writer != 0 );

    cond::PoolTransaction& pool = connection->poolTransaction() ;
    pool.start( false ) ;

    // update key to have new payload registered for record-type pair.
    std::string payloadToken = writer->save( setup, pool ) ;
    std::cout << recordType << " PAYLOAD TOKEN " << payloadToken << std::endl ;

    delete writer;
    pool.commit ();

    return payloadToken ;
}

void
DataWriter::writeKeyList( L1TriggerKeyList* keyList,
			  const std::string& tag,
			  const edm::RunNumber_t sinceRun )
{
   // Get IOVToken for given tag
   std::string tagToken = findTokenForTag( tag ) ;
   bool requireMapping = tagToken.empty() ;

    cond::PoolTransaction& pool = connection->poolTransaction() ;
    pool.start( false ) ;

    cond::TypedRef< L1TriggerKeyList > ref( pool, keyList ) ;
    ref.markWrite( "L1TriggerKeyListRcd" ) ;

    cond::IOVService iov( pool ) ;

    // Create editor, with or without tagToken
    cond::IOVEditor* editor ;
    editor = iov.newIOVEditor( tagToken ) ;

    // Insert new IOV
    cond::TimeType timetype = cond::runnumber; 
    cond::Time_t globalSince = cond::timeTypeSpecs[timetype].beginValue; 

    if( requireMapping )
      {
	std::cout << "GLOBAL SINCE " << globalSince
		  << " SINCE " << sinceRun 
		  << std::endl ;

	editor->create( globalSince, timetype ) ;
      }

    if( sinceRun == globalSince )
      {
	cond::Time_t globalTill = cond::timeTypeSpecs[timetype].endValue; 
	editor->insert (globalTill, ref.token ());
      }
    else
      {
	editor->append( sinceRun, ref.token() ) ;
      }
    tagToken = editor->token() ;
    delete editor ;

    // Is this necessary?
    pool.commit() ;

    if( tagToToken.find( tag ) != tagToToken.end () )
    {
       tagToToken.insert( std::make_pair( tag, tagToken ) ) ;
    }

    // Assign payload token with IOV value
    if( requireMapping )
    {
       addMappings( tag, tagToken ) ;
    }

    std::cout << "L1TriggerKeyList IOV TOKEN " << tagToken
	      << " TAG " << tag << std::endl ;
}

void
DataWriter::updateIOV( const std::string& tag,
		       const std::string& payloadToken,
		       const edm::RunNumber_t sinceRun )
{
    std::cout << tag << " PAYLOAD TOKEN " << payloadToken << std::endl ;

   // Get IOVToken for given tag
   std::string tagToken = findTokenForTag( tag ) ;
   bool requireMapping = tagToken.empty() ;

    cond::PoolTransaction& pool = connection->poolTransaction() ;
    pool.start( false ) ;

    cond::IOVService iov( pool ) ;

    // Create editor, with or without tagToken
    cond::IOVEditor* editor ;
    editor = iov.newIOVEditor( tagToken ) ;

    // Insert new IOV
    cond::TimeType timetype = cond::runnumber; 
    cond::Time_t globalSince = cond::timeTypeSpecs[timetype].beginValue; 

    if( requireMapping )
      {
	// insert() sets till-time, not since-time -- will this work?
	editor->create( globalSince, timetype ) ;
      }

    if( sinceRun == globalSince )
      {
	cond::Time_t globalTill = cond::timeTypeSpecs[timetype].endValue; 
	editor->insert (globalTill, payloadToken );
      }
    else
      {
	editor->append( sinceRun, payloadToken ) ;
      }
    tagToken = editor->token() ;
    delete editor ;

    // Is this necessary?
    pool.commit() ;

    if( tagToToken.find( tag ) != tagToToken.end () )
    {
       tagToToken.insert( std::make_pair( tag, tagToken ) ) ;
    }

    // Assign payload token with IOV value
    if( requireMapping )
    {
       addMappings( tag, tagToken ) ;
    }

    std::cout << tag << " IOV TOKEN " << tagToken << std::endl ;
}


} // ns
