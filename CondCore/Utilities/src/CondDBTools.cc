#include "CondCore/Utilities/interface/CondDBTools.h"
#include "CondCore/Utilities/interface/CondDBImport.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
//
#include "CondCore/CondDB/src/DbCore.h"
//
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/bind.hpp>

namespace cond {

  namespace persistency {

    size_t copyTag( const std::string& sourceTag, 
		    Session& sourceSession, 
		    const std::string& destTag, 
		    Session& destSession,
		    UpdatePolicy policy,
		    bool log ){
      persistency::TransactionScope ssc( sourceSession.transaction() );
      ssc.start();
      if( log ) std::cout <<"    Loading source iov..."<<std::endl;
      persistency::IOVProxy p = sourceSession.readIov( sourceTag, true );
      if( p.loadedSize()==0 ) {
	if( log ) std::cout <<"    Tag contains 0 iovs."<<std::endl; 
	return 0;
      }
      if( log ) std::cout <<"    Copying tag. Iov size:"<<p.loadedSize()<<" timeType:"<<p.timeType()<<" payloadObjectType=\""<<p.payloadObjectType()<<"\""<<std::endl;

      persistency::IOVEditor editor;
      persistency::TransactionScope dsc( destSession.transaction() );
      dsc.start( false );
      bool exists = false;
      if( !destSession.existsDatabase() ) {
	destSession.createDatabase();
      } else {
	exists = destSession.existsIov( destTag );
      }
      if( exists ){
	if( policy == REPLACE ){
	  destSession.clearIov( destTag );
	} else if( policy == NEW ){
	  destSession.transaction().rollback();
	  throwException("    Tag \""+destTag+"\" already exists.","copyTag");
	}
	editor = destSession.editIov( destTag );
      } else {
	editor = destSession.createIov( p.payloadObjectType(), destTag, p.timeType(), p.synchronizationType() );
      }
      editor.setDescription("Tag "+sourceTag+" migrated from "+sourceSession.connectionString());

      size_t niovs = 0;
      std::set<cond::Hash> pids;
      std::set<cond::Time_t> sinces;
      for(  auto iov : p ){
	// skip duplicated sinces
	if( sinces.find( iov.since ) != sinces.end() ){
	  if( log  ) std::cout <<"    WARNING. Skipping duplicated since="<<iov.since<<std::endl;
	  continue;
	}
	sinces.insert( iov.since );
	// make sure that we import the payload _IN_USE_
	auto usedIov = p.getInterval( iov.since );
	std::pair<std::string,boost::shared_ptr<void> > readBackPayload = fetch( usedIov.payloadId, sourceSession );
	cond::Hash ph = import( sourceSession, usedIov.payloadId, readBackPayload.first, readBackPayload.second.get(), destSession );
	editor.insert( iov.since, ph );
	pids.insert( ph );
	niovs++;
	if( log && niovs && (niovs%1000==0) ) std::cout <<"    Total of iov inserted: "<<niovs<<" payloads: "<<pids.size()<<std::endl;
      } 
      if( log ) std::cout <<"    Total of iov inserted: "<<niovs<<" payloads: "<<pids.size()<<std::endl;
      if( log ) std::cout <<"    Flushing changes..."<<std::endl;
      editor.flush();
      dsc.commit();
      ssc.commit();
      return niovs;
    }

    // comparison functor for iov tuples: Time_t only and Time_t,string
    struct IOVComp {
      bool operator()( const cond::Time_t& x, const std::pair<cond::Time_t,boost::posix_time::ptime>& y ){ return ( x < y.first ); }
    };
    
    size_t importIovs( const std::string& sourceTag, 
		       Session& sourceSession, 
		       const std::string& destTag, 
		       Session& destSession, 
		       cond::Time_t begin,
		       cond::Time_t end,
		       const std::string& description,
		       bool log ){
      persistency::TransactionScope ssc( sourceSession.transaction() );
      ssc.start();
      if( log ) std::cout <<"    Loading source iov..."<<std::endl;
      persistency::IOVProxy p = sourceSession.readIov( sourceTag, true );
      if( p.loadedSize()==0 ) {
	if( log ) std::cout <<"    Tag contains 0 iovs."<<std::endl; 
	return 0;
      } else {
	if( log ) std::cout <<"    Iov size:"<<p.loadedSize()<<" timeType:"<<p.timeType()<<" payloadObjectType=\""<<p.payloadObjectType()<<"\""<<std::endl;
      }
      if( (*p.begin()).since > begin ) begin = (*p.begin()).since;
      if( end < begin ) {
	if( log ) std::cout <<"    No Iov in the selected range."<<std::endl; 
	return 0;
      }
      persistency::IOVEditor editor;
      persistency::TransactionScope dsc( destSession.transaction() );
      dsc.start( false );
      bool exists = false;
      if( !destSession.existsDatabase() ) {
	destSession.createDatabase();
      } else {
	exists = destSession.existsIov( destTag );
      }
      if( exists ){
	editor = destSession.editIov( destTag );
	if( editor.timeType() != p.timeType() )
	  throwException( "TimeType of the destination tag does not match with the source tag timeType.", "importIovs"); 
	  if( editor.payloadType() != p.payloadObjectType() )
	  throwException( "PayloadType of the destination tag does not match with the source tag payloadType.", "importIovs");
      } else {
	editor = destSession.createIov( p.payloadObjectType(), destTag, p.timeType(), p.synchronizationType() );
	if( description.empty() ) editor.setDescription( "Created copying tag "+sourceTag+" from "+sourceSession.connectionString() );
	else editor.setDescription( description );
      }
      size_t niovs = 0;
      std::set<cond::Hash> pids;
      std::set<cond::Time_t> sinces;
      auto iiov = p.find( begin );
      cond::Time_t newSince = begin;
      while( iiov != p.end() ){	
	// skip duplicated sinces
	if( sinces.find( newSince ) != sinces.end() ){
	  if( log ) std::cout <<"    WARNING. Skipping duplicated since="<<newSince<<std::endl;
	  continue;
	}
	sinces.insert( newSince );
	// make sure that we import the payload _IN_USE_
	auto usedIov = p.getInterval( newSince );
	std::pair<std::string,boost::shared_ptr<void> > readBackPayload = fetch( usedIov.payloadId, sourceSession );
	cond::Hash ph = import( sourceSession, usedIov.payloadId, readBackPayload.first, readBackPayload.second.get(), destSession );
	editor.insert( newSince, ph );
	pids.insert( ph );
	niovs++;
	if( log && niovs && (niovs%1000==0) ) std::cout <<"    Total of iov inserted: "<<niovs<<" payloads: "<<pids.size()<<std::endl;
	iiov++;
	if( iiov == p.end() || (*iiov).since > end ){
	  break;
	} else {
	  newSince = (*iiov).since;
	}
      } 
      if( log ) std::cout <<"    Total of iov inserted: "<<niovs<<" payloads: "<<pids.size()<<std::endl;
      if( log ) std::cout <<"    Flushing changes..."<<std::endl;
      editor.flush();
      dsc.commit();
      ssc.commit();
      return niovs;
    }

    bool copyIov( Session& session,
		  const std::string& sourceTag,
		  const std::string& destTag,
		  cond::Time_t sourceSince,
		  cond::Time_t destSince,
		  const std::string& description,
		  bool log ){
      persistency::TransactionScope ssc( session.transaction() );
      ssc.start( false );
      if( log ) std::cout <<"    Loading source iov..."<<std::endl;
      persistency::IOVProxy p = session.readIov( sourceTag, true );
      if( p.loadedSize()==0 ) {
	if( log ) std::cout <<"    Tag contains 0 iovs."<<std::endl; 
	return false;
      } else {
	if( log ) std::cout <<"    Iov size:"<<p.loadedSize()<<" timeType:"<<p.timeType()<<" payloadObjectType=\""<<p.payloadObjectType()<<"\""<<std::endl;
      }

      auto iiov = p.find( sourceSince );
      if( iiov == p.end() ){
	if( log ) std::cout <<"ERROR: No Iov valid found for target time "<<sourceSince<<std::endl;
	return false;
      }

      persistency::IOVEditor editor;
      if( session.existsIov( destTag ) ){
	editor = session.editIov( destTag );
	if( editor.timeType() != p.timeType() )
	  throwException( "TimeType of the destination tag does not match with the source tag timeType.", "importIovs"); 
	  if( editor.payloadType() != p.payloadObjectType() )
	  throwException( "PayloadType of the destination tag does not match with the source tag payloadType.", "importIovs");
      } else {
	editor = session.createIov( p.payloadObjectType(), destTag, p.timeType(), p.synchronizationType() );
	if( description.empty() ) editor.setDescription( "Created copying iovs from tag "+sourceTag );
	else editor.setDescription( description );
      }

      editor.insert( destSince, (*iiov).payloadId );

      if( log ) std::cout <<"    Flushing changes..."<<std::endl;
      editor.flush();
      ssc.commit();
      return true;
    }

 }
}

