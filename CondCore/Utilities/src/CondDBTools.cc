#include "CondCore/Utilities/interface/CondDBTools.h"
#include "CondCore/Utilities/interface/CondDBImport.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
//
#include "CondCore/CondDB/src/DbCore.h"
//
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/bind.hpp>
#include <memory>

namespace cond {

  namespace persistency {

    cond::Hash importPayload( Session& sourceSession, const cond::Hash& sourcePayloadId, Session& destSession, bool reserialize ){
      if( reserialize ){
	std::pair<std::string,std::shared_ptr<void> > readBackPayload = fetch( sourcePayloadId, sourceSession );
	return import( sourceSession, sourcePayloadId, readBackPayload.first, readBackPayload.second.get(), destSession );
      } else {
	std::string payloadType("");
	cond::Binary payloadData;
	cond::Binary streamerInfoData;
	if( !sourceSession.fetchPayloadData( sourcePayloadId, payloadType, payloadData, streamerInfoData ) ){
	  cond::throwException( "Payload with hash"+sourcePayloadId+" has not been found in the source database.","importPayload");
	}
	boost::posix_time::ptime now = boost::posix_time::microsec_clock::universal_time();
	return destSession.storePayloadData( payloadType, std::make_pair( payloadData, streamerInfoData ),now );  
      }
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
		       const std::string& editingNote,
                       bool override,
		       bool reserialize,
		       bool forceInsert ){
      persistency::TransactionScope ssc( sourceSession.transaction() );
      ssc.start();
      std::cout <<"    Loading source iov..."<<std::endl;
      persistency::IOVProxy p = sourceSession.readIov( sourceTag, true );
      if( p.loadedSize()==0 ) {
	std::cout <<"    Tag contains 0 iovs."<<std::endl; 
	return 0;
      } else {
	std::cout <<"    Iov size:"<<p.loadedSize()<<" timeType:"<<p.timeType()<<" payloadObjectType=\""<<p.payloadObjectType()<<"\""<<std::endl;
      }
      if( (*p.begin()).since > begin ) begin = (*p.begin()).since;
      if( end < begin ) {
	std::cout <<"    No Iov in the selected range."<<std::endl; 
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
      persistency::IOVProxy dp;
      if( exists ){
	dp = destSession.readIov( destTag );
	editor = destSession.editIov( destTag );
	if( !description.empty() ) std::cout <<"   INFO. Destination Tag "<<destTag<<" already exists. Provided description will be ignored."<<std::endl;
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
	  std::cout <<"    WARNING. Skipping duplicated since="<<newSince<<std::endl;
	  continue;
	}
	// make sure that we import the payload _IN_USE_
	auto usedIov = p.getInterval( newSince );
	cond::Hash ph = importPayload( sourceSession, usedIov.payloadId, destSession, reserialize );
	pids.insert( ph );
        bool skip = false;
	if( exists ){
	  // don't insert if the same entry is already there...
	  auto ie = dp.find( newSince );
	  if( ie != dp.end() ){
	    if( ((*ie).since == newSince) && ((*ie).payloadId == usedIov.payloadId) ) {
	      skip = true;
	    }
	  }
	}
	if( !skip ){
	  editor.insert( newSince, ph );
	  sinces.insert( newSince );
	  niovs++;
	  if( niovs && (niovs%1000==0) ) std::cout <<"    Total of iov inserted: "<<niovs<<" payloads: "<<pids.size()<<std::endl;
	}
	iiov++;
	if( iiov == p.end() || (*iiov).since > end ){
	  break;
	} else {
	  newSince = (*iiov).since;
	}
      } 
      if( exists && override ){
	std::cout <<"    Adding overlying iovs..."<<std::endl;
	persistency::IOVProxy dp;
	dp = destSession.iovProxy();
	dp.loadRange( destTag, begin, end );
	std::set<cond::Time_t> extraSinces;
        for( auto iov : dp ){
          auto siov = p.getInterval( iov.since );
	  if( siov.since != iov.since ) {
            if( extraSinces.find( iov.since )==extraSinces.end() ){
	      editor.insert( iov.since, siov.payloadId );
	      extraSinces.insert( iov.since );
	      niovs++;
	      if( niovs && (niovs%1000==0) ) std::cout <<"    Total of iov inserted: "<<niovs<<" payloads: "<<pids.size()<<std::endl;
	    }
	  }
	}        	
      }
      std::cout <<"    Total of iov inserted: "<<niovs<<" payloads: "<<pids.size()<<std::endl;
      std::cout <<"    Flushing changes..."<<std::endl;
      editor.flush( editingNote, forceInsert );
      dsc.commit();
      ssc.commit();
      return niovs;
    }

    bool copyIov( Session& session,
		  const std::string& sourceTag,
		  const std::string& destTag,
		  cond::Time_t sourceSince,
		  cond::Time_t destSince,
		  const std::string& description ){
      persistency::TransactionScope ssc( session.transaction() );
      ssc.start( false );
      std::cout <<"    Loading source iov..."<<std::endl;
      persistency::IOVProxy p = session.readIov( sourceTag, true );
      if( p.loadedSize()==0 ) {
	std::cout <<"    Tag contains 0 iovs."<<std::endl; 
	return false;
      } else {
	std::cout <<"    Iov size:"<<p.loadedSize()<<" timeType:"<<p.timeType()<<" payloadObjectType=\""<<p.payloadObjectType()<<"\""<<std::endl;
      }

      auto iiov = p.find( sourceSince );
      if( iiov == p.end() ){
	std::cout <<"ERROR: No Iov valid found for target time "<<sourceSince<<std::endl;
	return false;
      }

      persistency::IOVEditor editor;
      if( session.existsIov( destTag ) ){
	if( !description.empty() ) std::cout <<"   INFO. Destination Tag "<<destTag<<" already exists. Provided description will be ignored."<<std::endl;
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

      std::cout <<"    Flushing changes..."<<std::endl;
      editor.flush();
      ssc.commit();
      return true;
    }

 }
}

