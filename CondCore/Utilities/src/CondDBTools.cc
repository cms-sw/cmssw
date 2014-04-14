#include "CondCore/Utilities/interface/CondDBTools.h"
#include "CondCore/Utilities/interface/CondDBImport.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
//
#include <boost/filesystem.hpp>

namespace cond {

  namespace persistency {

    size_t copyTag( const std::string& sourceTag, 
		    Session& sourceSession, 
		    const std::string& destTag, 
		    Session& destSession,
		    UpdatePolicy policy,
		    bool log,
		    bool forValidation ){
      persistency::TransactionScope ssc( sourceSession.transaction() );
      ssc.start();
      if( log && !forValidation ) std::cout <<"    Loading source iov..."<<std::endl;
      persistency::IOVProxy p = sourceSession.readIov( sourceTag, true );
      if( p.loadedSize()==0 ) {
	if( log ) std::cout <<"    Tag contains 0 iovs."<<std::endl; 
	return 0;
      }
      if( log && !forValidation ) std::cout <<"    Copying tag. Iov size:"<<p.loadedSize()<<" timeType:"<<p.timeType()<<" payloadObjectType=\""<<p.payloadObjectType()<<"\""<<std::endl;

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
      if( forValidation ) {
	editor.setValidationMode();
	editor.setDescription("Validation");
      } else {
	editor.setDescription("Tag "+sourceTag+" migrated from "+sourceSession.connectionString());
      }
      size_t niovs = 0;
      std::set<cond::Hash> pids;
      std::set<cond::Time_t> sinces;
      for(  auto iov : p ){
	// skip duplicated sinces
	if( sinces.find( iov.since ) != sinces.end() ){
	  if( log && !forValidation ) std::cout <<"    WARNING. Skipping duplicated since="<<iov.since<<std::endl;
	  continue;
	}
	sinces.insert( iov.since );
	// make sure that we import the payload _IN_USE_
	auto usedIov = p.getInterval( iov.since );
	std::pair<std::string,boost::shared_ptr<void> > readBackPayload = fetch( usedIov.payloadId, sourceSession );
	cond::Hash ph = import( readBackPayload.first, readBackPayload.second.get(), destSession );
	editor.insert( iov.since, ph );
	pids.insert( ph );
	niovs++;
	if( log && !forValidation && niovs && (niovs%1000==0) ) std::cout <<"    Total of iov inserted: "<<niovs<<" payloads: "<<pids.size()<<std::endl;
      } 
      if( log && !forValidation) std::cout <<"    Total of iov inserted: "<<niovs<<" payloads: "<<pids.size()<<std::endl;
      if( log && !forValidation) std::cout <<"    Flushing changes..."<<std::endl;
      editor.flush();
      dsc.commit();
      ssc.commit();
      return niovs;
    }

    size_t importIovs( const std::string& sourceTag, 
		       Session& sourceSession, 
		       const std::string& destTag, 
		       Session& destSession, 
		       cond::Time_t begin,
		       cond::Time_t end,
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
	editor.setDescription( "Copied tag "+sourceTag+" from "+sourceSession.connectionString() );
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
	cond::Hash ph = import( readBackPayload.first, readBackPayload.second.get(), destSession );
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
	if( log ) std::cout <<"    No Iov valid found for target time "<<sourceSince<<std::endl;
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
	editor.setDescription( "Copied tag "+sourceTag+" from "+session.connectionString()  );
      }

      editor.insert( destSince, (*iiov).payloadId );

      if( log ) std::cout <<"    Flushing changes..."<<std::endl;
      editor.flush();
      ssc.commit();
      return true;
    }

    size_t exportTagToFile( const std::string& tag, const std::string& destTag, Session& session, const std::string fileName ){
      ConnectionPool localPool;
      Session writeSession = localPool.createSession( "sqlite:"+fileName, true ); 
      size_t ret = copyTag( tag, session, destTag, writeSession, NEW, false,true );
      return ret;
    }

    bool compareTags( const std::string& firstTag,
		      Session& firstSession, 
		      const std::string& firstFileName, 
		      const std::string& secondTag, 
		      Session& secondSession,
		      const std::string& secondFileName ){
      size_t n1 = exportTagToFile( firstTag, firstTag, firstSession, firstFileName );
      if( ! n1 ){
	std::cout <<"Can't compare empty tag "<<firstTag<<std::endl;
	return false;
      }
      size_t n2 = exportTagToFile( secondTag, firstTag, secondSession, secondFileName );
      if( ! n2 ){
	std::cout <<"Can't compare empty tag "<<secondTag<<std::endl;
	return false;
      }
      if( n1 != n2 ) {
        std::cout <<"    Tag size is different. "<<firstSession.connectionString()<<":"<<firstTag<<": "<<n1<<" "<<
        secondSession.connectionString()<<":"<<secondTag<<": "<<n2<<std::endl;
      }

      FILE* file1 = fopen( firstFileName.c_str(), "r" );
      if( file1 == NULL ){
	throwException("Can't open file "+firstFileName, "compareTags" ); 
      }
      FILE* file2 = fopen( secondFileName.c_str(), "r" );
      if( file2 == NULL ){
	throwException("Can't open file "+secondFileName, "compareTags" ); 
      }
      int N = 10000;
      char buf1[N];
      char buf2[N];
      
      bool cmpOk = true;
      size_t totSize = 0;
      do {
	size_t r1 = fread( buf1, 1, N, file1 );
	size_t r2 = fread( buf2, 1, N, file2 );
	
	if( r1 != r2 || memcmp( buf1, buf2, r1)) {
	  cmpOk = false;
	  break;
	}
	totSize += r1;
      } while(!feof(file2) || !feof(file2));
      
      std::cout <<"    "<<totSize<<" bytes compared."<<std::endl;
      fclose( file1 );
      fclose( file2 );

      if( cmpOk ){
	boost::filesystem::path fp1( firstFileName );
	boost::filesystem::remove( fp1 );
	boost::filesystem::path fp2( secondFileName );
	boost::filesystem::remove( fp2 );
      }

      return cmpOk;
    }

    bool validateTag( const std::string& refTag, 
		      Session& refSession, 
		      const std::string& candTag, 
		      Session& candSession ){
      std::cout <<"    Validating..."<<std::endl;
      std::tuple<std::string,std::string,std::string> connPars = persistency::parseConnectionString( refSession.connectionString() );  
      std::string dbLabel = std::get<2>( connPars );
      std::string tagLabel = dbLabel+"_"+refTag;
      std::string refFile = tagLabel+"_ref.db";
      std::string candFile = tagLabel+"_cand.db";
      bool ret = compareTags( refTag, refSession, refFile, candTag, candSession, candFile );
      if( ret ) {    
	boost::filesystem::path refF( refFile );
	if( boost::filesystem::exists(refF) ) boost::filesystem::remove( refF );
	boost::filesystem::path candF( candFile );
	if( boost::filesystem::exists(candF) ) boost::filesystem::remove( candF );
      }
      return ret;
    }

 }
}

