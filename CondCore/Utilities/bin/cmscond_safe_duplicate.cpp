#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVProxy.h"


#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
#include "CondCore/DBCommon/interface/TagInfo.h"

#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/DBCommon/interface/IOVInfo.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include <iterator>
#include <limits>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include<sstream>

namespace cond {
  class SafeDuplicateIOVUtilities : public Utilities {
    public:
      SafeDuplicateIOVUtilities();
      ~SafeDuplicateIOVUtilities();
      int execute();
  };
}

cond::SafeDuplicateIOVUtilities::SafeDuplicateIOVUtilities():Utilities("cmscond_safe_duplicate"){
  addConnectOption("connect","c","destionation connection string(required)");
  addAuthenticationOptions();
  addLogDBOption();
  addOption<std::string>("tag","t","tag to duplicate (required)");
  addOption<std::string>("destTag","d","destination tag (default=tag)");
  addOption<cond::Time_t>("beginTime","b","the start time of the IOV interval to duplicate (required)");
  addOption<cond::Time_t>("endTime","e","the end time of the IOV interval to duplicate (-1 for the full range)");
  addOption<std::string>("usertext","x","user text, to be included in usertext column (optional, must be enclosed in double quotes)");
  addOption<bool>("replace","r","allow to update the existing IOVs via insertions or replacements");
  addOption<bool>("dryrun","n","dry run: only print the actions of the call, with no changes on the database entries (optional)");
  addOption<bool>("noconfirm","f","force action without require confirmation (optional)"); 
}

cond::SafeDuplicateIOVUtilities::~SafeDuplicateIOVUtilities(){
}

int cond::SafeDuplicateIOVUtilities::execute(){

  // collecting options
  std::string sourceTag = getOptionValue<std::string>("tag");
  std::string destTag(sourceTag);
  if(hasOptionValue("destTag")) destTag = getOptionValue<std::string>("destTag");
  cond::Time_t begin = 0;
  cond::Time_t end = -1;
  if( hasOptionValue("beginTime") ){
    begin = getOptionValue<cond::Time_t>("beginTime");
  }
  if( hasOptionValue("endTime") ){
    end = getOptionValue<cond::Time_t>("endTime");
  }
  bool doLog = hasOptionValue("logDB");
  std::string destConnect = getOptionValue<std::string>("connect" );
  bool replace = hasOptionValue("replace");
  bool dry = hasOptionValue("dryrun");
  if( dry ){
    std::cout <<"Running in \"dry\" mode: changes won't be applied to the database."<<std::endl;
  }
  bool noconfirm = hasOptionValue("noconfirm");
  std::string usertext("no user comments");
  if( hasOptionValue("usertext")) usertext = getOptionValue<std::string>("usertext");
  
  // inspecting source...
  cond::DbSession db = openDbSession( "connect", Auth::COND_ADMIN_ROLE );
  cond::DbScopedTransaction transaction(db);
  transaction.start(dry);

  // find source tag
  cond::MetaData  metadata(db);
  std::string  iovToken = metadata.getToken(sourceTag);
  if(iovToken.empty()) {
    std::cerr <<"Error: tag \""<<sourceTag<<"\" has not been found in the source database."<<std::endl;
    return 1;
  }
  cond::IOVProxy sourceIov(db);
  sourceIov.load( iovToken );
  cond::TimeType iovType=sourceIov.timetype();
  std::string timetypestr = cond::timeTypeSpecs[iovType].name;
  cond::IOVRange elements = sourceIov.range(begin,end);
  if(elements.size()==0){
    std::cerr <<"Error: no IOV elements found in the selected interval."<<std::endl;
    return 1;
  }

  if( noconfirm || dry ){
    std::cout <<"Found "<<elements.size()<<" IOV element(s) to duplicate."<<std::endl;
  } else {
    std::cout << std::endl;
    std::cout << ">> Confirm the duplication of "<<elements.size()<<" IOV(s)? (Y/N)";
    char k;
    std::cin >> k;
    if( k!='Y' && k!='y' ){
      return 0;
    }
  }

  std::string payload = elements.front().token();
  std::string payloadClass = db.classNameForItem( payload );

  cond::IOVEditor destIov(db);
  if(!dry){
    db.storage().lockContainer(  IOVNames::container() );
  }

  std::vector<std::pair<cond::Time_t,std::string> > extraElements;

  std::string destIovToken("");
  if( metadata.hasTag(destTag) ){
    destIovToken=metadata.getToken(destTag);
    destIov.load( destIovToken );
    if (iovType!=destIov.proxy().timetype()) {
      throw std::runtime_error("iov type in source and dest differs");
    }
    cond::IOVProxy diov = destIov.proxy();
    cond::IOVProxy::const_iterator beginElem = diov.find( begin );
    if( beginElem == diov.end() ){
      beginElem = diov.begin();
    }
    for( cond::IOVProxy::const_iterator i=beginElem; i!= diov.end(); i++ ){
      if( i->since() >= begin ){
	extraElements.push_back( std::make_pair( i->since(), i->token() ) );
      }
    }
  } else {
    std::cout <<"Creating tag \""<<destTag<<"\"."<<std::endl;
    if(!dry){
      destIovToken = destIov.create( iovType );
      metadata.addMapping(destTag,destIovToken,iovType);
    } else {
      std::cout <<"Appending "<<elements.size()<<" new element(s)."<<std::endl;
      transaction.commit();
      return 0;
    }
  }

  size_t extraSize = extraElements.size();
  if( extraSize ){
    std::cout <<"Found "<<extraSize<<" element(s) exceeding target since time="<<begin<<std::endl;
    if(!replace){
      std::cout<<"Can't modify existing IOVs, \"replace\" option has not been specified."<<std::endl;
      return 1;
    }
    std::cout <<"Truncating "<<extraSize<<" element(s)."<<std::endl;
    if(!dry){
      for(size_t i=0;i<extraSize;i++) {
	destIov.truncate( false );
      }      
    }
  }

  for( cond::IOVRange::const_iterator iEl = elements.begin(); iEl != elements.end(); iEl++ ){
    std::cout <<"Appending new element with since="<<iEl->since()<<std::endl;
    if(!dry) {
      destIov.append( iEl->since(), iEl->token() );
    }
  }

  if( extraSize ){
    for( std::vector<std::pair<cond::Time_t,std::string> >::const_iterator iV = extraElements.begin();
	 iV != extraElements.end(); ++iV ){
      if( iV->first > end ){
	std::cout <<"Re-appending existing element with since="<<iV->first<<std::endl;
	if(!dry) {
	  destIov.append( iV->first, iV->second );

	}
      }
    }
  }
  if(!dry)destIov.stamp(cond::userInfo(),false);

  transaction.commit();

  ::sleep(1);
  
  // setup logDB and write on it...
  if (doLog && !dry){
    std::auto_ptr<cond::Logger> logdb;
    cond::DbSession logSession = openDbSession( "logDB", Auth::COND_WRITER_ROLE );
    logdb.reset(new cond::Logger( logSession ));
    logdb->createLogDBIfNonExist();

    cond::UserLogInfo a;
    a.provenance=destConnect+"/"+destTag;
    a.usertext="duplicateIOV V1.0;";
    std::ostringstream ss;
    ss << "From="<< begin <<" to="<<end<<"; " << usertext;
    a.usertext +=ss.str();

    logdb->logOperationNow(a,destConnect,payloadClass,payload,destTag,timetypestr,0,begin);
  }

  return 0;
}

int main( int argc, char** argv ){

  cond::SafeDuplicateIOVUtilities utilities;
  return utilities.run(argc,argv);
}

