#include "CondCore/Utilities/interface/ExportIOVUtilities.h"

#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"

#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVProxy.h"

#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
#include "CondCore/DBCommon/interface/TagInfo.h"

#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include <iterator>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

cond::ExportIOVUtilities::ExportIOVUtilities(std::string const & name):Utilities(name){
  addDictionaryOption();
  addAuthenticationOptions();
  addConfigFileOption();
  addLogDBOption();
  addConnectOption("sourceConnect","s","source connection string(required)");
  addConnectOption("destConnect","d","destionation connection string(required)");
  addOption<std::string>("inputTag","i","tag to export( default = destination tag)");
  addOption<std::string>("destTag","t","destination tag (required)");
  addOption<cond::Time_t>("beginTime","b","begin time (first since) (optional)");
  addOption<cond::Time_t>("endTime","e","end time (last till) (optional)");
  addOption<bool>("outOfOrder","o","allow out of order merge (optional, default=false)");
  addOption<size_t>("bunchSize","n","iterate with bunches of specific size (optional)");
  addOption<std::string>("usertext","x","user text, to be included in usertext column (optional, must be enclosed in double quotes)");
  addSQLOutputOption();
}

cond::ExportIOVUtilities::~ExportIOVUtilities(){
}

int cond::ExportIOVUtilities::execute(){
    
  std::string sourceConnect = getOptionValue<std::string>("sourceConnect");
  std::string destConnect = getOptionValue<std::string>("destConnect");

  std::string destTag = getOptionValue<std::string>("destTag");
  std::string inputTag(destTag);
  if( hasOptionValue("inputTag") ) inputTag = getOptionValue<std::string>("inputTag");
  std::string usertext("no user comments");
  if( hasOptionValue("usertext")) usertext = getOptionValue<std::string>("usertext");
  bool doLog = hasOptionValue("logDB");

  cond::Time_t since = std::numeric_limits<cond::Time_t>::min();
  if( hasOptionValue("beginTime" )) since = getOptionValue<cond::Time_t>("beginTime");
  cond::Time_t till = std::numeric_limits<cond::Time_t>::max();
  if( hasOptionValue("endTime" )) till = getOptionValue<cond::Time_t>("endTime");
  
  size_t bunchSize = 1;
  if(hasOptionValue("bunchSize")) bunchSize = getOptionValue<size_t>("bunchSize"); 
  
  std::string sqlOutputFileName("sqlmonitoring.out");
  bool debug=hasDebug();
  bool outOfOrder = hasOptionValue("outOfOrder");

  std::string sourceIovToken("");
  std::string destIovToken("");
  bool newIOV = true;
  cond::TimeType sourceIovType;

  cond::DbSession sourceDb = openDbSession("sourceConnect", Auth::COND_READER_ROLE, true);
  cond::DbSession destDb = openDbSession("destConnect", Auth::COND_WRITER_ROLE );
    
  std::auto_ptr<cond::Logger> logDb;
  cond::DbSession logSession;

  std::string payloadToken("");
  std::string payloadClasses("");
  int iovSize = 0;
  int ncopied = 0;
  cond::UserLogInfo a;
  if (doLog) {
    logSession = openDbSession( "logDB",Auth::COND_WRITER_ROLE );
    logDb.reset(new cond::Logger(logSession));
    logDb->createLogDBIfNonExist();
    a.provenance=sourceConnect+"/"+inputTag;
    a.usertext="exportIOV V4.0;";
  }

  // find tag in source
  sourceDb.transaction().start(true);
  cond::MetaData  sourceMetadata(sourceDb);
  sourceIovToken=sourceMetadata.getToken(inputTag);
  if(sourceIovToken.empty()) 
    throw std::runtime_error(std::string("tag ")+inputTag+std::string(" not found") );
  
  if(debug){
    std::cout<<"source iov token "<<sourceIovToken<<std::endl;
  }
  
  cond::IOVProxy sourceIov( sourceDb );
  sourceIov.load( sourceIovToken );
  sourceIovType = sourceIov.timetype();
  std::string const & timetypestr = cond::timeTypeSpecs[sourceIovType].name;
  if(debug){
    std::cout<<"source iov type "<<sourceIovType<<std::endl;
  }
 
  if( doLog ){
    std::set<std::string> pclasses = sourceIov.payloadClasses();
    iovSize = sourceIov.size();
    std::ostringstream stream;
    std::copy(pclasses.begin(), pclasses.end(), std::ostream_iterator<std::string>(stream, ", "));
    payloadClasses = stream.str();
  }

  try{
    // find tag in destination
    cond::DbScopedTransaction transaction(destDb);
    transaction.start(false);

    int oldSize=0;
    cond::IOVEditor destIov( destDb );
    destIov.createIOVContainerIfNecessary();
    destDb.storage().lockContainer( IOVNames::container() );

    cond::MetaData  destMetadata( destDb );
    if( destMetadata.hasTag(destTag) ){
      destIovToken=destMetadata.getToken(destTag);
      destIov.load( destIovToken );
      oldSize = destIov.proxy().size();
      if (sourceIovType!=destIov.timetype()) {
	throw std::runtime_error("iov type in source and dest differs");
      }
    } else {
      newIOV = true;
      destIovToken=destIov.create( sourceIovType, sourceIov.iov().lastTill(),sourceIov.iov().metadata() );
      destMetadata.addMapping(destTag,destIovToken,sourceIovType);
      destIov.setScope( cond::IOVSequence::Tag );
    }
    if(debug){
      std::cout<<"dest iov token "<<destIovToken<<std::endl;
      std::cout<<"dest iov type "<<sourceIovType<<std::endl;
    }
    
    since = std::max(since, cond::timeTypeSpecs[sourceIovType].beginValue);
    till  = std::min(till,  cond::timeTypeSpecs[sourceIovType].endValue);
    
    boost::shared_ptr<IOVImportIterator> importIterator = destIov.importIterator();
    importIterator->setUp( sourceIov, since, till, outOfOrder, bunchSize );

    size_t totalImported = 0;
    if( bunchSize>1 ){
      unsigned int iter = 0;
      while( importIterator->hasMoreElements() ){
	if(iter>0){
	  transaction.commit();
	  transaction.start();
	  destIov.reload();
	}
        iter++;
        size_t imported = importIterator->importMoreElements();
        totalImported += imported; 
	std::cout <<"Iteration #"<<iter<<": "<<imported<<" element(s)."<<std::endl;
      }
    } else {
      totalImported = importIterator->importAll();
    }    
    std::cout <<totalImported<<" element(s) exported."<<std::endl;

   ::sleep(1);
    
    // grab info
    // call IOV proxy with keep open option: it is required to lookup the payload class. A explicit commit will be needed at the end.
    if (doLog) {
      IOVProxy diov = destIov.proxy();
      std::ostringstream stream;
      std::copy(diov.payloadClasses().begin(), diov.payloadClasses().end(), std::ostream_iterator<std::string>(stream, ", "));
      payloadClasses = stream.str();
      iovSize = diov.size();
      ncopied = iovSize-oldSize; 
      if ( ncopied == 1) {
	// get last object
        const IOVElement& last = diov.iov().iovs().back();
	payloadToken=last.token();
	payloadClasses = destDb.classNameForItem( payloadToken );
      } 
      if (newIOV) a.usertext+= "new tag;";
      std::ostringstream ss;
      ss << "since="<< since <<", till="<< till << ", " << usertext << ";";
      ss << " copied="<< ncopied <<";";
      a.usertext +=ss.str();  
      logDb->logOperationNow(a,destConnect,payloadClasses,payloadToken,destTag,timetypestr,iovSize-1,since);
    }
    transaction.commit();
    sourceDb.transaction().commit();
  }catch ( cond::Exception const& er ){
    if (doLog) {
      if (newIOV) a.usertext+= "new tag;";
      std::ostringstream ss;
      ss << "since="<< since <<", till="<< till << ", " << usertext << ";";
      ss << " copied="<< ncopied <<";";
      a.usertext +=ss.str();
      logDb->logFailedOperationNow(a,destConnect,payloadClasses,payloadToken,destTag,timetypestr,iovSize-1,since,std::string(er.what()));
    }   
    sourceDb.transaction().commit();
    throw;
  }
      
  return 0;
}


