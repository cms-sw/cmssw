#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/DBCommon/interface/IOVInfo.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
//
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"

#include <fstream>

namespace cond {

  const char* orconProd = "oracle://cms_orcon_prod/";
  const char* orconAdg = "oracle://cms_orcon_adg/";
  const char* frontierProd = "frontier://FrontierProd/";
  const char* frontierArc = "frontier://FrontierArc/";

  const char* runInfoAccount = "frontier://FrontierProd/CMS_COND_31X_RUN_INFO";
  const char* runInfoTag = "runinfo_31X_hlt";

  const char* COND_GT_BOOK_TABLE = "COND_GT_BOOK";
  const char* COND_GT_MERGE_TABLE = "COND_GT_MERGE";

  static const std::string NEW_GT("new_gt=");
  static const std::string TAG_PFIX("tag_pfix=");
  static const std::string GT_DB("gt_db=");
  static const std::string MERGE("merge_gt=");
  static const std::string VALID("valid_till=");
  static const std::string DEST_DB("dest_db=");

  struct RecordData {
    std::string tag;
    std::string pfn;
    std::string objectName;
    std::string recordName;
    std::string recordLabel;
  };

  class GTManager : public Utilities {
    public:
    GTManager();
    ~GTManager();
    int execute();
  private:
    cond::Time_t getTimeStampForRunNumber( cond::Time_t runN );

    cond::Time_t rebuildTime( cond::Time_t runN, cond::TimeType iovType );

    void parseInput( const std::string& inputFile );

    bool processTag( const std::string&  inputConnectionString, const std::string& inputTag, 
		     const std::string& destConnectionString, const std::string& destTag,
		     Time_t begin, Time_t end, bool newTag );

    void dumpConfiguration( const std::string& gtConnectionString, 
			    const std::string& gtName,
			    const std::map<std::string, RecordData >& gtData, 
			    std::ostream& out );

    std::string templateFile();

  private:

    std::string m_new_gt;
    std::string m_tag_pfix;
    std::string m_gt_db;
    std::string m_dest_db;
    std::vector<std::pair<std::string,Time_t> > m_merge_list;
    std::map<cond::Time_t,cond::Time_t> m_timeStampCache;

  };

}

cond::GTManager::GTManager():
  Utilities("cmscond_gt_merge"),
  m_new_gt(""),
  m_tag_pfix(""),
  m_gt_db(""),
  m_dest_db(""),
  m_merge_list(),
  m_timeStampCache()
{
  addOption<std::string>("input_file","i","the input file with the directives");
  addOption<std::string>("output_file","o","the output file with the new GT configuration");
  addOption<bool>("dump_template","d","dump a template for the input file");
  addOption<bool>("verbose","v","switch on the verbose output");
}

cond::GTManager::~GTManager(){
}

namespace cond {

  std::string getAccountName( const std::string& connectionString ){
    if ( connectionString.find( "sqlite_file:" )==0 ){
      return std::string("");
    }
    const char* db[4];
    db[0] = orconProd;
    db[1] = orconAdg;
    db[2] = frontierProd;
    db[3] = frontierArc;
    std::string account("");
    size_t index = std::string::npos;
    unsigned int i=0;
    while ( index == std::string::npos ){
      index = connectionString.find( db[i] );
      std::string acc("");
      if( index != std::string::npos ){
	std::string prefix(db[i]);
	acc = connectionString.substr( prefix.size() );
	account = acc;
	if( i==3 ) account = acc.substr( 0,acc.size()-5);
      }
      i++;
    }
    return account;
  }

  std::string getUpdateConnectionString( const std::string& connectionString ){
    return orconProd+getAccountName(connectionString);
  }

  bool getRecordList( DbSession& db, 
		      const std::string& globalTag, 
		      std::map<std::string, RecordData >& recordList ){
    coral::ISchema& schema = db.nominalSchema();
    std::string gtTable("TAGTREE_TABLE_");
    gtTable += globalTag;
    std::auto_ptr<coral::IQuery> query( schema.newQuery() );
    query->addToTableList("TAGINVENTORY_TABLE","TAGS");
    query->addToTableList( gtTable,"GT");
    query->addToOutputList( "TAGS.tagname" );
    query->defineOutputType( "TAGS.tagname", coral::AttributeSpecification::typeNameForType<std::string>());
    query->addToOutputList( "TAGS.recordname" );
    query->defineOutputType( "TAGS.recordname", coral::AttributeSpecification::typeNameForType<std::string>());
    query->addToOutputList( "TAGS.labelname" );
    query->defineOutputType( "TAGS.labelname", coral::AttributeSpecification::typeNameForType<std::string>());
    query->addToOutputList( "TAGS.objectname" );
    query->defineOutputType( "TAGS.objectname", coral::AttributeSpecification::typeNameForType<std::string>());
    query->addToOutputList( "TAGS.pfn" );
    query->defineOutputType( "TAGS.pfn", coral::AttributeSpecification::typeNameForType<std::string>());
    coral::AttributeList condData;
    std::string condition("TAGS.tagid = GT.tagid");
    query->setCondition( condition, condData );
    coral::ICursor& cursor = query->execute();
    bool ret = false;
    while( cursor.next() ) {
      const coral::AttributeList& row = cursor.currentRow();
      RecordData data;
      data.recordName = row["TAGS.recordname"].data<std::string>();
      data.recordLabel = row["TAGS.labelname"].data<std::string>();
      std::string recordKey = data.recordName;
      if(!data.recordLabel.empty()) recordKey += "_"+data.recordLabel;
      data.tag = row["TAGS.tagname"].data<std::string>();
      data.pfn = row["TAGS.pfn"].data<std::string>();
      data.objectName = row["TAGS.objectname"].data<std::string>();
      recordList.insert( std::make_pair( recordKey, data ) );
      ret = true;
    }
    cursor.close();
    return ret;
  }

  bool compareGT( const std::pair<std::string,Time_t>& gt1, const std::pair<std::string,Time_t>& gt2 ){
    if( gt1.second==gt2.second ){
      return gt1.first < gt2.first;
    }
    return gt1.second < gt2.second;
  }

  bool findParam( const std::string& line, const std::string& paramTag, size_t start){
    if( line.size()<start ) return false; 
    if( paramTag.size() > (line.size()-start-1)) return false;
    if( line.substr(start,paramTag.size())!=paramTag ) return false; 
    return true;
  }

  size_t getParam( const std::string& line, const std::string& paramTag, size_t start, std::string& param ){
    size_t index = line.find( paramTag, start );
    if( index == std::string::npos ) return std::string::npos;
    size_t end = line.find( ";",index );
    if( end == std::string::npos ) {
      throw UtilitiesError("Error in parsing configuration file, can't resolve parameter \""+paramTag.substr(0,paramTag.size()-1)+"\"");
    }
    param = line.substr(index+paramTag.size(),end-index-paramTag.size());
    return end;
  }

}

cond::Time_t cond::GTManager::getTimeStampForRunNumber( cond::Time_t runN ){
  std::map<cond::Time_t,cond::Time_t>::const_iterator iT = m_timeStampCache.find( runN );
  cond::Time_t ts = 0;
  if( iT != m_timeStampCache.end() ){
    ts = iT->second;
  } else {
    DbSession runInfoDb = newDbSession( runInfoAccount, Auth::COND_READER_ROLE, true );
    runInfoDb.transaction().start( true );
    MetaData md( runInfoDb );
    std::string tk = md.getToken(runInfoTag);
    if(tk.empty()) 
      throw std::runtime_error(std::string("RunInfo tag ")+std::string(runInfoTag)+std::string(" not found") );
    cond::IOVProxy iov( runInfoDb );
    iov.load( tk );
    cond::IOVProxy::const_iterator iR = iov.find( runN );
    // take the next iov: the stop is the start of the next...
    iR++;
    if( iR == iov.end() ){
      std::cout <<"Run "<<runN<<" has not been found in RunInfo IOV."<<std::endl;
      throw std::runtime_error(std::string("Could not convert Run to Timestamp."));
    }
    std::string tok = iR->token();
    boost::shared_ptr<RunInfo> rs = runInfoDb.getTypedObject<RunInfo>( tok );
    ts = rs->m_start_time_ll;
    runInfoDb.transaction().commit();
    m_timeStampCache.insert( std::make_pair( runN, ts ) );
  }
  std::cout <<"     --> run "<<runN<<" has been converted to timestamp="<<ts<<std::endl;
  return ts;
}

cond::Time_t cond::GTManager::rebuildTime( cond::Time_t runN, cond::TimeType iovType ){
  if ( runN == 0 ) return 0;
  if( iovType == cond::timestamp ) return getTimeStampForRunNumber( runN );
  if( iovType == cond::lumiid ){
    UnpackedTime ut(runN,1);
    return time::pack(ut);
  } 
  return runN;
}

void cond::GTManager::parseInput( const std::string& inputFileName ){
  m_new_gt.clear();
  m_gt_db.clear();
  m_dest_db.clear();
  m_merge_list.clear();
  std::ifstream inputFile (inputFileName.c_str());
  if (inputFile.is_open()){
    while ( inputFile.good() ){
      std::string line;
      getline (inputFile, line);
      if( findParam( line, MERGE, 0  ) ){
	std::string merge_gt("");
	size_t curs = getParam( line, MERGE, 0, merge_gt );
	if( !findParam( line, VALID, curs+1 ) ) throw UtilitiesError("Parameter \"valid_till\" not specified for merge_gt=\""+merge_gt+"\"");
	std::string validityStr("");
	getParam( line, VALID, curs+1, validityStr );
	Time_t validity = 0;
	std::istringstream istr(validityStr);
	istr >> validity;
	std::ostringstream ostr;
	ostr << validity;
	if( ostr.str()!=validityStr ){
	  throw UtilitiesError("Provided parameter \"valid_till\" for merge_gt=\""+line.substr(MERGE.size())+"\" is invalid.");
	}
	m_merge_list.push_back( std::make_pair( merge_gt, validity ) );
      } else if ( findParam( line, DEST_DB, 0 ) ){
	getParam( line, DEST_DB, 0 , m_dest_db );
      } else if ( findParam( line, NEW_GT, 0 ) ){
	size_t curs = getParam( line, NEW_GT, 0, m_new_gt );
	if( !findParam( line, TAG_PFIX, curs+1 ) ) throw UtilitiesError("Parameter \"tag_pfix\" not specified for new_gt");
	getParam( line, TAG_PFIX, curs+1, m_tag_pfix );
      } else if ( findParam( line, GT_DB, 0 ) ){
	getParam( line, GT_DB, 0, m_gt_db );
      }
    }
    inputFile.close();
  } else {
    std::string msg = "Provided Input File \""+inputFileName+"\n is invalid.";
    throw UtilitiesError(msg);      
  }
}

bool cond::GTManager::processTag( const std::string& inputConnectionString, const std::string& inputTag, 
				  const std::string& destConnectionString, const std::string& destTag,
				  Time_t begin, Time_t end, bool newTag ){
  std::cout <<"     --> exporting tag \""<<inputTag<<"\" from db \""<<inputConnectionString<<"\""<<std::endl;
  std::cout <<"     --> destination tag \""<<destTag<<"\" on destination db \""<<destConnectionString<<"\"."<<std::endl;
  DbSession srcDb = newDbSession( inputConnectionString, Auth::COND_READER_ROLE, true );
  srcDb.transaction().start( true );
  MetaData sourceMetadata( srcDb );
  std::string sourceIovToken = sourceMetadata.getToken(inputTag);
  if(sourceIovToken.empty()) 
    throw std::runtime_error(std::string("tag ")+inputTag+std::string(" not found") );
    
  cond::IOVProxy sourceIov( srcDb );
  sourceIov.load( sourceIovToken );
  //std::set<std::string> const& sourceClasses = sourceIov.payloadClasses();
  cond::TimeType  sourceIovType = sourceIov.timetype();

  if( sourceIovType != cond::timestamp  &&  sourceIovType != cond::lumiid && sourceIovType != cond::runnumber ){
    srcDb.transaction().commit();
    return false;
  } 

  begin = rebuildTime( begin, sourceIovType );
  if( begin ) begin++;
  end = rebuildTime( end, sourceIovType );

  DbSession destDb = newDbSession( destConnectionString, Auth::COND_WRITER_ROLE );
  cond::DbScopedTransaction transaction( destDb);
  transaction.start(false);
  
  cond::IOVEditor destIov( destDb );
  destIov.createIOVContainerIfNecessary();
  destDb.storage().lockContainer( IOVNames::container() );
  
  cond::MetaData  destMetadata( destDb );
  if( destMetadata.hasTag( destTag ) ){
    if(newTag) throw cond::Exception(" Tag \""+destTag+"\" already exists in destination database.");
    std::string destIovToken = destMetadata.getToken(destTag);
    destIov.load( destIovToken );
  } else {
    if(!newTag) throw cond::Exception(" Tag \""+destTag+"\" does not exists in destination database.");
    std::string destIovToken = destIov.create( sourceIovType, sourceIov.iov().lastTill(),sourceIov.iov().metadata() );
    destIov.setScope( cond::IOVSequence::Tag );
    destMetadata.addMapping( destTag,destIovToken,sourceIovType);
  }
  
  cond::IOVRange rg = sourceIov.range( begin, end );
  size_t exported = 0;
  std::cout <<"     --> selecting interval from "<<begin<<" to "<<end<<": "<<rg.size()<<" element(s)"<<std::endl;
  for( cond::IOVRange::const_iterator iEl = rg.begin();
       iEl != rg.end(); iEl++ ){
    Time_t since = iEl->since();
    std::string payload = iEl->token();
    if( destConnectionString != inputConnectionString ){
      // copy the payload in the dest db
      payload = destDb.importObject( srcDb,payload );
    }
    destIov.append(since,payload);
    exported++;
  }
  destIov.stamp(cond::userInfo(),false);
  transaction.commit();
  srcDb.transaction().commit();
  //std::cout <<exported<<" element(s) exported in tag \""<<destTag<<"\"."<<std::endl;
  return true;
}

void cond::GTManager::dumpConfiguration( const std::string& gtConnectionString,
					 const std::string& gtName,
					 const std::map<std::string, RecordData>& gtData, 
					 std::ostream& out ){
  out << "[COMMON]"<<std::endl;
  out << "connect="<<gtConnectionString<<std::endl;
  out << std::endl;
  out <<"[TAGINVENTORY]"<<std::endl;
  out << "tagdata="<<std::endl;
  for( std::map<std::string, RecordData>::const_iterator iR = gtData.begin();
       iR != gtData.end(); ++iR ){
    out<<" "<<iR->second.tag<<"{pfn="<<iR->second.pfn<<",objectname="<<iR->second.objectName<<",recordname="<<iR->second.recordName; 
    std::string label = iR->second.recordLabel;
    if( !label.empty() ){
      out<<",labelname="<<label;
    }
    out << "};"<<std::endl;
  }
  out<<std::endl;
  out <<"[TAGTREE "<<gtName<<"]"<<std::endl;
  out <<"root=All"<<std::endl;
  out <<"nodedata=Calibration{parent=All}"<<std::endl;
  out <<"leafdata="<<std::endl;
  for( std::map<std::string, RecordData>::const_iterator iR = gtData.begin();
       iR != gtData.end(); ++iR ){
    out<<" "<<iR->first<<"{parent=Calibration,tagname="<<iR->second.tag<<",pfn="<<iR->second.pfn<<"};"<<std::endl; 
  }
}

std::string cond::GTManager::templateFile(){
  std::stringstream s;
  s<<NEW_GT<<"<label for new GT>;"<<TAG_PFIX<<"<Postfix for the new tags generated>;"<<std::endl;
  s<<GT_DB<<"<connection string for the GT database>;"<<std::endl;
  s<<DEST_DB<<"<connection string for the new tags destination database, optional>;"<<std::endl;
  s<<MERGE<<"<label for the GT1 to merge>;"<<VALID<<"<GT1 validity upper limit>;"<<std::endl;
  s<<MERGE<<"<label for the GT2 to merge>;"<<VALID<<"<GT2 validity upper limit>;"<<std::endl;
  s<<MERGE<<"<label for the GT3 to merge>;"<<VALID<<"<GT3 validity upper limit>;"<<std::endl;
  return s.str();
}

int cond::GTManager::execute(){

  if( hasOptionValue("dump_template") ) {
    std::cout << templateFile() <<std::endl;
    return 0;
  }

  if( hasDebug() ) coral::MessageStream::setMsgVerbosity( coral::Debug );

  std::string inputFileName("");
  if ( !hasOptionValue("input_file") ) {
    throw UtilitiesError("Mandatory parameter \"input_file\" has not been provided.");
  }
  inputFileName = getOptionValue<std::string>( "input_file" );
  std::string outputFileName("");
  if ( hasOptionValue("output_file") ) outputFileName = getOptionValue<std::string>( "output_file" );

  parseInput( inputFileName );

  if( hasOptionValue("verbose") ){
    std::cout <<"# Input file name:\""<<inputFileName<<"\""<<std::endl;
    if( !outputFileName.empty() )std::cout <<"# Output file name:\""<<outputFileName<<"\""<<std::endl;
    std::cout <<"# Building new GT \""<<m_new_gt<<"\". Post fix for new tags:\""<<m_tag_pfix<<"\""<<std::endl;
    std::cout <<"# GT source database: \""<<m_gt_db<<"\""<<std::endl;
    if( !m_dest_db.empty() ) std::cout <<"# New tags destination database: \""<<m_dest_db<<"\""<<std::endl;
    for( std::vector<std::pair<std::string,Time_t> >::const_iterator igt = m_merge_list.begin();
	 igt != m_merge_list.end(); ++igt ){
      std::cout <<"# GT \""<<igt->first<<"\" will be merged until run:"<<igt->second<<"."<<std::endl;
    }
  }
  
  if( m_merge_list.size() == 1 ){
    std::cout <<"WARNING: The merge list contains only 1 gt." <<std::endl;
    return 1;
  }

  std::sort( m_merge_list.begin(),m_merge_list.end(), compareGT );

  // the list of final tags is the tag content of the most recent = the longest lasting among the gt to merge
  std::map<std::string,std::pair<std::string,std::string> > newTagList;
  std::map<std::string, RecordData > tmpGTList;
  
  DbSession sourcedb = newDbSession( m_gt_db, Auth::COND_READER_ROLE, true );
  sourcedb.transaction().start(true);      
  if(!getRecordList( sourcedb, m_merge_list.back().first, tmpGTList ) ){
    std::cout <<"ERROR: The gt \""<<m_merge_list.back().first<<"\" is empty." <<std::endl;
    return 1;
  }
      
  unsigned long long min = 0;
  unsigned long long max = TIMELIMIT;

  for(size_t ig = 0; ig < m_merge_list.size()-1; ig++ ){
    std::string gt = m_merge_list[ig].first;
    std::cout <<"## Processing GT=\""<<gt<<"\""<<std::endl;
    max = m_merge_list[ig].second;
    std::map<std::string, RecordData > oldGTList;
    if(getRecordList( sourcedb, gt, oldGTList ) ){
      for( std::map<std::string, RecordData >::const_iterator it = tmpGTList.begin();
	   it != tmpGTList.end(); it++ ){
	std::string rec = it->first;
	// search the record in the old list 
	std::map<std::string, RecordData >::const_iterator ift = oldGTList.find(rec);
	if( ift != oldGTList.end() && ift->second.tag != it->second.tag ){
	  // in this case a new tag is required!
	  // check if it is already available in the new tag list
	  std::map<std::string,std::pair<std::string,std::string> >::iterator inew = newTagList.find( rec );
	  std::string destTag("");
	  std::string destConnString("");
	  bool newTag = false;
	  if( inew != newTagList.end() ){
	    // the new tag already exist... 
	    destTag = inew->second.first;
	    destConnString = inew->second.second;
	  } else {
	    // the new tag has to be created...
	    newTag = true;
	    destTag = it->second.tag+"_"+m_tag_pfix;
	    destConnString = getUpdateConnectionString( it->second.pfn );
	  }
	  if( !m_dest_db.empty() ){
	    destConnString = m_dest_db;
	  }
	  std::cout <<"  ** Record \""<<ift->first<<"\" has a different tag."<<std::endl; 		
	  processTag( ift->second.pfn, ift->second.tag, destConnString, destTag, min, max, newTag );
	  if( newTag ) newTagList.insert(std::make_pair(it->first,std::make_pair( destTag, destConnString )));
	  
	} else {
	  if( ift== oldGTList.end() ) {
	    if( hasOptionValue("verbose") ) std::cout <<"  ** Record \""<<rec<<"\" not found in GT \""<<gt<<"\""<<std::endl; 
	  } else {
	    if( hasOptionValue("verbose") ) std::cout <<"  ** Record \""<<ift->first<<"\" has the same tag as in the reference GT."<<std::endl; 		
	  }
	}
      }
    } else {
      std::cout <<"## WARNING: GT=\""<<gt<<"\" is empty"<<std::endl;	  
    }
    min = max; 
  }
  std::string gtConnStr = sourcedb.connectionString();
  sourcedb.transaction().commit();
  std::map<std::string, RecordData > finalTagList;
  max = m_merge_list[m_merge_list.size()-1].second;
  std::cout <<"## Processing GT=\""<<m_merge_list[m_merge_list.size()-1].first<<"\""<<std::endl;
  // the most recent has still to be processed...
  for( std::map<std::string, RecordData >::const_iterator it = tmpGTList.begin();
       it != tmpGTList.end(); it++ ){
    std::map<std::string,std::pair<std::string,std::string> >::const_iterator iRec = newTagList.find( it->first );
    RecordData data;
    data = it->second;
    if( iRec != newTagList.end() ){
      std::cout <<"  ** Record \""<<it->first<<"\" has a new tag in the new GT."<<std::endl;
      data.tag = iRec->second.first;
      std::string destConnString = iRec->second.second;
      std::string accountLabel = getAccountName( destConnString );
      if( !accountLabel.empty() ) { 
	data.pfn = frontierProd+accountLabel ;
      } else {
	data.pfn = destConnString;
      }
      if( !m_dest_db.empty() ){
	destConnString = m_dest_db;
      }
      processTag( it->second.pfn, it->second.tag, destConnString, data.tag, min, max, false );
    } else {
      if( hasOptionValue("verbose") ) std::cout <<"  ** Record \""<<it->first<<"\" has the same tag as in the new GT."<<std::endl; 
    }
    finalTagList.insert( std::make_pair( it->first,data) );
  }
  // finally dump the new GT configuration!
  std::string fileName = m_new_gt+".conf";
  std::ofstream outFile ( fileName.c_str() );
  if (outFile.is_open()){
    dumpConfiguration( gtConnStr, m_new_gt, finalTagList, outFile );
  }
  outFile.close();
  std::cout <<"GT \""<<m_new_gt<<"\" built."<<std::endl;
  return 0;
}

int main( int argc, char** argv ){
  cond::GTManager mgr;
  return mgr.run(argc,argv);
}

