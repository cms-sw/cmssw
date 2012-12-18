#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
//
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"

namespace cond {
  class ExportAccountUtilities : public Utilities {
    public:
    ExportAccountUtilities();
    ~ExportAccountUtilities();
    int exportTags( const std::vector<std::string>& tagList );
    bool getTagList( const std::string& globalTag, const std::string& accountName, std::vector<std::string>& tagList );
    int execute();
    DbSession m_sourceDb;
    DbSession m_destDb;

  };

  std::string getAccountName( const std::string& connectionString ){
    std::string ret("");
    size_t lastSlash = connectionString.find_last_of('/');
    if( lastSlash != std::string::npos ){
      ret = connectionString.substr( lastSlash+1 );
    }
    return ret;
  }
}

cond::ExportAccountUtilities::ExportAccountUtilities():
  Utilities("cmscond_export_database"),
  m_sourceDb(),
  m_destDb(){
  addAuthenticationOptions();
  addConnectOption("sourceConnect","s","source connection string(required)");
  addConnectOption("destConnect","d","destionation connection string(required)");
  addOption<std::string>("globalTag","t","only export the tags referenced in the specified GT");
  addConnectOption("gtConnect","g","source connection string(required)");
  addOption<size_t>("bunchSize","n","iterate with bunches of specific size (optional)");
  addOption<bool>("check","c","check the iov elements imported.");
  addOption<bool>("verbose","v","verbose printout.");
  addOption<bool>("noconfirm","f","force action without require confirmation (optional)"); 
}

cond::ExportAccountUtilities::~ExportAccountUtilities(){
}

int cond::ExportAccountUtilities::exportTags( const std::vector<std::string>& inputTags ){
  size_t bunchSize = 1;
  if(hasOptionValue("bunchSize")) bunchSize = getOptionValue<size_t>("bunchSize"); 
  bool check = hasOptionValue("check");

  MetaData sourceMetadata( m_sourceDb );
  cond::ExportRegistry registry;

  registry.open( "sqlite_file:export_registry.db" );

  for( std::vector<std::string>::const_iterator iT = inputTags.begin(); iT!= inputTags.end(); ++iT ){
    std::string sourceIovToken=sourceMetadata.getToken( *iT );
      
    cond::IOVProxy sourceIov( m_sourceDb );
    sourceIov.load( sourceIovToken );
    cond::TimeType  sourceIovType = sourceIov.timetype();
    std::set<std::string> const& sourceClasses = sourceIov.payloadClasses();
    int sourceSz = sourceIov.size();
    std::cout <<"---> Exporting tag \""<<*iT<<"\" ("<<sourceSz<<" iov element(s))"<<std::endl;

    cond::DbScopedTransaction transaction(m_destDb);
    transaction.start(false);

    cond::IOVEditor destIov( m_destDb );
    destIov.createIOVContainerIfNecessary();
    m_destDb.storage().lockContainer( IOVNames::container() );

    cond::MetaData  destMetadata( m_destDb );
    if( destMetadata.hasTag( *iT ) ){
      throw cond::Exception(" Tag \""+*iT+"\" already exists in destination database.");
    }
    std::string destIovToken=destIov.create( sourceIovType, sourceIov.iov().lastTill(),sourceIov.iov().metadata() );
    destMetadata.addMapping( *iT ,destIovToken,sourceIovType);
    destIov.setScope( cond::IOVSequence::Tag );

    boost::shared_ptr<IOVImportIterator> importIterator = destIov.importIterator();
    importIterator->setUp( sourceIov, registry, bunchSize );

    size_t totalImported = 0;
    if( bunchSize>1 ){
      unsigned int iter = 0;
      while( importIterator->hasMoreElements() ){
	if(iter>0){
	  transaction.commit();
	  registry.flush();
	  transaction.start();
	  destIov.reload();
	}
        iter++;
        size_t imported = importIterator->importMoreElements();
        totalImported += imported; 
	std::cout <<">> Iteration #"<<iter<<": "<<imported<<" element(s)."<<std::endl;
      }
    } else {
      totalImported = importIterator->importAll();
    }    
    std::cout <<">> Tag \""<<*iT<<"\": "<<totalImported<<" element(s) exported."<<std::endl;
    transaction.commit();
    registry.flush();

    if( check ){
      std::cout<<">> Checking exported iov..."<<std::endl;
      ::sleep(1);
      m_destDb.transaction().start(true);
      std::string dTok = destMetadata.getToken( *iT );
      cond::IOVProxy dIov( m_destDb );
      dIov.load( dTok );
      cond::TimeType  dIovType = sourceIov.timetype();
      if( dIovType != sourceIovType ){
	std::cout <<"ERROR: destionation iov type different from the source type."<<std::endl;
	return -1;
      }
      int dSz = sourceIov.size();
      if( dSz != sourceSz ){
	std::cout <<"ERROR: destionation iov size different from the source size."<<std::endl;
	return -1;
      }

      std::set<std::string> const& dClasses = dIov.payloadClasses();
      if( dClasses != sourceClasses ){
	std::cout <<"ERROR: destination payload type list is different from the source type list."<<std::endl;
	return -1;
      }
      cond::IOVProxy::const_iterator iEnd = dIov.end();
      bool ok = true;
      for( cond::IOVProxy::const_iterator iEl = dIov.begin(); iEl != iEnd; ++iEl ){
	ora::Object payload = m_destDb.getObject(iEl->token());
        bool ok = true;
	if( payload.address() == 0 ){
	  std::cout <<"ERROR: payload retrieved for token \""<<iEl->token()<<"\" cannot be loaded."<<std::endl;
	  ok = false;
	}
        if( dClasses.find( payload.typeName() )== dClasses.end() ){
	  std::cout <<"ERROR: payload type \""<<payload.typeName()<<"\" has not been found in the IOV payload types."<<std::endl;
	  ok = false;
	}
        payload.destruct();
	if(!ok) break;
      }
      m_destDb.transaction().commit();
      if(!ok) return -1;
      std::cout << ">> All of the tests passed. Tag \""<<*iT<<"\" successfully exported."<<std::endl; 
    }
  }
  registry.close();
  return 0;
}

bool cond::ExportAccountUtilities::getTagList( const std::string& globalTag, 
					       const std::string& accountName, 
					       std::vector<std::string>& tagList ){
  if(hasDebug() || hasOptionValue("verbose")) std::cout <<">> Account name=\""<<accountName<<"\""<<std::endl;
  DbSession gtSession =  openDbSession("gtConnect",Auth::COND_READER_ROLE,true);
  gtSession.transaction().start(true);
  coral::ISchema& schema = gtSession.nominalSchema();
  std::string gtTable("TAGTREE_TABLE_");
  gtTable += globalTag;
  if( !schema.existsTable( gtTable ) ){
    std::cout <<"ERROR: The specified Global Tag \"" << globalTag <<"\" has not been found in the database." <<std::endl;
    return false;
  }
  std::auto_ptr<coral::IQuery> query( schema.newQuery() );
  query->addToTableList("TAGINVENTORY_TABLE","TAGS");
  query->addToTableList( gtTable,"GT");
  query->addToOutputList( "TAGS.tagname" );
  coral::AttributeList condData;
  condData.extend<std::string>( "PFNHINT" );
  std::string condDataToBind("%");
  condDataToBind += accountName;
  condData[ "PFNHINT" ].data<std::string>() =  condDataToBind;
  std::string condition("TAGS.tagid = GT.tagid");
  condition += " AND ";
  condition += "TAGS.pfn LIKE :PFNHINT";
  //condition += "TAGS.pfn LIKE " + condDataToBind;
  coral::AttributeList qresult;
  qresult.extend<std::string>("TAGS.tagname");
  query->defineOutput(qresult);
  query->setCondition( condition, condData );
  coral::ICursor& cursor = query->execute();
  while( cursor.next() ) {
    const coral::AttributeList& row = cursor.currentRow();
    tagList.push_back( row["TAGS.tagname"].data<std::string>() );
  }
  cursor.close();
  gtSession.transaction().commit();
  return true;
}

int cond::ExportAccountUtilities::execute(){

  int ret = 0;
  m_sourceDb = openDbSession("sourceConnect", Auth::COND_READER_ROLE, true);
  m_destDb = openDbSession("destConnect", Auth::COND_ADMIN_ROLE );
  std::string sourceConnect = getOptionValue<std::string>( "sourceConnect" );
  std::string destConnect = getOptionValue<std::string>( "destConnect" );
  bool verbose = hasOptionValue("verbose");
  bool noconfirm = hasOptionValue("noconfirm");

  // listing tag in source
  cond::DbScopedTransaction transaction(m_sourceDb);
  transaction.start(true);
  std::vector<std::string> allTags;
  MetaData sourceMetadata( m_sourceDb );
  sourceMetadata.listAllTags( allTags );
  size_t ntags = allTags.size();
  std::vector<std::string> gtTags;
  std::vector<std::string>* tags = 0;

  std::string gt("");
  if( hasOptionValue("globalTag") ) {
    gt = getOptionValue<std::string>("globalTag");
    std::string accountName = getAccountName( sourceConnect );
    if( accountName.empty() ){
	std::cout <<"ERROR: cannot resolve account name from connection string \""<<sourceConnect<<"\""<<std::endl;
	return -1;      
    }
    bool gtok = getTagList( gt, accountName, gtTags );
    if(!gtok){
      return -1;
    } 
    for( std::vector<std::string>::const_iterator igT = gtTags.begin();
	 igT != gtTags.end(); ++igT ){
      bool found = false;
      for( std::vector<std::string>::const_iterator iaT = allTags.begin();
	 iaT != allTags.end(); ++iaT ){
	if( *igT == *iaT ) found = true;
      }
      if(!found) {
	std::cout <<"ERROR: the tag \""<<*igT<<"\" referenced in the Global Tag \""<<gt<<"\" has not been found in the database \""<< sourceConnect<<"\""<<std::endl;
	return -1;
      }
    }
    ntags = gtTags.size();
    std::cout << ">> Global Tag \"" << gt << "\" references "<<ntags << " tag(s) in database \""<<sourceConnect<<"\""<<std::endl;
    tags = &gtTags;
  } else {
    std::cout << ">> "<<allTags.size() <<" tag(s) found in the database \""<<sourceConnect<<"\""<<std::endl;
    tags = &allTags;
  }
  if ( verbose ){
    for ( std::vector<std::string>::const_iterator iT = tags->begin();
	  iT != tags->end(); ++iT ){
      std::cout <<"    "<<*iT<<std::endl;
    }
  }
  if( !noconfirm ){
    std::cout << std::endl;
    std::cout << ">> Confirm the export of "<<ntags<<" tag(s)? (Y/N)";
    char k;
    std::cin >> k;
    if( k!='Y' && k!='y' ){
      return 0;
    }
  }
  int nexp = exportTags( *tags );
  if(nexp<0) {
    return -1;
  }
  std::cout <<">> "<<ntags<<" tag(s) exported to \""<<destConnect<<"\""<<std::endl;
  transaction.commit();
  return ret;
}

int main( int argc, char** argv ){
  cond::ExportAccountUtilities utilities;
  return utilities.run(argc,argv);
}

