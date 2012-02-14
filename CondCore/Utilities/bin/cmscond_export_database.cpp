#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/IOVService/interface/IOVEditor.h"

namespace cond {
  class ExportAccountUtilities : public Utilities {
    public:
      ExportAccountUtilities();
      ~ExportAccountUtilities();
      int execute();
  };
}

cond::ExportAccountUtilities::ExportAccountUtilities():Utilities("cmscond_export_database"){
  addAuthenticationOptions();
  addConnectOption("sourceConnect","s","source connection string(required)");
  addConnectOption("destConnect","d","destionation connection string(required)");
  addOption<size_t>("bunchSize","n","iterate with bunches of specific size (optional)");
  addOption<bool>("check","c","check the iov elements imported.");
}

cond::ExportAccountUtilities::~ExportAccountUtilities(){
}

int cond::ExportAccountUtilities::execute(){

  cond::DbSession sourceDb = openDbSession("sourceConnect", true);
  cond::DbSession destDb = openDbSession("destConnect");
  size_t bunchSize = 1;
  if(hasOptionValue("bunchSize")) bunchSize = getOptionValue<size_t>("bunchSize"); 

  // listing tag in source
  sourceDb.transaction().start(true);
  cond::MetaData  sourceMetadata(sourceDb);
  std::vector<std::string> inputTags;
  sourceMetadata.listAllTags( inputTags );
  std::cout << inputTags.size() <<" tags found in the source database."<<std::endl;
  for( std::vector<std::string>::const_iterator iT = inputTags.begin(); iT!= inputTags.end(); ++iT ){
    std::string sourceIovToken=sourceMetadata.getToken( *iT );
      
    cond::IOVProxy sourceIov( sourceDb );
    sourceIov.load( sourceIovToken );
    cond::TimeType  sourceIovType = sourceIov.timetype();
    std::set<std::string> const& sourceClasses = sourceIov.payloadClasses();
    int sourceSz = sourceIov.size();
    std::cout <<"---> Exporting tag \""<<*iT<<"\" ("<<sourceSz<<" iov elements)"<<std::endl;

    cond::DbScopedTransaction transaction(destDb);
    transaction.start(false);

    cond::IOVEditor destIov( destDb );
    destIov.createIOVContainerIfNecessary();
    destDb.storage().lockContainer( IOVNames::container() );

    cond::MetaData  destMetadata( destDb );
    if( destMetadata.hasTag( *iT ) ){
      std::cout <<"ERROR: Tag \""<<*iT<<"\" already exists in destination database."<<std::endl;
      throw std::runtime_error("Destination account is not clean.");
    }
    std::string destIovToken=destIov.create( sourceIovType, sourceIov.iov().lastTill(),sourceIov.iov().metadata() );
    destMetadata.addMapping( *iT ,destIovToken,sourceIovType);
    destIov.setScope( cond::IOVSequence::Tag );

    boost::shared_ptr<IOVImportIterator> importIterator = destIov.importIterator();
    importIterator->setUp( sourceIov, bunchSize );

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
	std::cout <<">> Iteration #"<<iter<<": "<<imported<<" element(s)."<<std::endl;
      }
    } else {
      totalImported = importIterator->importAll();
    }    
    std::cout <<">> Tag \""<<*iT<<"\": "<<totalImported<<" element(s) exported."<<std::endl;
    transaction.commit();

    bool check = hasOptionValue("check");
    if( check ){
      std::cout<<">> Checking exported iov..."<<std::endl;
      ::sleep(1);
      destDb.transaction().start(true);
      std::string dTok = destMetadata.getToken( *iT );
      cond::IOVProxy dIov( destDb );
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
	ora::Object payload = destDb.getObject(iEl->token());
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
      destDb.transaction().commit();
      if(!ok) return -1;
      std::cout << ">> All of the tests passed. Tag \""<<*iT<<"\" successfully exported."<<std::endl; 
    }
  }

  sourceDb.transaction().commit();
  return 0;
}

int main( int argc, char** argv ){
  cond::ExportAccountUtilities utilities;
  return utilities.run(argc,argv);
}

