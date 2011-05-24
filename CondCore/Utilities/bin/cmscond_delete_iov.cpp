#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/Utilities/interface/Utilities.h"
#include <iostream>

namespace cond {
  class DeleteIOVUtilities : public Utilities {
    public:
      DeleteIOVUtilities();
      ~DeleteIOVUtilities();
      int execute();
  };
}

cond::DeleteIOVUtilities::DeleteIOVUtilities():Utilities("cmscond_delete_iov"){
  addConnectOption();
  addAuthenticationOptions();
  addDictionaryOption();
  addOption<bool>("all","a","delete all tags");
  addOption<std::string>("tag","t","delete the specified tag and IOV");
  addOption<bool>("withPayload","w","delete payload data associated with the specified tag (default off)");
}

cond::DeleteIOVUtilities::~DeleteIOVUtilities(){
}

int cond::DeleteIOVUtilities::execute(){
  bool deleteAll = hasOptionValue("all");
  bool withPayload = hasOptionValue("withPayload");
  cond::DbSession rdbms = openDbSession( "connect" );

  cond::DbScopedTransaction transaction( rdbms );
  transaction.start(false);
  if( deleteAll ){
    // irrelevant which tymestamp
    cond::IOVService iovservice(rdbms);
    transaction.start(false);
    iovservice.deleteAll(withPayload);
    cond::MetaData metadata_svc(rdbms);
    metadata_svc.deleteAllEntries();
  }else{
    std::string tag = getOptionValue<std::string>("tag");
    
    cond::MetaData metadata_svc(rdbms);
    std::string token=metadata_svc.getToken(tag);
    if( token.empty() ) {
      std::cout<<"non-existing tag "<<tag<<std::endl;
      return 11;
    }
    cond::IOVService iovservice(rdbms);
    std::auto_ptr<cond::IOVEditor> ioveditor(iovservice.newIOVEditor(token));
    ioveditor->deleteEntries(withPayload);
    metadata_svc.deleteEntryByTag(tag);
  }
  transaction.commit();
  return 0;
}

int main( int argc, char** argv ){

  cond::DeleteIOVUtilities utilities;
  return utilities.run(argc,argv);
}

