#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
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
  addOption<std::string>("tag","t","delete the specified tag and IOV (required)");
  addOption<bool>("withPayload","w","delete payload data associated with the specified tag (default off)");
  addOption<bool>("onlyTag","o","delete only the tag, leaving the IOV (default off)");
}

cond::DeleteIOVUtilities::~DeleteIOVUtilities(){
}

int cond::DeleteIOVUtilities::execute(){
  if( !hasOptionValue("tag") ){
    cond::Exception("Mandatory option \"tag\" not provided.");
  }
  std::string tag = getOptionValue<std::string>("tag");
  bool withPayload = hasOptionValue("withPayload");
  bool onlyTag = hasOptionValue("onlyTag");
  cond::DbSession rdbms = openDbSession( "connect" , Auth::COND_WRITER_ROLE );

  cond::DbScopedTransaction transaction( rdbms );
  transaction.start(false);
    
  cond::MetaData metadata_svc(rdbms);
  std::string token=metadata_svc.getToken(tag);
  if( token.empty() ) {
    std::cout<<"non-existing tag "<<tag<<std::endl;
    return 11;
  }
  if(!onlyTag){
    cond::IOVEditor editor(rdbms, token );
    editor.deleteEntries(withPayload);
  }
  metadata_svc.deleteEntryByTag(tag);

  transaction.commit();
  return 0;
}

int main( int argc, char** argv ){

  cond::DeleteIOVUtilities utilities;
  return utilities.run(argc,argv);
}

