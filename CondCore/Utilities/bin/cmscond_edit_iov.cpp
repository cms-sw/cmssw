#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/Utilities/interface/Utilities.h"
#include <iostream>

namespace cond {
  class EditIOVUtilities : public Utilities {
    public:
      EditIOVUtilities();
      ~EditIOVUtilities();
      int execute();
  };
}

cond::EditIOVUtilities::EditIOVUtilities():Utilities("cmscond_truncate_iov"){
  addConnectOption();
  addAuthenticationOptions();
  addOption<std::string>("tag","t","select the IOV by Tag");
  addOption<std::string>("metadata","m","metadata string to append or replace");
  addOption<bool>("replace","r","replace the existing metadata with the new (default=false)");
}

cond::EditIOVUtilities::~EditIOVUtilities() {}

int cond::EditIOVUtilities::execute() {
  bool replace = hasOptionValue("replace");
  if( !hasOptionValue("tag") ){
    std::cout <<"ERROR: Missing mandatory option tag."<<std::endl;
    return 1;
  }
  if( !hasOptionValue("metadata") ){
    std::cout <<"ERROR: Missing mandatory option metadata."<<std::endl;
    return 1;
  }
  std::string tag = getOptionValue<std::string>("tag");
  std::string metadata = getOptionValue<std::string>("metadata");
  cond::DbSession rdbms = openDbSession( "connect" );
  cond::DbScopedTransaction transaction( rdbms );
  cond::MetaData metadata_svc( rdbms );
  transaction.start(false);
  std::string token=metadata_svc.getToken(tag);
  if( token.empty() ) {
    std::cout<<"ERROR: non-existing tag "<<tag<<std::endl;
    return 1;
  }
  
  cond::IOVEditor ioveditor(rdbms,token);
  ioveditor.editMetadata( metadata, !replace ); 
  transaction.commit();
  return 0;
}

int main( int argc, char** argv ){

  cond::EditIOVUtilities utilities;
  return utilities.run(argc,argv);
}
