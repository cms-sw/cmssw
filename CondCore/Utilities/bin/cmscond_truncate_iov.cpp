#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/Utilities/interface/Utilities.h"
#include <iostream>

namespace cond {
  class TruncateIOVUtilities : public Utilities {
    public:
      TruncateIOVUtilities();
      ~TruncateIOVUtilities();
      int execute();
  };
}

cond::TruncateIOVUtilities::TruncateIOVUtilities():Utilities("cmscond_truncate_iov"){
  addConnectOption();
  addAuthenticationOptions();
  addDictionaryOption();
  addOption<std::string>("tag","t","remove last entry from the specified tag");
  addOption<bool>("withPayload","w","delete payload data associated with the removed entry (default off)");
}

cond::TruncateIOVUtilities::~TruncateIOVUtilities() {}

int cond::TruncateIOVUtilities::execute() {
  bool withPayload = hasOptionValue("withPayload");
  std::string tag = getOptionValue<std::string>("tag");
  cond::DbSession rdbms = openDbSession( "connect" );
  cond::DbScopedTransaction transaction( rdbms );
  cond::MetaData metadata_svc( rdbms );
  transaction.start(false);
  std::string token=metadata_svc.getToken(tag);
  if( token.empty() ) {
    std::cout<<"non-existing tag "<<tag<<std::endl;
    return 11;
  }
  
  cond::IOVEditor ioveditor(rdbms,token);
  ioveditor.truncate(withPayload);
  transaction.commit();
  return 0;
}

int main( int argc, char** argv ){

  cond::TruncateIOVUtilities utilities;
  return utilities.run(argc,argv);
}
