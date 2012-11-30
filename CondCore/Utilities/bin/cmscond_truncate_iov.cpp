#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"
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
  addOption<std::string>("tag","t","the concerned tag. Mandatory.");
  addOption<size_t>("numberOfElements","n","number of IOV elements to truncate (default 1)");
  addOption<cond::Time_t>("lastKeptSince","s","last kept since in the sequence after truncation");
}

cond::TruncateIOVUtilities::~TruncateIOVUtilities() {}

int cond::TruncateIOVUtilities::execute() {
  size_t nelem = 1;
  if ( hasOptionValue("numberOfElements") ){
    nelem = getOptionValue<size_t>("numberOfElements");
  }
  cond::Time_t fkSince = cond::invalidTime;
  if( hasOptionValue("firstKeptSince") ){
    fkSince = getOptionValue<cond::Time_t>("firstKeptSince");
  }
  std::string tag = getOptionValue<std::string>("tag");
  cond::DbSession rdbms = openDbSession( "connect", Auth::COND_ADMIN_ROLE );
  cond::DbScopedTransaction transaction( rdbms );
  cond::MetaData metadata_svc( rdbms );
  transaction.start(false);
  std::string token=metadata_svc.getToken(tag);
  if( token.empty() ) {
    std::cout<<"non-existing tag "<<tag<<std::endl;
    return 11;
  }
  
  cond::IOVEditor ioveditor(rdbms,token);
  if( fkSince != cond::invalidTime ){
    std::cout <<"Searching for since time="<<fkSince<<" in the IOV sequence."<<std::endl;
    nelem = 0;
    IOVProxy iov = ioveditor.proxy();
    while( iov.iov().iovs()[iov.size()-1-nelem].sinceTime() > fkSince ){
      nelem++;
    }
  }
  std::cout <<"Truncating "<<nelem<<" element(s) in the IOV sequence."<<std::endl;
  for(size_t i=0;i<nelem;i++){
    // truncate removing the payload is no longer supported...
    ioveditor.truncate(false);
  }
  transaction.commit();
  if( nelem )
    std::cout <<"Update completed. New iov size="<<ioveditor.proxy().size()<<std::endl;
  return 0;
}

int main( int argc, char** argv ){

  cond::TruncateIOVUtilities utilities;
  return utilities.run(argc,argv);
}
