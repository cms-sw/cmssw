#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
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
  addOption<std::string>("scope","s","scope code to replace");
}

cond::EditIOVUtilities::~EditIOVUtilities() {}

int cond::EditIOVUtilities::execute() {
  bool replace = hasOptionValue("replace");
  if( !hasOptionValue("tag") ){
    std::cout <<"ERROR: Missing mandatory option tag."<<std::endl;
    return 1;
  }
  std::string tag = getOptionValue<std::string>("tag");
  cond::DbSession rdbms = openDbSession( "connect", Auth::COND_WRITER_ROLE );
  cond::DbScopedTransaction transaction( rdbms );
  cond::MetaData metadata_svc( rdbms );
  transaction.start(false);
  std::string token=metadata_svc.getToken(tag);
  if( token.empty() ) {
    std::cout<<"ERROR: non-existing tag "<<tag<<std::endl;
    return 1;
  }
  cond::IOVEditor ioveditor(rdbms,token);

  bool update = false;
  if( hasOptionValue("metadata") ){
    std::string metadata = getOptionValue<std::string>("metadata");
    ioveditor.editMetadata( metadata, !replace ); 
    update = true;
  }
  if( hasOptionValue("scope") ){
    cond::IOVSequence::ScopeType scope = cond::IOVSequence::Unknown;
    std::string scopeLabel = getOptionValue<std::string>("scope");
    if( scopeLabel == "Unknown" ){
      scope = cond::IOVSequence::Unknown;
    } else if( scopeLabel == "Obsolete" ){
      scope = cond::IOVSequence::Obsolete;
    } else if( scopeLabel == "Tag" ){
      scope = cond::IOVSequence::Tag;
    } else if( scopeLabel == "TagInGT" ){
      scope = cond::IOVSequence::TagInGT;
    } else if( scopeLabel == "ChildTag" ){
      scope = cond::IOVSequence::ChildTag;
    } else if( scopeLabel == "ChildTagInGT" ){
      scope = cond::IOVSequence::ChildTagInGT;
    } else {
      std::cout<<"ERROR: non-existing Scope Label "<<scopeLabel<<std::endl;
      return 1;
    }
    ioveditor.setScope( scope ); 
    update = true;
  }
  if( update ) transaction.commit();
  return 0;
}

int main( int argc, char** argv ){

  cond::EditIOVUtilities utilities;
  return utilities.run(argc,argv);
}
