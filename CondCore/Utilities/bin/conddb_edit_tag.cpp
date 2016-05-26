#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "CondCore/CondDB/interface/IOVEditor.h"
#include "CondCore/CondDB/interface/IOVProxy.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/Utilities/interface/CondDBTools.h"
#include <iostream>

#include <sstream>

namespace cond {

  class EditTagUtilities : public cond::Utilities {
    public:
      EditTagUtilities();
      ~EditTagUtilities();
      int execute();
  };
}

cond::EditTagUtilities::EditTagUtilities():Utilities("conddb_edit_tag"){
  addConnectOption("connect","c","target connection string (required)");
  addAuthenticationOptions();
  addOption<std::string>("tag","t","target tag (required)");
  addOption<std::string>("payloadClassName","C","typename of the target payload object (required for new tags)");
  addOption<std::string>("timeType","T","the IOV time type (required for new tag)");
  addOption<std::string>("synchronizationType","S","the IOV synchronization type (optional, default=any for new tags)"); 
  addOption<cond::Time_t>("endOfValidity","E","the IOV sequence end of validity (optional, default=infinity for new tags");
  addOption<std::string>("description","D","user text (required for new tags)");
  addOption<std::string>("editingNote","N","editing note (required for existing tags)");
}

cond::EditTagUtilities::~EditTagUtilities(){
}

int cond::EditTagUtilities::execute(){

  bool debug = hasDebug();
  std::string connect = getOptionValue<std::string>("connect");

  // this is mandatory
  std::string tag = getOptionValue<std::string>("tag");
  std::cout <<"# Target tag is "<<tag<<std::endl;

  persistency::ConnectionPool connPool;
  if( hasOptionValue("authPath") ){
    connPool.setAuthenticationPath( getOptionValue<std::string>( "authPath") ); 
  }
  connPool.configure();

  std::cout <<"# Connecting to source database on "<<connect<<std::endl;
  persistency::Session session = connPool.createSession( connect, true );

  persistency::IOVEditor editor;
  persistency::TransactionScope tsc( session.transaction() );
  tsc.start( false );
  bool exists = false;
  if( !session.existsDatabase() ) {
    session.createDatabase();
  } else {
    exists = session.existsIov( tag );
  }

  bool change = false;
  std::map<std::string,std::pair<std::string,std::string> > changes;

  std::string payloadType("");
  if( hasOptionValue("payloadClassName") ) payloadType =  getOptionValue<std::string>( "payloadClassName" );
  std::string description("");
  if( hasOptionValue("description") ) description =  getOptionValue<std::string>( "description" );
  std::string timeType("");
  if( hasOptionValue("timeType") ) timeType =  getOptionValue<std::string>( "timeType" );
  std::string synchronizationType("");
  if( hasOptionValue("synchronizationType") ) synchronizationType =  getOptionValue<std::string>( "synchronizationType" );
  std::string editingNote("");
  if( hasOptionValue("editingNote") ) editingNote =  getOptionValue<std::string>( "editingNote" );

  if( exists ){
    if( !payloadType.empty() ){
      std::cout <<"ERROR: can't change the payload type for an existing tag."<<std::endl;
      return -1;
    }
    if( !timeType.empty() ){
      std::cout <<"ERROR: can't change the time type for an existing tag."<<std::endl;
      return -1;
    }
    if( editingNote.empty() ){
      std::cout <<"ERROR: can't make changes to an existing tag without to provide the editing note."<<std::endl;
      return -1;
    }
    editor = session.editIov( tag );
    if( !synchronizationType.empty() ){
      cond::SynchronizationType st = cond::synchronizationTypeFromName( synchronizationType );
      changes.insert(std::make_pair("synchronizationType",std::make_pair(cond::synchronizationTypeNames(editor.synchronizationType()),synchronizationType)));
      editor.setSynchronizationType( st );
      change = true;
    }
    if( !description.empty() ){
      changes.insert(std::make_pair("description",std::make_pair(editor.description(),description)));
      editor.setDescription( description );
      change = true;
    }
  } else {
    if( payloadType.empty() ){
      std::cout <<"ERROR: can't create the new tag, since the payload type has not been provided."<<std::endl;
      return -1;
    }
    if( timeType.empty() ){
      std::cout <<"ERROR: can't create the new tag, since the time type has not been provided."<<std::endl;
      return -1;
    }
    if( description.empty() ){
      std::cout <<"ERROR: can't create the new tag, since the description has not been provided."<<std::endl;
      return -1;
    }
    if( synchronizationType.empty() ){
      std::cout<<"# Synchronization type has not been provided. Using default value = \'any\'"<<std::endl;
      synchronizationType = "any";
    }
    cond::TimeType tt = cond::time::timeTypeFromName( timeType );
    cond::SynchronizationType st = cond::synchronizationTypeFromName( synchronizationType );
    editor = session.createIov( payloadType, tag, tt, st );
    change = true;
    editor.setDescription( description );
  }

  if( hasOptionValue("endOfValidity") ){
    cond::Time_t endOfValidity = getOptionValue<cond::Time_t>("endOfValidity");
    changes.insert(std::make_pair("endOfValidity",std::make_pair(boost::lexical_cast<std::string>(editor.endOfValidity()),boost::lexical_cast<std::string>(endOfValidity))));
    editor.setEndOfValidity( endOfValidity );
    change = true;
  }
    
  if( change ) {
    if( exists ){
      bool more = false;
      std::cout <<"# Modifying existing tag.";
      auto ie = changes.find("synchronizationType");
      if( ie != changes.end() ) {
	std::cout <<" "<<ie->first<<"=\""<<ie->second.second<<"\" (was \""<<ie->second.first<<"\")";
	more = true;
      }
      ie = changes.find("endOfValidity");
      if( ie!= changes.end() ){
	if( more ) std::cout <<",";
	std::cout <<" "<<ie->first<<"=\""<<ie->second.second<<"\" (was \""<<ie->second.first<<"\")";
	more = true;	
      }
      if( more ) std::cout << std::endl;
      ie = changes.find("description");
      if( ie!= changes.end() ){
	std::cout <<"# "<<ie->first<<": \""<<ie->second.second<<"\" (was \""<<ie->second.first<<"\")"<<std::endl;
      }
    } else {
      
      std::cout <<"# Creating new tag "<<tag<<" with: payloadType=\""<<payloadType<<"\", timeType=\""<<timeType;
      std::cout <<"\", synchronizationType=\""<<synchronizationType<<"\", endOfValidity=\""<<editor.endOfValidity()<<"\""<<std::endl;
      std::cout <<"# description: \""<<description<<"\""<<std::endl;
    }
    std::string confirm("");
    std::cout <<"# Confirm changes (Y/N)? [N]";
    std::getline(std::cin, confirm) ;
    if(confirm != "Y" && confirm != "y") return 0;
    editor.flush( editingNote );
    tsc.commit();
    std::cout <<"# Changes committed. "<<std::endl;
  } else {
    std::cout <<"#No change needed to be saved."<<std::endl;
  }
  return 0;
}

int main( int argc, char** argv ){

  cond::EditTagUtilities utilities;
  return utilities.run(argc,argv);
}

