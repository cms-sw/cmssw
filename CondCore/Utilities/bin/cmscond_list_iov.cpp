#include "CondCore/Utilities/interface/Utilities.h"

#include "CondCore/ORA/interface/Object.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"

#include "CondCore/IOVService/interface/IOVProxy.h"

#include <iterator>
#include <iostream>
#include <sstream>
#include "TFile.h"
#include "Cintex/Cintex.h"

namespace cond {
  class ListIOVUtilities : public Utilities {
    public:
      ListIOVUtilities();
      ~ListIOVUtilities();
      int execute();
  };
}

cond::ListIOVUtilities::ListIOVUtilities():Utilities("cmscond_list_iov"){
  addConnectOption();
  addAuthenticationOptions();
  addOption<bool>("verbose","v","verbose");
  addOption<bool>("all","a","list all tags(default mode)");
  addOption<bool>("summary","s","print also the summary for each payload");
  addOption<std::string>("tag","t","list info of the specified tag");

  ROOT::Cintex::Cintex::Enable();
  
}

cond::ListIOVUtilities::~ListIOVUtilities(){
}

int cond::ListIOVUtilities::execute(){
  initializePluginManager();
  
  bool listAll = hasOptionValue("all");
  cond::DbSession session = openDbSession( "connect", true );
  if( listAll ){
    cond::MetaData metadata_svc(session);
    std::vector<std::string> alltags;
    cond::DbScopedTransaction transaction(session);
    transaction.start(true);
    metadata_svc.listAllTags(alltags);
    transaction.commit();
    std::copy (alltags.begin(),
               alltags.end(),
               std::ostream_iterator<std::string>(std::cout,"\n")
	       );
  }else{
    std::string tag = getOptionValue<std::string>("tag");
    cond::MetaData metadata_svc(session);
    std::string token;
    cond::DbScopedTransaction transaction(session);
    transaction.start(true);
    token=metadata_svc.getToken(tag);
    transaction.commit();
    {
      bool verbose = hasOptionValue("verbose");
      bool details = hasOptionValue("summary");
      TFile * xml=0;
      if (details) {
	xml =  TFile::Open(std::string(tag+".xml").c_str(),"recreate");
      } 
      cond::IOVProxy iov( session, token, !details, details);
      unsigned int counter=0;
      std::string payloadContainer=iov.payloadContainerName();
      std::cout<<"Tag "<<tag;
      if (verbose) std::cout << "\nStamp: " << iov.iov().comment()
                             << "; time " <<  cond::time::to_boost(iov.iov().timestamp())
                             << "; revision " << iov.iov().revision();
      std::cout <<"\nTimeType " << cond::timeTypeSpecs[iov.timetype()].name
                <<"\nPayloadContainerName "<<payloadContainer<<"\n"
                <<"since \t till \t payloadToken"<<std::endl;
      for (cond::IOVProxy::const_iterator ioviterator=iov.begin(); ioviterator!=iov.end(); ioviterator++) {
        std::cout<<ioviterator->since() << " \t "<<ioviterator->till() <<" \t "<<ioviterator->wrapperToken();
        if (details) {
	  ora::Object obj = session.getObject(ioviterator->wrapperToken());
	  std::ostringstream ss; ss << tag << '_' << ioviterator->since(); 
	  xml->WriteObjectAny(obj.address(),obj.typeName().c_str(), ss.str().c_str());
	  obj.destruct();
        }
        std::cout<<std::endl;
        ++counter;
      }
      if (xml) xml->Close();
      std::cout<<"Total # of payload objects: "<<counter<<std::endl;
    }
  }
  return 0;
}

  
int main( int argc, char** argv ){

  cond::ListIOVUtilities utilities;
  return utilities.run(argc,argv);
}

