#include "CondCore/Utilities/interface/Utilities.h"

#include "CondCore/ORA/interface/Object.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"

#include "CondCore/IOVService/interface/IOVProxy.h"

#include <boost/program_options.hpp>
#include <iterator>
#include <iostream>
#include <sstream>
#include "TFile.h"
#include "Cintex/Cintex.h"

namespace cond {
  class XMLUtilities : public Utilities {
    public:
      XMLUtilities();
      ~XMLUtilities();
      int execute();
  };
}

cond::XMLUtilities::XMLUtilities():Utilities("cmscond_2XML"){
  addConnectOption();
  addAuthenticationOptions();
  addOption<bool>("verbose","v","verbose");
  addOption<bool>("multifile","m","one file per IOV");
  addOption<cond::Time_t>("beginTime","b","begin time (first since) (optional)");
  addOption<cond::Time_t>("endTime","e","end time (last till) (optional)");
  addOption<std::string>("tag","t","list info of the specified tag");

  ROOT::Cintex::Cintex::Enable();
}

cond::XMLUtilities::~XMLUtilities(){
}

int cond::XMLUtilities::execute(){
  initializePluginManager();
  
  cond::DbSession session = openDbSession( "connect", true );

  std::string tag = getOptionValue<std::string>("tag");
  cond::MetaData metadata_svc(session);
  std::string token;
  cond::DbScopedTransaction transaction(session);
  transaction.start(true);
  token=metadata_svc.getToken(tag);
  transaction.commit();
  cond::Time_t since = std::numeric_limits<cond::Time_t>::min();
  if( hasOptionValue("beginTime" )) since = getOptionValue<cond::Time_t>("beginTime");
  cond::Time_t till = std::numeric_limits<cond::Time_t>::max();
  if( hasOptionValue("endTime" )) till = getOptionValue<cond::Time_t>("endTime");

  bool verbose = hasOptionValue("verbose");
  bool multi = hasOptionValue("multifile");
  TFile * xml =0;
  if (!multi) xml = TFile::Open(std::string(tag+".xml").c_str(),"recreate");


  cond::IOVProxy iov( session, token, false, true);

  since = std::max(since, cond::timeTypeSpecs[iov.timetype()].beginValue);
  till  = std::min(till,  cond::timeTypeSpecs[iov.timetype()].endValue);
  iov.setRange(since,till);
 
if (verbose)
  std::cout << "dumping " << tag << " from "<< since << " to " << till
	    << " in file " << ((multi) ? std::string(tag+"_since.xml") : std::string(tag+".xml")) 
	    << std::endl;
	

  unsigned int counter=0;
  std::string payloadContainer=iov.payloadContainerName();
  std::cout<<"Tag "<<tag;
  if (verbose) { 
    std::cout << "\nStamp: " << iov.iov().comment()
	      << "; time " <<  cond::time::to_boost(iov.iov().timestamp())
	      << "; revision " << iov.iov().revision();
    std::cout <<"\nTimeType " << cond::timeTypeSpecs[iov.timetype()].name
	      <<"\nPayloadContainerName "<<payloadContainer<<"\n"
	      <<"since \t till \t payloadToken"<<std::endl;
  }
  for (cond::IOVProxy::const_iterator ioviterator=iov.begin(); ioviterator!=iov.end(); ioviterator++) {
    if (verbose)
      std::cout<<ioviterator->since() << " \t "<<ioviterator->till() <<" \t "
	       <<ioviterator->wrapperToken()<<std::endl;
    
    ora::Object obj = session.getObject(ioviterator->wrapperToken());
    std::ostringstream ss; ss << tag << '_' << ioviterator->since(); 
    if (multi) xml = TFile::Open(std::string(ss.str()+".xml").c_str(),"recreate");
    xml->WriteObjectAny(obj.address(),obj.typeName().c_str(), ss.str().c_str());
    ++counter;
    if (multi)  xml->Close();
    obj.destruct();
  }
  if (!multi) xml->Close();
  if (verbose)  std::cout<<"Total # of payload objects: "<<counter<<std::endl;

  return 0;
}

  
int main( int argc, char** argv ){

  cond::XMLUtilities utilities;
  return utilities.run(argc,argv);
}

