#include "CondCore/Utilities/interface/Utilities.h"

#include "CondCore/ORA/interface/Object.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"
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
  
  cond::DbSession session = openDbSession( "connect", Auth::COND_READER_ROLE, true );

  std::string tag = getOptionValue<std::string>("tag");
  cond::MetaData metadata_svc(session);
  std::string token;
  cond::DbScopedTransaction transaction(session);
  transaction.start(true);
  if( !metadata_svc.hasTag( tag ) ){
    std::cout <<"Tag \""<<tag<<"\" has not been found."<<std::endl;
    transaction.commit();
    return 0; 
  }
  token=metadata_svc.getToken(tag);
  cond::Time_t since = std::numeric_limits<cond::Time_t>::min();
  if( hasOptionValue("beginTime" )) since = getOptionValue<cond::Time_t>("beginTime");
  cond::Time_t till = std::numeric_limits<cond::Time_t>::max();
  if( hasOptionValue("endTime" )) till = getOptionValue<cond::Time_t>("endTime");

  bool verbose = hasOptionValue("verbose");
  bool multi = hasOptionValue("multifile");
  TFile * xml =0;
  if (!multi) xml = TFile::Open(std::string(tag+".xml").c_str(),"recreate");


  cond::IOVProxy iov( session, token );

  since = std::max(since, cond::timeTypeSpecs[iov.timetype()].beginValue);
  till  = std::min(till,  cond::timeTypeSpecs[iov.timetype()].endValue);
  cond::IOVRange rg = iov.range(since,till);
 
  if (verbose)
    std::cout << "dumping " << tag << " from "<< since << " to " << till
	      << " in file " << ((multi) ? std::string(tag+"_since.xml") : std::string(tag+".xml")) 
	      << std::endl;

  unsigned int counter=0;
  std::set<std::string> const&  payloadClasses=iov.payloadClasses();
  std::cout<<"Tag "<<tag;
  if (verbose) { 
    std::cout << "\nStamp: " << iov.iov().comment()
	      << "; time " <<  cond::time::to_boost(iov.iov().timestamp())
	      << "; revision " << iov.iov().revision();
    std::cout <<"\nTimeType " << cond::timeTypeSpecs[iov.timetype()].name
	      <<"since \t till \t payloadToken"<<std::endl;
  }
  for (iov_range_iterator ioviterator=rg.begin(); ioviterator!=rg.end(); ioviterator++) {
    if (verbose)
      std::cout<<ioviterator->since() << " \t "<<ioviterator->till() <<" \t "
	       <<ioviterator->token()<<std::endl;
    
    ora::Object obj = session.getObject(ioviterator->token());
    std::ostringstream ss; ss << tag << '_' << ioviterator->since(); 
    if (multi) xml = TFile::Open(std::string(ss.str()+".xml").c_str(),"recreate");
    xml->WriteObjectAny(obj.address(),obj.typeName().c_str(), ss.str().c_str());
    ++counter;
    if (multi)  xml->Close();
    obj.destruct();
  }
  std::cout <<" PayloadClasses: "<<std::endl;
  for( std::set<std::string>::const_iterator iCl = payloadClasses.begin(); iCl !=payloadClasses.end(); iCl++){
    std::cout <<*iCl<<std::endl;
  }
  transaction.commit();

  if (!multi) xml->Close();
  if (verbose)  std::cout<<"Total # of payload objects: "<<counter<<std::endl;

  return 0;
}

  
int main( int argc, char** argv ){

  cond::XMLUtilities utilities;
  return utilities.run(argc,argv);
}

