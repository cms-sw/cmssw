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

namespace cond {
  class XMLUtilities : public Utilities {
    public:
      XMLUtilities();
      ~XMLUtilities();
      int execute() override;
  };

  static const size_t sinceTillColumnSize = 20;
  static const size_t sinceTillTSColumnSize = 20;

  std::string printValidityHeader( cond::TimeType timeType ){
      std::stringstream headerLine;
      headerLine<<std::setfill('-');
      switch ( timeType ){
      case runnumber:
	std::cout <<std::setw(sinceTillColumnSize)<<"Since"<<"  "<<std::setw(sinceTillColumnSize)<<"Till";
	//std::cout <<" ";
        headerLine << std::setw(sinceTillColumnSize)<<"";
        headerLine << "  "<<std::setw(sinceTillColumnSize)<<"";
        break;
      case timestamp:
        std::cout <<std::setw(sinceTillColumnSize)<<"Since"<<"  "<<std::setw(sinceTillTSColumnSize)<<"Since (time)";
        std::cout <<"  "<<std::setw(sinceTillColumnSize)<<"Till"<<"  "<<std::setw(sinceTillTSColumnSize)<<"Till (time)";
	//std::cout <<" ";
        headerLine << std::setw(sinceTillColumnSize)<<"";
        headerLine << "  "<<std::setw(sinceTillTSColumnSize)<<"";
        headerLine << "  "<<std::setw(sinceTillColumnSize)<<"";
        headerLine << "  "<<std::setw(sinceTillTSColumnSize)<<"";
        break; 
      case lumiid:
        std::cout <<std::setw(sinceTillColumnSize)<<"Since"<<"  "<<std::setw(sinceTillColumnSize)<<"Since (runn)"<<"  "<<std::setw(sinceTillColumnSize)<<"Since (lumi)";
        std::cout <<"  "<<std::setw(sinceTillColumnSize)<<"Till"<<"  "<<std::setw(sinceTillColumnSize)<<"Till (runn)"<<"  "<<std::setw(sinceTillColumnSize)<<"Till (lumi)";
        headerLine << std::setw(sinceTillColumnSize)<<"";
        headerLine << "  "<<std::setw(sinceTillColumnSize)<<"";
        headerLine << "  "<<std::setw(sinceTillColumnSize)<<"";
        headerLine << "  "<<std::setw(sinceTillColumnSize)<<"";
        headerLine << "  "<<std::setw(sinceTillColumnSize)<<"";
        headerLine << "  "<<std::setw(sinceTillColumnSize)<<"";
        break; 
      case hash:
	std::cout <<std::setw(sinceTillColumnSize)<<"Since"<<"  "<<std::setw(sinceTillColumnSize)<<"Till  ";
        headerLine << std::setw(sinceTillColumnSize)<<"";
        headerLine << "  "<<std::setw(sinceTillColumnSize)<<"";
        break;
      case userid:
	std::cout <<std::setw(sinceTillColumnSize)<<"Since"<<"  "<<std::setw(sinceTillColumnSize)<<"Till  ";
        headerLine << std::setw(sinceTillColumnSize)<<"";
        headerLine << "  "<<std::setw(sinceTillColumnSize)<<"";
        break; 
      case invalid:
	break;
      }
      return headerLine.str();
  }

  void printValidity( cond::TimeType timeType, cond::Time_t validity ){
     std::stringstream val;
     switch ( timeType ){
     case runnumber:
       std::cout <<std::setw(sinceTillColumnSize)<< validity;
       break;
     case timestamp:
       val << cond::time::to_boost(validity);
       std::cout <<std::setw(sinceTillColumnSize)<< validity;
       if( validity != cond::timeTypeSpecs[timestamp].endValue ){
	 std::cout <<"  "<<std::setw(sinceTillTSColumnSize)<< val.str().substr(0,20);
       } else {
	 std::cout <<"  "<<std::setw(sinceTillTSColumnSize)<< "Infinity";
       }
       break; 
     case lumiid:
       std::cout <<std::setw(sinceTillColumnSize)<<validity;
       std::cout <<"  "<<std::setw(sinceTillColumnSize)<< cond::time::unpack(validity).first;
       std::cout <<"  "<<std::setw(sinceTillColumnSize)<< cond::time::unpack(validity).second;
       break; 
     case hash:
       std::cout <<std::setw(sinceTillColumnSize)<< validity;
       break;
     case userid:
       std::cout <<std::setw(sinceTillColumnSize)<< validity;
       break; 
     case invalid:
       break;
     }
  }
}

cond::XMLUtilities::XMLUtilities():Utilities("cmscond_2XML"){
  addConnectOption();
  addAuthenticationOptions();
  addOption<bool>("verbose","v","verbose");
  addOption<bool>("multifile","m","one file per IOV");
  addOption<cond::Time_t>("beginTime","b","begin time (first since) (optional)");
  addOption<cond::Time_t>("endTime","e","end time (last till) (optional)");
  addOption<std::string>("tag","t","list info of the specified tag");
}

cond::XMLUtilities::~XMLUtilities(){
}

int cond::XMLUtilities::execute(){
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
    std::cout << "DEBUG: dumping " << tag << " from "<< since << " to " << till
	      << " in file " << ((multi) ? std::string(tag+"_since.xml") : std::string(tag+".xml")) 
	      << std::endl;

  unsigned int counter=0;
  std::set<std::string> const& payloadClasses=iov.payloadClasses();
  size_t maxClassSize = 13;
  for( std::set<std::string>::const_iterator iCl = payloadClasses.begin(); iCl !=payloadClasses.end(); iCl++){
    if(iCl->size()>maxClassSize) maxClassSize = iCl->size();
  }
  size_t headerSize = 43+maxClassSize;
      if( headerSize< (tag.size()+5))  headerSize = tag.size()+5;
      std::cout << std::setiosflags(std::ios_base::left);
      std::cout <<"\t"<<std::setfill('=');
      std::cout<<std::setw(headerSize)<<""<<std::endl;
      std::cout<<"\tTag: "<<tag<<std::endl;
      std::cout <<"\t"<<std::setfill('=');
      std::cout<<std::setw(headerSize)<<""<<std::endl;
      std::cout << std::setfill(' ');   
  if (verbose) { 
    std::cout << "\tStamp: " << iov.iov().comment()
	      << " Time " <<  cond::time::to_boost(iov.iov().timestamp())
	      << "; Revision " << iov.iov().revision() << std::endl;
  }
  std::cout <<"\tOID: "<<token<<std::endl;
  std::string scopeType("");
  int scope = iov.iov().scope();
  if( scope == cond::IOVSequence::Unknown ){
    scopeType = "Unknown";
  } else if( scope == cond::IOVSequence::Obsolete ){
    scopeType = "Obsolete";
  } else if ( scope == cond::IOVSequence::Tag ){
    scopeType = "Tag";
  } else if ( scope == cond::IOVSequence::TagInGT ){
    scopeType = "TagInGT";
  } else if ( scope == cond::IOVSequence::ChildTag ){
    scopeType = "ChildTag";
  } else if ( scope == cond::IOVSequence::ChildTagInGT ) {
    scopeType = "ChildTagInGT";
  }
  std::cout <<"\tScope: " <<scopeType<<std::endl;
  std::cout <<"\tDescription: " << iov.iov().metadata()<<std::endl;
  std::cout <<"\tTimeType: " << cond::timeTypeSpecs[iov.timetype()].name<<std::endl;
  if(verbose){
    std::cout <<"\t";
    std::string headerValLine = printValidityHeader( cond::timeTypeSpecs[iov.timetype()].type );
    std::cout << "  "<<std::setw(13)<<"Payload OID";
    std::cout << "  "<<std::setw(maxClassSize)<<"Payload Class";
    std::cout << std::endl;
    std::cout << std::setfill('-');
    std::cout <<"\t"<<headerValLine;
    std::cout <<"  "<< std::setw(13)<<"";
    std::cout <<"  "<< std::setw(maxClassSize)<<"";
    std::cout <<std::endl;
    std::cout << std::setfill(' ');
    std::cout << std::setiosflags(std::ios_base::right);
  }
  
  for (iov_range_iterator ioviterator=rg.begin(); ioviterator!=rg.end(); ioviterator++) {
    if (verbose) {
       std::cout <<"\t";
       printValidity( cond::timeTypeSpecs[iov.timetype()].type, ioviterator->since() );
       std::cout << "  ";
       printValidity( cond::timeTypeSpecs[iov.timetype()].type, ioviterator->till() );
       std::cout <<"  "<<std::setw(13)<<ioviterator->token();
       std::cout <<"  "<<std::setw(maxClassSize)<<session.classNameForItem(ioviterator->token());
       std::cout<<std::endl;
    }
    ora::Object obj = session.getObject(ioviterator->token());
    std::ostringstream ss; ss << tag << '_' << ioviterator->since(); 
    if (multi) xml = TFile::Open(std::string(ss.str()+".xml").c_str(),"recreate");
    xml->WriteObjectAny(obj.address(),obj.typeName().c_str(), ss.str().c_str());
    ++counter;
    if (multi)  xml->Close();
    obj.destruct();
  }
  std::cout <<std::endl;
  std::cout <<"\tPayloadClasses: "<<std::endl;
  std::cout <<"\t"<<std::setfill('-');
  std::cout<<std::setw(maxClassSize)<<""<<std::endl;
  std::cout << std::setfill(' ');
  std::cout << std::setiosflags(std::ios_base::left);
  for( std::set<std::string>::const_iterator iCl = payloadClasses.begin(); iCl !=payloadClasses.end(); iCl++){
    std::cout <<"\t"<<*iCl<<std::endl;
  }
  transaction.commit();

  if (!multi) xml->Close();
  if (verbose)  std::cout<< std::endl << "\tTotal # of payload objects: "<<counter<<std::endl;

  return 0;
}

  
int main( int argc, char** argv ){

  cond::XMLUtilities utilities;
  return utilities.run(argc,argv);
}

