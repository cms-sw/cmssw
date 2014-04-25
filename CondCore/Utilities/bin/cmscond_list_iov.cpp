#include "CondCore/Utilities/interface/Utilities.h"

#include "CondCore/ORA/interface/Object.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"

#include "CondCore/IOVService/interface/IOVProxy.h"

#include <iterator>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include "TFile.h"
#include "Cintex/Cintex.h"

namespace cond {
  class ListIOVUtilities : public Utilities {
    public:
      ListIOVUtilities();
      ~ListIOVUtilities();
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

cond::ListIOVUtilities::ListIOVUtilities():Utilities("cmscond_list_iov"){
  addConnectOption();
  addAuthenticationOptions();
  addOption<bool>("verbose","v","verbose");
  addOption<bool>("all","a","list all tags(default mode)");
  addOption<bool>("summary","s","print also the summary for each payload");
  addOption<bool>("outputfile","o","dump iov content for loading");
  addOption<std::string>("tag","t","list info of the specified tag");

  ROOT::Cintex::Cintex::Enable();
  
}

cond::ListIOVUtilities::~ListIOVUtilities(){
}

int cond::ListIOVUtilities::execute(){
  bool listAll = hasOptionValue("all");
  bool dump = hasOptionValue("outputfile");
  std::auto_ptr<std::ofstream> outFile;
  cond::DbSession session = openDbSession( "connect", Auth::COND_READER_ROLE, true );
  cond::DbScopedTransaction transaction(session);
  transaction.start(true);
  if( listAll ){
    cond::MetaData metadata_svc(session);
    std::vector<std::string> alltags;
    metadata_svc.listAllTags(alltags);
    std::copy (alltags.begin(),
               alltags.end(),
               std::ostream_iterator<std::string>(std::cout,"\n")
	       );
  }
  else {
    std::string tag = getOptionValue<std::string>("tag");
    std::string outFileName(tag);
    outFileName += ".dump";
    cond::MetaData metadata_svc(session);
    std::string token;
    token=metadata_svc.getToken(tag);
    {
      bool verbose = hasOptionValue("verbose");
      bool details = hasOptionValue("summary");
      TFile * xml=0;
      if (details) {
	xml =  TFile::Open(std::string(tag+".xml").c_str(),"recreate");
      } 
      cond::IOVProxy iov( session, token );
      unsigned int counter=0;
      const std::set<std::string>& payloadClasses=iov.payloadClasses();
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
                  << "  Time: " <<  cond::time::to_boost(iov.iov().timestamp())
                  << ";  Revision: " << iov.iov().revision()<<std::endl;
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
      if( dump ) {
        outFile.reset( new std::ofstream( outFileName.c_str() ));
        *outFile << "Tag: "<<tag<<std::endl;
        *outFile << "TimeType: " << cond::timeTypeSpecs[iov.timetype()].name<<std::endl;
        *outFile << "Elements:"<<std::endl;
      }
      for (cond::IOVProxy::const_iterator ioviterator=iov.begin(); ioviterator!=iov.end(); ioviterator++) {
        std::cout <<"\t";
        printValidity( cond::timeTypeSpecs[iov.timetype()].type, ioviterator->since() );
	std::cout << "  ";
        printValidity( cond::timeTypeSpecs[iov.timetype()].type, ioviterator->till() );
        std::cout <<"  "<<std::setw(13)<<ioviterator->token();
        std::cout <<"  "<<std::setw(maxClassSize)<<session.classNameForItem(ioviterator->token());
        std::cout<<std::endl;
        if (details) {
	  ora::Object obj = session.getObject(ioviterator->token());
	  std::ostringstream ss; ss << tag << '_' << ioviterator->since(); 
	  xml->WriteObjectAny(obj.address(),obj.typeName().c_str(), ss.str().c_str());
	  obj.destruct();
        }
        if( dump ) {
	  *outFile <<ioviterator->since() <<" "<<ioviterator->till()<<" "<<ioviterator->token()<<std::endl;
        }
        ++counter;
      }
      std::cout<<std::endl;
      if (verbose) {
        std::cout <<"\tPayload Classes "<<std::endl;
        std::cout <<"\t"<<std::setfill('-');
        std::cout<<std::setw(maxClassSize)<<""<<std::endl;
        std::cout << std::setfill(' ');
        std::cout << std::setiosflags(std::ios_base::left);
        for( std::set<std::string>::const_iterator iCl = payloadClasses.begin(); iCl !=payloadClasses.end(); iCl++){
	  std::cout <<"\t"<<*iCl<<std::endl;
        }
        std::cout << std::endl;
     }
     if (xml) xml->Close();
     if( dump ) {
       outFile->flush();
       outFile->close();
     }
     std::cout<<"\tTotal # of payload objects: "<<counter<<std::endl;
    }
  }
  transaction.commit();
  return 0;
}

  
int main( int argc, char** argv ){

  cond::ListIOVUtilities utilities;
  return utilities.run(argc,argv);
}

