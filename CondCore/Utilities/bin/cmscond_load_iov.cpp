#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/DBCommon/interface/IOVInfo.h"
#include "CondCore/Utilities/interface/Utilities.h"


//#include <boost/program_options.hpp>
#include <iterator>
#include <iostream>
#include <fstream>

namespace{
  struct Parser {
      typedef std::pair<cond::Time_t, std::string> Item;
      std::string tag;
      cond::TimeType timetype;
      std::string contName;
      std::vector<Item> values;
      cond::Time_t lastTill;

      void parseInputFile(std::fstream& file){
        std::string dummy;
        std::string timename;
        cond::Time_t since, till;
        std::string token;

        file >> dummy >> tag;
        file >> dummy >> timename;
        timetype = cond::findSpecs(timename).type;
        file >> dummy >> contName;
        char buff[1024];
        file.getline(buff,1024);
        file.getline(buff,1024);
        char p;
        while(file) {
          file.get(p); if (p=='T') break;
          file.putback(p);
          file >> since >> till >> token;  file.getline(buff,1024);
          values.push_back(Item(since,token));
        }
        lastTill = till;
      }

  };

}

namespace cond {
  class LoadIOVUtilities : public Utilities {
    public:
      LoadIOVUtilities();
      ~LoadIOVUtilities();
      int execute();
  };
}

cond::LoadIOVUtilities::LoadIOVUtilities():Utilities("cmscond_load_iov","inputFile"){
  addConnectOption();
  addAuthenticationOptions();
}

cond::LoadIOVUtilities::~LoadIOVUtilities(){
}

int cond::LoadIOVUtilities::execute(){

  std::string inputFileName = getOptionValue<std::string>("inputFile");
  bool debug=hasDebug();

  std::fstream inputFile;
  inputFile.open(inputFileName.c_str(), std::fstream::in);
  Parser parser;
  parser.parseInputFile(inputFile);
  inputFile.close();
  
  std::string iovtoken("");

  cond::DbSession session = openDbSession("connect");
  cond::DbScopedTransaction transaction(session);
  transaction.start(false);

  //session.initializeMapping( cond::IOVNames::iovMappingVersion(),
  //                           cond::IOVNames::iovMappingXML() );
   
  cond::IOVEditor editor(session);
  editor.create(parser.timetype,parser.lastTill);
  editor.bulkAppend(parser.values);
  editor.stamp(cond::userInfo(),false);
  iovtoken=editor.token();
  cond::MetaData metadata( session );
  metadata.addMapping(parser.tag,iovtoken,parser.timetype);
  transaction.commit();
  if(debug){
    std::cout<<"source iov token "<<iovtoken<<std::endl;
    std::cout<<"source iov timetype "<<parser.timetype<<std::endl;
  }
  return 0;
}

int main( int argc, char** argv ){

  cond::LoadIOVUtilities utilities;
  return utilities.run(argc,argv);
}

