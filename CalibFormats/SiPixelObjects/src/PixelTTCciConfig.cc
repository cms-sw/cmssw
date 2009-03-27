//
// This class reads the TTC configuration file
//
//
//
 
#include "CalibFormats/SiPixelObjects/interface/PixelTTCciConfig.h"
#include <cassert>
  
using namespace pos;
using namespace std;
 
PixelTTCciConfig::PixelTTCciConfig(vector< vector<string> > &tableMat):PixelConfigBase(" ", " ", " ")
{
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  /**
     View's name: CONF_KEY_TTC_CONFIG_MV

     CONFIG_KEY_ID                             NOT NULL NUMBER(38)
     CONFG_KEY                                 NOT NULL VARCHAR2(80)
     VERSION                                            VARCHAR2(40)
     KIND_OF_COND                              NOT NULL VARCHAR2(40)
     RUN_TYPE                                           VARCHAR2(40)
     RUN_NUMBER                                         NUMBER(38)
     TTC_OBJ_DATA_FILE                         NOT NULL VARCHAR2(200)
     TTC_OBJ_DATA_CLOB                         NOT NULL CLOB
  */

  colNames.push_back("CONFIG_KEY_ID"    );
  colNames.push_back("CONFG_KEY"        );
  colNames.push_back("VERSION"          );
  colNames.push_back("KIND_OF_COND"     );
  colNames.push_back("RUN_TYPE"         );
  colNames.push_back("RUN_NUMBER"       );
  colNames.push_back("TTC_OBJ_DATA_FILE");
  colNames.push_back("TTC_OBJ_DATA_CLOB");
  
  for(unsigned int c = 0 ; c < tableMat[0].size() ; c++)
    {
      for(unsigned int n=0; n<colNames.size(); n++)
	{
	  if(tableMat[0][c] == colNames[n])
	    {
	      colM[colNames[n]] = c;
	      break;
	    }
	}
    }//end for
  for(unsigned int n=0; n<colNames.size(); n++)
    {
      if(colM.find(colNames[n]) == colM.end())
	{
	  std::cerr << "[PixelTTCciConfig::PixelTTCciConfig()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }
  ttcConfigStream_ << tableMat[1][colM["TTC_OBJ_DATA_CLOB"]] ;
//   cout << "[PixelTTCciConfig::PixelTTCciConfig()]\tRead: "<< endl<< ttcConfigStream_.str() << endl ;
}

PixelTTCciConfig::PixelTTCciConfig(std::string filename):
  PixelConfigBase(" "," "," "){

    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << "Could not open:"<<filename<<std::endl;
	assert(0);
    }
    else {
	std::cout << "Opened:"<<filename<<std::endl;
    }

    //ttcConfigPath_ = filename;
    string line;
    while (!in.eof()) {
       getline (in,line);
       ttcConfigStream_ << line << endl;
    }

} 

void PixelTTCciConfig::writeASCII(std::string dir) const {

  
  if (dir!="") dir+="/";
  std::string filename=dir+"TTCciConfiguration.txt";
  std::ofstream out(filename.c_str());

  //std::ifstream in(ttcConfigPath_.c_str());
  //assert(in.good());

  string configstr = ttcConfigStream_.str();

  out << configstr << endl;

  out.close();

}

 

