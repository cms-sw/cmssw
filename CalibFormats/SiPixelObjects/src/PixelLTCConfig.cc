//
// This class reads the LTC configuration file
//
//
//
 
#include "CalibFormats/SiPixelObjects/interface/PixelLTCConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include <sstream>   
#include <cassert>   
#include <stdexcept>

using namespace pos;
using namespace std;

PixelLTCConfig::PixelLTCConfig(vector< vector<string> > &tableMat):PixelConfigBase(" ", " ", " ")
{
  std::string mthn = "[PixelLTCConfig::PixelLTCConfig()]\t\t\t    " ;
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  /**
     EXTENSION_TABLE_NAME: PIXEL_LTC_PARAMETERS (VIEW: CONF_KEY_LTC_CONFIG_V)

     CONFIG_KEY 			       NOT NULL VARCHAR2(80)
     KEY_TYPE				       NOT NULL VARCHAR2(80)
     KEY_ALIAS  			       NOT NULL VARCHAR2(80)
     VERSION						VARCHAR2(40)
     KIND_OF_COND			       NOT NULL VARCHAR2(40)
     LTC_OBJ_DATA_FILE  		       NOT NULL VARCHAR2(200)
     LTC_OBJ_DATA_CLOB  		       NOT NULL CLOB
  */

  colNames.push_back("CONFIG_KEY" 	);
  colNames.push_back("KEY_TYPE"   	);
  colNames.push_back("KEY_ALIAS"  	);
  colNames.push_back("VERSION"    	);
  colNames.push_back("KIND_OF_COND"	);
  colNames.push_back("LTC_OBJ_DATA_FILE");
  colNames.push_back("LTC_OBJ_DATA_CLOB");

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
	  std::cerr << __LINE__ << "]\t" << mthn << "Couldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }
  ltcConfigStream_ << tableMat[1][colM["LTC_OBJ_DATA_CLOB"]] ;
//   cout << "[PixelTTCciConfig::PixelTTCciConfig()]\tRead: "<< endl<< ttcConfigStream_.str() << endl ;
}

PixelLTCConfig::PixelLTCConfig(std::string filename):
  PixelConfigBase(" "," "," "){

    std::string mthn = "[PixelLTCConfig::PixelLTCConfig()]\t\t\t    " ;
    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << __LINE__ << "]\t" << mthn << "Could not open: " << filename << std::endl;
	throw std::runtime_error("Failed to open file "+filename);
    }
    else {
	std::cout << __LINE__ << "]\t" << mthn << "Opened: "         << filename << std::endl;
    }

    //ltcConfigPath_ = filename;
    string line;
    while (!in.eof()) {
       getline (in,line);
       ltcConfigStream_ << line << endl;
    }

} 

void PixelLTCConfig::writeASCII(std::string dir) const {

  if (dir!="") dir+="/";
  std::string filename=dir+"LTCConfiguration.txt";
  std::ofstream out(filename.c_str());

  //std::ifstream in(ltcConfigPath_.c_str());
  //assert(in.good());

  string configstr = ltcConfigStream_.str();

  out << configstr << endl;

  out.close();

}

//=============================================================================================
void PixelLTCConfig::writeXMLHeader(pos::PixelConfigKey key, 
                                    int version, 
                                    std::string path, 
                                    std::ofstream *outstream,
                                    std::ofstream *out1stream,
                                    std::ofstream *out2stream) const
{
  std::string mthn = "[PixelLTCConfig::writeXMLHeader()]\t\t\t    " ;
  std::stringstream maskFullPath ;

  writeASCII(path) ;

  maskFullPath << path << "/Pixel_LtcParameters_" << PixelTimeFormatter::getmSecTime() << ".xml";
  std::cout << __LINE__ << "]\t" << mthn << "Writing to: " << maskFullPath.str() << std::endl ;

  outstream->open(maskFullPath.str().c_str()) ;
  
  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"                                 << std::endl ;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 		 	          << std::endl ;
  *outstream << ""                                                                                        << std::endl ; 
  *outstream << " <HEADER>"                                                                               << std::endl ; 
  *outstream << "  <TYPE>"                                                                                << std::endl ; 
  *outstream << "   <EXTENSION_TABLE_NAME>PIXEL_LTC_PARAMETERS</EXTENSION_TABLE_NAME>"                    << std::endl ; 
  *outstream << "   <NAME>LTC Configuration Parameters</NAME>"                                            << std::endl ; 
  *outstream << "  </TYPE>"                                                                               << std::endl ; 
  *outstream << "  <RUN>"                                                                                 << std::endl ; 
  *outstream << "   <RUN_TYPE>LTC Configuration Parameters</RUN_TYPE>"                                    << std::endl ; 
  *outstream << "   <RUN_NUMBER>1</RUN_NUMBER>"                                                           << std::endl ; 
  *outstream << "   <RUN_BEGIN_TIMESTAMP>" << PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>"   << std::endl ; 
  *outstream << "   <LOCATION>CERN P5</LOCATION>"                                                         << std::endl ; 
  *outstream << "  </RUN>"                                                                                << std::endl ; 
  *outstream << " </HEADER>"                                                                              << std::endl ; 
  *outstream << ""                                                                                        << std::endl ; 
  *outstream << " <DATA_SET>"                                                                             << std::endl ;
  *outstream << ""                                                                                        << std::endl ;
  *outstream << "  <VERSION>"             << version      << "</VERSION>"                                 << std::endl ;
  *outstream << "  <COMMENT_DESCRIPTION>" << getComment() << "</COMMENT_DESCRIPTION>"			  << std::endl ;
  *outstream << "  <CREATED_BY_USER>"     << getAuthor()  << "</CREATED_BY_USER>"  			  << std::endl ;
  *outstream << ""                                                                                        << std::endl ;
  *outstream << "  <PART>"                                                                                << std::endl ;
  *outstream << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"                                              << std::endl ;      
  *outstream << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"                                           << std::endl ;         
  *outstream << "  </PART>"                                                                               << std::endl ;

}

//=============================================================================================
void PixelLTCConfig::writeXML( std::ofstream *outstream,
                               std::ofstream *out1stream,
                               std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelLTCConfig::writeXML()]\t\t\t    " ;

  *outstream << " <DATA>"                                                            			  << std::endl ;
  *outstream << "  <LTC_OBJ_DATA_FILE>./LTCConfiguration.txt</LTC_OBJ_DATA_FILE>" 			  << std::endl ;
  *outstream << " </DATA>"                                                            			  << std::endl ;
}

//=============================================================================================
void PixelLTCConfig::writeXMLTrailer(std::ofstream *outstream,
                             	     std::ofstream *out1stream,
                             	     std::ofstream *out2stream ) const 
{
  std::string mthn = "[PixelLTCConfig::writeXMLTrailer()]\t\t\t    " ;
  
  *outstream << " </DATA_SET>"		 								  << std::endl ;
  *outstream << "</ROOT>"  		 								  << std::endl ;
  
  outstream->close() ;
  std::cout << __LINE__ << "]\t" << mthn << "Data written "   						  << std::endl ;

}

   
 
