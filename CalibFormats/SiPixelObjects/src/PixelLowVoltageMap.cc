//
// Implementation of the detector configuration
//
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelLowVoltageMap.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <ios>
#include <assert.h>
#include <stdio.h>
#include <stdexcept>

using namespace std;
using namespace pos;


PixelLowVoltageMap::PixelLowVoltageMap(std::vector< std::vector < std::string> > &tableMat):PixelConfigBase("","","")
{
  std::string mthn = "[PixelLowVoltageMap::PixelLowVoltageMap()] " ;
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
/*
  EXTENSION_TABLE_NAME: XDAQ_LOW_VOLTAGE_MAP  (VIEW: CONF_KEY_XDAQ_LOW_VOLTAGE_V) 
  
  CONFIG_KEY				    NOT NULL VARCHAR2(80)
  KEY_TYPE				    NOT NULL VARCHAR2(80)
  KEY_ALIAS				    NOT NULL VARCHAR2(80)
  VERSION					     VARCHAR2(40)
  KIND_OF_COND  			    NOT NULL VARCHAR2(40)
  PANEL_NAME				    NOT NULL VARCHAR2(200)
  DATAPOINT				    NOT NULL VARCHAR2(200)
  LV_DIGITAL				    NOT NULL VARCHAR2(200)
  LV_ANALOG				    NOT NULL VARCHAR2(200)
  
*/

  colNames.push_back("CONFIG_KEY"  );
  colNames.push_back("KEY_TYPE"    );
  colNames.push_back("KEY_ALIAS"   );
  colNames.push_back("VERSION"     );
  colNames.push_back("KIND_OF_COND");
  colNames.push_back("PANEL_NAME"  );
  colNames.push_back("DATAPOINT"   );
  colNames.push_back("LV_DIGITAL"  );
  colNames.push_back("LV_ANALOG"   );
/*
  colNames.push_back("CONFIG_KEY_ID"	);
  colNames.push_back("CONFG_KEY"	);
  colNames.push_back("VERSION"	);
  colNames.push_back("KIND_OF_COND"	);
  colNames.push_back("RUN_TYPE"	);
  colNames.push_back("RUN_NUMBER"	);
  colNames.push_back("PANEL_NAME"	);
  colNames.push_back("DATAPOINT"	);
  colNames.push_back("LV_DIGITAL"	);
  colNames.push_back("LV_ANALOG"	);
*/
  
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
	  std::cerr << mthn << "Couldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }
  
  std::string modulename   ;
  std::string dpNameBase   ;
  std::string ianaChannel  ;
  std::string idigiChannel ;
  for(unsigned int r = 1 ; r < tableMat.size() ; r++)    //Goes to every row of the Matrix
    {
      modulename  = tableMat[r][colM["PANEL_NAME"]] ;
      dpNameBase  = tableMat[r][colM["DATAPOINT"]]  ;
      ianaChannel = tableMat[r][colM["LV_ANALOG"]] ; 
      idigiChannel= tableMat[r][colM["LV_DIGITAL"]]  ; 
      PixelModuleName module(modulename);
      pair<string, string> channels(ianaChannel,idigiChannel);
      pair<string, pair<string,string> > dpName(dpNameBase,channels);
      dpNameMap_[module]=dpName;
    }
}//end constructor

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PixelLowVoltageMap::PixelLowVoltageMap(std::string filename):
  PixelConfigBase("","",""){
  
  std::string mthn = "[PixelLowVoltageMap::PixelLowVoltageMap()]\t\t\t    " ;
  
  if (filename[filename.size()-1]=='t'){
    
    std::ifstream in(filename.c_str());
    
    if (!in.good()){
      std::cout << __LINE__ << "]\t" << mthn << "Could not open: " << filename << std::endl;
      throw std::runtime_error("Failed to open file "+filename);
    }
    else {
      std::cout << __LINE__ << "]\t" << mthn << "Opened: "         << filename << std::endl;
    }
    
    if (in.eof()){
      std::cout << __LINE__ << "]\t" << mthn << "eof before reading anything!" << std::endl;
      throw std::runtime_error("Failure when reading file; file seems to be empty: "+filename);
    }

    
    dpNameMap_.clear();
    
    std::string modulename;
    std::string dpNameBase;
    std::string ianaChannel;
    std::string idigiChannel;
    
    in >> modulename >> dpNameBase >> ianaChannel >> idigiChannel;
    
    while (!in.eof()){
      cout << __LINE__ << "]\t" << mthn << "Read modulename: " << modulename << endl;
      PixelModuleName module(modulename);
      pair<string, string> channels(ianaChannel,idigiChannel);
      pair<string, pair<string,string> > dpName(dpNameBase,channels);
      dpNameMap_[module]=dpName;
      in >> modulename >> dpNameBase >> ianaChannel >> idigiChannel;
    }
    
  }
  else{
    assert(0);
  }
}

std::string PixelLowVoltageMap::dpNameIana(const PixelModuleName& module) const{

  std::string mthn = "[PixelLowVoltageMap::dpNameIana()]\t\t\t    " ;
  std::map<PixelModuleName, pair< string, pair<string, string> > >::const_iterator i=
    dpNameMap_.find(module);
  
  if (i==dpNameMap_.end()) {
    cout << __LINE__ << "]\t" << mthn << "Could not find module: " << module << endl;
  }
  
  return i->second.first+"/"+i->second.second.first;

}

std::string PixelLowVoltageMap::dpNameIdigi(const PixelModuleName& module) const{

  std::string mthn = "[PixelLowVoltageMap::dpNameIdigi()]\t\t\t    " ;
  std::map<PixelModuleName, pair< string, pair<string, string> > >::const_iterator i=
    dpNameMap_.find(module);
  
  if (i==dpNameMap_.end()) {
    cout << __LINE__ << "]\t" << mthn << "Could not find module: " << module << endl;
  }

  return i->second.first+"/"+i->second.second.second;

}


void PixelLowVoltageMap::writeASCII(std::string dir) const {

  std::string mthn = "[PixelLowVoltageMap::writeASCII()]\t\t\t    " ;
  if (dir!="") dir+="/";
  std::string filename=dir+"lowvoltagemap.dat";

  std::ofstream out(filename.c_str(), std::ios_base::out) ;
  if(!out) {
    std::cout << __LINE__ << "]\t" << mthn << "Could not open file " << filename << " for write" << std::endl ;
    exit(1);
  }
  std::map<PixelModuleName, pair< string, pair<string, string> > >::const_iterator imodule=
    dpNameMap_.begin();

  for (;imodule!=dpNameMap_.end();++imodule) {
    out <<       imodule->first
        << " "<< imodule->second.first 
	<< " "<< imodule->second.second.first
	<< " "<< imodule->second.second.second 
	<<endl;
  }

  out.close();

}

//=============================================================================================
void PixelLowVoltageMap::writeXMLHeader(pos::PixelConfigKey key, 
                                      	int version, 
                                      	std::string path, 
                                      	std::ofstream *outstream,
                                      	std::ofstream *out1stream,
                                      	std::ofstream *out2stream) const
{
  std::string mthn = "[PixelLowVoltageMap::writeXMLHeader()]\t\t\t    " ;
  std::stringstream maskFullPath ;

  maskFullPath << path << "/XDAQLowVoltageMap_Test_" << PixelTimeFormatter::getmSecTime() << ".xml";
  std::cout << __LINE__ << "]\t" << mthn << "Writing to: " << maskFullPath.str() << std::endl ;

  outstream->open(maskFullPath.str().c_str()) ;
  
  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"                                 << std::endl ;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 		 	          << std::endl ;
  *outstream << ""                                                                                        << std::endl ; 
  *outstream << " <HEADER>"                                                                               << std::endl ; 
  *outstream << "  <TYPE>"                                                                                << std::endl ; 
  *outstream << "   <EXTENSION_TABLE_NAME>XDAQ_LOW_VOLTAGE_MAP</EXTENSION_TABLE_NAME>"                    << std::endl ; 
  *outstream << "   <NAME>XDAQ Low Voltage Map</NAME>"                                                    << std::endl ; 
  *outstream << "  </TYPE>"                                                                               << std::endl ; 
  *outstream << "  <RUN>"                                                                                 << std::endl ; 
  *outstream << "   <RUN_TYPE>XDAQ Low Voltage Map</RUN_TYPE>"                                            << std::endl ; 
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
  *outstream << " "                                                                                       << std::endl ;

}

//=============================================================================================
void PixelLowVoltageMap::writeXML( std::ofstream *outstream,
                            	   std::ofstream *out1stream,
                            	   std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelLowVoltageMap::writeXML()]\t\t\t    " ;

  std::map<PixelModuleName, pair< string, pair<string, string> > >::const_iterator imodule=dpNameMap_.begin();

  for (;imodule!=dpNameMap_.end();++imodule) {
    *outstream << "  <DATA>"                                                                              << std::endl ;
    *outstream << "   <PANEL_NAME>" << imodule->first		     << "</PANEL_NAME>"   		  << std::endl ;
    *outstream << "   <DATAPOINT>"  << imodule->second.first	     << "</DATAPOINT>"    		  << std::endl ;
    *outstream << "   <LV_DIGITAL>" << imodule->second.second.first  << "</LV_DIGITAL>"   		  << std::endl ;
    *outstream << "   <LV_ANALOG>"  << imodule->second.second.second << "</LV_ANALOG>" 			  << std::endl ;
    *outstream << "  </DATA>"                                                                             << std::endl ;
    *outstream << ""                                                                                      << std::endl ;
  }
}

//=============================================================================================
void PixelLowVoltageMap::writeXMLTrailer(std::ofstream *outstream,
                             	         std::ofstream *out1stream,
                             	         std::ofstream *out2stream ) const 
{
  std::string mthn = "[PixelLowVoltageMap::writeXMLTrailer()]\t\t\t    " ;
  
  *outstream << " </DATA_SET>"		 								  << std::endl ;
  *outstream << "</ROOT>"  		 								  << std::endl ;
  
  outstream->close() ;
  std::cout << __LINE__ << "]\t" << mthn << "Data written "   						  << std::endl ;

}

