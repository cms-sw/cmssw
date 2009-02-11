//
// Implementation of the detector configuration
//
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelLowVoltageMap.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <ios>
#include <assert.h>
#include <stdio.h>

using namespace std;
using namespace pos;


PixelLowVoltageMap::PixelLowVoltageMap(std::vector< std::vector < std::string> > &tableMat):PixelConfigBase("","","")
{
  std::string mthn = "[PixelLowVoltageMap::PixelLowVoltageMap()] " ;
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
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
  

  if (filename[filename.size()-1]=='t'){
    
    std::ifstream in(filename.c_str());
    
    if (!in.good()){
      std::cout << "Could not open:"<<filename<<std::endl;
      assert(0);
    }
    else {
      std::cout << "Opened:"<<filename<<std::endl;
    }
    
    if (in.eof()){
      std::cout << "eof before reading anything!"<<std::endl;
      ::abort();
    }

    
    dpNameMap_.clear();
    
    std::string modulename;
    std::string dpNameBase;
    std::string ianaChannel;
    std::string idigiChannel;
    
    in >> modulename >> dpNameBase >> ianaChannel >> idigiChannel;
    
    while (!in.eof()){
      cout << "Read modulename:"<<modulename<<endl;
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

  std::map<PixelModuleName, pair< string, pair<string, string> > >::const_iterator i=
    dpNameMap_.find(module);
  
  if (i==dpNameMap_.end()) {
    cout << "PixelLowVoltageMap::dpName: Could not find module:"<<module
	 << endl;
  }
  
  return i->second.first+"/"+i->second.second.first;

}

std::string PixelLowVoltageMap::dpNameIdigi(const PixelModuleName& module) const{

  std::map<PixelModuleName, pair< string, pair<string, string> > >::const_iterator i=
    dpNameMap_.find(module);
  
  if (i==dpNameMap_.end()) {
    cout << "PixelLowVoltageMap::dpName: Could not find module:"<<module
	 << endl;
  }

  return i->second.first+"/"+i->second.second.second;

}


void PixelLowVoltageMap::writeASCII(std::string dir) const {

  if (dir!="") dir+="/";
  std::string filename=dir+"lowvoltagemap.dat";

  std::ofstream out(filename.c_str(), std::ios_base::out) ;
  if(!out) {
    std::cout << "[PixelLowVoltageMap::writeASCII()] Could not open file " << filename << " for write" << std::endl ;
    exit(1);
  }
  std::map<PixelModuleName, pair< string, pair<string, string> > >::const_iterator imodule=
    dpNameMap_.begin();

  for (;imodule!=dpNameMap_.end();++imodule) {
    out << imodule->first<<" "<<imodule->second.first 
	<< " "<<imodule->second.second.first
	<< " "<<imodule->second.second.first<<endl;
  }

  out.close();

}



