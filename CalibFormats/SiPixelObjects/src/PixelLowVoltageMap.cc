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


PixelLowVoltageMap::PixelLowVoltageMap(std::vector< std::vector < std::string> > &tableMat):PixelConfigBase("","",""){

  //FIXME not implemented
  assert(0);

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



