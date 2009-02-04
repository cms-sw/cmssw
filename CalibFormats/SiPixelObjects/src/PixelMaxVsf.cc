//
// Implementation of the max Vsf
//
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelMaxVsf.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <ios>
#include <assert.h>
#include <stdio.h>

using namespace std;
using namespace pos;


PixelMaxVsf::PixelMaxVsf(std::vector< std::vector< std::string > > &tableMat):PixelConfigBase("","","")
{
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  /**

  View's name: CONF_KEY_ROC_MAXVSF_MV
  CONFIG_KEY_ID                                      NUMBER(38)
  CONFG_KEY                                          VARCHAR2(80)
  VERSION                                            VARCHAR2(40)
  KIND_OF_COND                                       VARCHAR2(40)
  ROC_NAME                                           VARCHAR2(187)
  MAXVSF                                             NUMBER(38)
  */

  colNames.push_back("CONFIG_KEY_ID");
  colNames.push_back("CONFG_KEY"    );
  colNames.push_back("VERSION"      );
  colNames.push_back("KIND_OF_COND" );
  colNames.push_back("ROC_NAME"     );
  colNames.push_back("MAXVSF"       );
  
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
	  std::cerr << "[PixelMaxVsf::PixelMaxVsf()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }
  
  rocs_.clear();
  
  for(unsigned int r = 1 ; r < tableMat.size() ; r++)    //Goes to every row of the Matrix
    {
      PixelROCName roc(tableMat[r][colM["ROC_NAME"]]);
      unsigned int vsf;
      vsf = atoi(tableMat[r][colM["MAXVSF"]].c_str());
      rocs_[roc]=vsf;
    }
}//end constructor

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PixelMaxVsf::PixelMaxVsf(std::string filename):
  PixelConfigBase("","",""){


  if (filename[filename.size()-1]=='t'){

    std::ifstream in(filename.c_str());

    if (!in.good()){
      std::cout << "[PixelMaxVsf::PixelMaxVsf()]\t\tCould not open:"<<filename<<std::endl;
      assert(0);
    }
    else {
      std::cout << "[PixelMaxVsf::PixelMaxVsf()]\t\tOpened:"<<filename<<std::endl;
    }
	
    if (in.eof()){
      std::cout << "[PixelMaxVsf::PixelMaxVsf()]\t\teof before reading anything!"<<std::endl;
      ::abort();
    }

	
    rocs_.clear();
	
    std::string rocname;
	
    in >> rocname;
    while (!in.eof()){
      //cout << "Read rocname:"<<rocname<<endl;
      PixelROCName roc(rocname);
      unsigned int vsf;
      in >> vsf;
      rocs_[roc]=vsf;
      in >> rocname;
    }
    return;
  }
  else{
    assert(0);
  }

}
 
bool PixelMaxVsf::getVsf(PixelROCName roc, unsigned int& Vsf) const{

  std::map<PixelROCName,unsigned int>::const_iterator itr = rocs_.find(roc);

  if (itr==rocs_.end()) {
    return false;
  }

  Vsf=itr->second;

  return true;

}


void PixelMaxVsf::setVsf(PixelROCName roc, unsigned int Vsf){

  rocs_[roc]=Vsf;

}



void PixelMaxVsf::writeASCII(std::string dir) const {

  if (dir!="") dir+="/";
  std::string filename=dir+"maxvsf.dat";

  std::ofstream out(filename.c_str(), std::ios_base::out) ;
  if(!out) {
    std::cout << "[PixelMaxVsf::writeASCII()] Could not open file " << filename << " for write" << std::endl ;
    exit(1);
  }


  std::map<PixelROCName, unsigned int>::const_iterator irocs = rocs_.begin();
  for(; irocs != rocs_.end() ; irocs++){
    out << (irocs->first).rocname() << " " << irocs->second << endl ;
  }
  
  out.close();

}



