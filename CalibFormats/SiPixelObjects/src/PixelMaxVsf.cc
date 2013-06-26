//
// Implementation of the max Vsf
//
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelMaxVsf.h"
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


PixelMaxVsf::PixelMaxVsf(std::vector< std::vector< std::string > > &tableMat):PixelConfigBase("","","")
{
  std::string mthn = "[PixelMaxVsf::PixelMaxVsf()]\t\t\t\t    " ;
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  /**
  
  EXTENSION_TABLE_NAME: ROC_MAXVSF (VIEW: CONF_KEY_ROC_MAXVSF_V)
  
  CONFIG_KEY				    NOT NULL VARCHAR2(80)
  KEY_TYPE				    NOT NULL VARCHAR2(80)
  KEY_ALIAS				    NOT NULL VARCHAR2(80)
  VERSION					     VARCHAR2(40)
  KIND_OF_COND  			    NOT NULL VARCHAR2(40)
  ROC_NAME					     VARCHAR2(200)
  MAXVSF				    NOT NULL NUMBER(38)
  */

  colNames.push_back("CONFIG_KEY"  );
  colNames.push_back("KEY_TYPE"    );
  colNames.push_back("KEY_ALIAS"   );
  colNames.push_back("VERSION"     );
  colNames.push_back("KIND_OF_COND");
  colNames.push_back("ROC_NAME"    );
  colNames.push_back("MAXVSF"	   );

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

  std::string mthn = "[PixelMaxVsf::PixelMaxVsf()]\t\t\t\t    " ;

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
      throw std::runtime_error("File appears to be empty: "+filename);
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

  std::string mthn = "[PixelMaxVsf::writeASCII()]\t\t\t\t    " ;
  if (dir!="") dir+="/";
  std::string filename=dir+"maxvsf.dat";

  std::ofstream out(filename.c_str(), std::ios_base::out) ;
  if(!out) {
    std::cout << __LINE__ << "]\t" << mthn << "Could not open file " << filename << " for write" << std::endl ;
    exit(1);
  }


  std::map<PixelROCName, unsigned int>::const_iterator irocs = rocs_.begin();
  for(; irocs != rocs_.end() ; irocs++){
    out << (irocs->first).rocname() << " " << irocs->second << endl ;
  }
  
  out.close();

}

//=============================================================================================
void PixelMaxVsf::writeXMLHeader(pos::PixelConfigKey key, 
                                 int version, 
                                 std::string path, 
                                 std::ofstream *outstream,
                                 std::ofstream *out1stream,
                                 std::ofstream *out2stream) const
{
  std::string mthn = "[PixelMaxVsf::writeXMLHeader()]\t\t\t    " ;
  std::stringstream maskFullPath ;

  maskFullPath << path << "/Pixel_RocMaxVsf_" << PixelTimeFormatter::getmSecTime() << ".xml";
  std::cout << __LINE__ << "]\t" << mthn << "Writing to: " << maskFullPath.str() << std::endl ;

  outstream->open(maskFullPath.str().c_str()) ;
  
  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"                                 << std::endl ;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 		 	          << std::endl ;
  *outstream << ""                                                                                        << std::endl ; 
  *outstream << " <HEADER>"                                                                               << std::endl ; 
  *outstream << "  <TYPE>"                                                                                << std::endl ; 
  *outstream << "   <EXTENSION_TABLE_NAME>ROC_MAXVSF</EXTENSION_TABLE_NAME>"                              << std::endl ; 
  *outstream << "   <NAME>ROC MaxVsf Setting</NAME>"                                                      << std::endl ; 
  *outstream << "  </TYPE>"                                                                               << std::endl ; 
  *outstream << "  <RUN>"                                                                                 << std::endl ; 
  *outstream << "   <RUN_TYPE>ROC MaxVsf Settings</RUN_TYPE>"                                             << std::endl ; 
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
void PixelMaxVsf::writeXML( std::ofstream *outstream,
                            std::ofstream *out1stream,
                            std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelMaxVsf::writeXML()]\t\t\t    " ;

  std::map<PixelROCName, unsigned int>::const_iterator irocs = rocs_.begin();
  for(; irocs != rocs_.end() ; irocs++){
    *outstream << "  <DATA>"                                                   << std::endl ;
    *outstream << "   <ROC_NAME>" << (irocs->first).rocname() << "</ROC_NAME>" << std::endl ;
    *outstream << "   <MAXVSF>"   << irocs->second            << "</MAXVSF>"   << std::endl ;
    *outstream << "  </DATA>"                                                  << std::endl ;
    *outstream                                                                 << std::endl ;
  }
}

//=============================================================================================
void PixelMaxVsf::writeXMLTrailer(std::ofstream *outstream,
                             	  std::ofstream *out1stream,
                             	  std::ofstream *out2stream ) const 
{
  std::string mthn = "[PixelMaxVsf::writeXMLTrailer()]\t\t\t    " ;
  
  *outstream << " </DATA_SET>"		 								  << std::endl ;
  *outstream << "</ROOT>"  		 								  << std::endl ;
  
  outstream->close() ;
  std::cout << __LINE__ << "]\t" << mthn << "Data written "   						  << std::endl ;

}
