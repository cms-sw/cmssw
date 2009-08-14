//
// This class stores the information about a FED.
// This include the number, crate, and base address
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelFEDConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include <fstream>
#include <iostream>
#include <map>
#include <assert.h>
#include <stdexcept>

using namespace pos;
using namespace std;

PixelFEDConfig::PixelFEDConfig(std::vector<std::vector<std::string> >& tableMat ) : PixelConfigBase(" "," "," "){

  std::string mthn = "[PixelFEDConfig::PixelFEDConfig()]\t\t\t    " ;

  std::vector< std::string > ins = tableMat[0];
  std::map<std::string , int > colM;
   std::vector<std::string > colNames;
/*
   EXTENSION_TABLE_NAME: FED_CRATE_CONFIG (VIEW: CONF_KEY_FED_CRATE_CONFIGV)
   
   CONFIG_KEY				     NOT NULL VARCHAR2(80)
   KEY_TYPE				     NOT NULL VARCHAR2(80)
   KEY_ALIAS				     NOT NULL VARCHAR2(80)
   VERSION					      VARCHAR2(40)
   KIND_OF_COND 			     NOT NULL VARCHAR2(40)
   PIXEL_FED				     NOT NULL NUMBER(38)
   CRATE_NUMBER 			     NOT NULL NUMBER(38)
   VME_ADDR				     NOT NULL VARCHAR2(200)
*/

    colNames.push_back("CONFIG_KEY"    );
    colNames.push_back("KEY_TYPE"      );
    colNames.push_back("KEY_ALIAS"     );
    colNames.push_back("VERSION"       );
    colNames.push_back("KIND_OF_COND"  );
    colNames.push_back("PIXEL_FED"     );
    colNames.push_back("CRATE_NUMBER"  );
    colNames.push_back("VME_ADDR" );
/*
   colNames.push_back("PIXEL_FED"    ); //0
   colNames.push_back("CRATE_NUMBER" ); //1
   colNames.push_back("VME_ADDRS_HEX"); //2
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
   /*
   for(unsigned int n=0; n<colNames.size(); n++)
     {
       if(colM.find(colNames[n]) == colM.end())
	 {
	   std::cerr << __LINE__ << "]\t" << mthn << "Couldn't find in the database the column with name " << colNames[n] << std::endl;
	   assert(0);
	 }
     }
   */

   std::string fedname = "";
   unsigned int fednum = 0;
   fedconfig_.clear();
   bool flag = false;
   for(unsigned int r = 1 ; r < tableMat.size() ; r++){    //Goes to every row of the Matrix
     
     fedname = tableMat[r][colM["PIXEL_FED"]]; //This is not going to work if you change in the database "PxlFed_#" in the FED column.Im removing "PlxFed_" and store the number
     //becuase the PixelFecConfig class ask for the fec number not the name.  
     // 01234567
     // PxlFED_XX
//      fedname.erase(0,7); 
     fednum = (unsigned int)atoi(fedname.c_str()) ;
     
     if(fedconfig_.empty())
       {
       PixelFEDParameters tmp;
       unsigned int vme_base_address = 0 ;
       vme_base_address = strtoul(tableMat[r][colM["VME_ADDR"]].c_str(), 0, 16);
//        string hexVMEAddr = tableMat[r][colM["VME_ADDRS_HEX"]] ;
//        sscanf(hexVMEAddr.c_str(), "%x", &vme_base_address) ;
       tmp.setFEDParameters( fednum, (unsigned int)atoi(tableMat[r][colM["CRATE_NUMBER"]].c_str()) , 
			     vme_base_address);   
       fedconfig_.push_back(tmp);
     }
     else
       {
	 for( unsigned int y = 0; y < fedconfig_.size() ; y++)
	   {
	     if (fedconfig_[y].getFEDNumber() == fednum)    // This is to check if there are Pixel Feds already in the vector because
	       {	                                    // in the view of the database that I'm reading there are many repeated entries (AS FAR AS THESE PARAMS ARE CONCERNED).
		 flag = true;				    // This ensure that there are no objects in the fedconfig vector with repeated values.
		 break;
	       }
	     else flag = false;
	   }
	 
	 if(flag == false)
	   {
	     PixelFEDParameters tmp;
	     tmp.setFEDParameters( fednum, (unsigned int)atoi(tableMat[r][colM["CRATE_NUMBER"]].c_str()) , 
				   (unsigned int)strtoul(tableMat[r][colM["VME_ADDR"]].c_str(), 0, 16));   
	     fedconfig_.push_back(tmp); 
	   }
       }//end else 
   }//end for r
/*   
   std::cout << __LINE__ << "]\t"    << mthn                      << std::endl;
   
   for( unsigned int x = 0 ; x < fedconfig_.size() ; x++)
     {
       std::cout<< __LINE__ << "]\t" << mthn << fedconfig_[x]     << std::endl;
     }
   
   std::cout<< __LINE__ << "]\t"     << mthn << fedconfig_.size() << std::endl;
*/   
}//end Constructor



//*****************************************************************************************************




PixelFEDConfig::PixelFEDConfig(std::string filename):
  PixelConfigBase(" "," "," "){

    std::string mthn = "[PixelFEDConfig::PixelFEDConfig()]\t\t\t    " ;
    std::ifstream in(filename.c_str());

    if (!in.good()){
      std::cout << __LINE__ << "]\t" << mthn << "Could not open: " << filename.c_str() << std::endl;
      throw std::runtime_error("Failed to open file "+filename);
    }
    else {
      std::cout << __LINE__ << "]\t" << mthn << "Opened: " << filename.c_str() << std::endl;
    }

    std::string dummy;

    in >> dummy;
    in >> dummy;
    in >> dummy;
    in >> dummy;
    in >> dummy;
    in >> dummy;

    do {
	
      unsigned int fednumber;
      unsigned int crate;
      unsigned int vme_base_address;

      in >> fednumber >> crate >> std::hex >> vme_base_address >> std::dec;

      if (!in.eof() ){
	//	std::cout << __LINE__ << "]\t" << mthn << std::dec << fednumber <<" "<< crate << " 0x"  
	//                << std::hex << vme_base_address<<std::dec<<std::endl;
	PixelFEDParameters tmp;
	    
	tmp.setFEDParameters(fednumber , crate , vme_base_address);
	    
	fedconfig_.push_back(tmp); 
      }

    }
    while (!in.eof());
    in.close();

  }

//std::ostream& operator<<(std::ostream& s, const PixelFEDConfig& table){

//for (unsigned int i=0;i<table.translationtable_.size();i++){
//	s << table.translationtable_[i]<<std::endl;
//   }
// return s;

//}

PixelFEDConfig::~PixelFEDConfig() {}

void PixelFEDConfig::writeASCII(std::string dir) const {

  std::string mthn = "[PixelFEDConfig::writeASCII()]\t\t\t\t    " ;
  if (dir!="") dir+="/";
  string filename=dir+"fedconfig.dat";

  ofstream out(filename.c_str());
  if(!out.good()){
    cout << __LINE__ << "]\t" << mthn << "Could not open file: " << filename << endl;
    assert(0);
  }

  out <<" #FED number     crate     vme base address" <<endl;
  for(unsigned int i=0;i<fedconfig_.size();i++){
    out << fedconfig_[i].getFEDNumber()<<"               "
	<< fedconfig_[i].getCrate()<<"         "
	<< "0x"<<hex<<fedconfig_[i].getVMEBaseAddress()<<dec<<endl;
  }
  out.close();
}


unsigned int PixelFEDConfig::getNFEDBoards() const{

  return fedconfig_.size();

}

unsigned int PixelFEDConfig::getFEDNumber(unsigned int i) const{

  assert(i<fedconfig_.size());
  return fedconfig_[i].getFEDNumber();

}


unsigned int PixelFEDConfig::getCrate(unsigned int i) const{

  assert(i<fedconfig_.size());
  return fedconfig_[i].getCrate();

}


unsigned int PixelFEDConfig::getVMEBaseAddress(unsigned int i) const{

  assert(i<fedconfig_.size());
  return fedconfig_[i].getVMEBaseAddress();

}


unsigned int PixelFEDConfig::crateFromFEDNumber(unsigned int fednumber) const{


  std::string mthn = "[PixelFEDConfig::crateFromFEDNumber()]\t\t\t    " ;
  for(unsigned int i=0;i<fedconfig_.size();i++){
    if (fedconfig_[i].getFEDNumber()==fednumber) return fedconfig_[i].getCrate();
  }

  std::cout << __LINE__ << "]\t" << mthn << "Could not find FED number: " << fednumber << std::endl;

  assert(0);

  return 0;

}


unsigned int PixelFEDConfig::VMEBaseAddressFromFEDNumber(unsigned int fednumber) const{

  std::string mthn = "[PixelFEDConfig::VMEBaseAddressFromFEDNumber()]\t\t    " ;
  for(unsigned int i=0;i<fedconfig_.size();i++){
    if (fedconfig_[i].getFEDNumber()==fednumber) return fedconfig_[i].getVMEBaseAddress();
  }

  std::cout << __LINE__ << "]\t" << mthn << "Could not find FED number: " << fednumber << std::endl;

  assert(0);

  return 0;

}

unsigned int PixelFEDConfig::FEDNumberFromCrateAndVMEBaseAddress(unsigned int crate, unsigned int vmebaseaddress) const {

  std::string mthn = "[PixelFEDConfig::FEDNumberFromCrateAndVMEBaseAddress()]\t    " ;
  for(unsigned int i=0;i<fedconfig_.size();i++){
    if (fedconfig_[i].getCrate()==crate&&
        fedconfig_[i].getVMEBaseAddress()==vmebaseaddress) return fedconfig_[i].getFEDNumber();
  }

  std::cout << __LINE__ << "]\t" << mthn << "Could not find FED crate and address: "<< crate << ", " << vmebaseaddress << std::endl;

  assert(0);

  return 0;

}

//=============================================================================================
void PixelFEDConfig::writeXMLHeader(pos::PixelConfigKey key, 
                                    int version, 
                                    std::string path, 
                                    std::ofstream *outstream,
                                    std::ofstream *out1stream,
                                    std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelFEDConfig::::writeXMLHeader()]\t\t\t    " ;
  std::stringstream fullPath ;
  fullPath << path << "/Pixel_FedCrateConfig_" << PixelTimeFormatter::getmSecTime() << ".xml" ;
  cout << __LINE__ << "]\t" << mthn << "Writing to: " << fullPath.str() << endl ;
  
  outstream->open(fullPath.str().c_str()) ;
  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"			 	     << endl ;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 		 	             << endl ;
  *outstream << " <HEADER>"								         	     << endl ;
  *outstream << "  <TYPE>"								         	     << endl ;
  *outstream << "   <EXTENSION_TABLE_NAME>FED_CRATE_CONFIG</EXTENSION_TABLE_NAME>"          	             << endl ;
  *outstream << "   <NAME>Pixel FED Crate Configuration</NAME>"				         	     << endl ;
  *outstream << "  </TYPE>"								         	     << endl ;
  *outstream << "  <RUN>"								         	     << endl ;
  *outstream << "   <RUN_NAME>Pixel FED Crate Configuration</RUN_NAME>" 		                     << endl ;
  *outstream << "   <RUN_BEGIN_TIMESTAMP>" << pos::PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << endl ;
  *outstream << "   <LOCATION>CERN P5</LOCATION>"                                                            << endl ; 
  *outstream << "  </RUN>"								         	     << endl ;
  *outstream << " </HEADER>"								         	     << endl ;
  *outstream << "  "								         	             << endl ;
  *outstream << " <DATA_SET>"								         	     << endl ;
  *outstream << "  <PART>"                                                                                   << endl ;
  *outstream << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"                                                 << endl ;
  *outstream << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"                                              << endl ;
  *outstream << "  </PART>"                                                                                  << endl ;
  *outstream << "  <VERSION>"             << version      << "</VERSION>"				     << endl ;
  *outstream << "  <COMMENT_DESCRIPTION>" << getComment() << "</COMMENT_DESCRIPTION>"			     << endl ;
  *outstream << "  <INITIATED_BY_USER>"   << getAuthor()  << "</INITIATED_BY_USER>"			     << endl ;
  *outstream << "  "								         	             << endl ;
}  

//=============================================================================================
void PixelFEDConfig::writeXML(std::ofstream *outstream,
                              std::ofstream *out1stream,
                              std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelFEDConfig::writeXML()]\t\t\t    " ;
  
  for(unsigned int i=0;i<fedconfig_.size();i++){
      *outstream << "  <DATA>" 								 	      	             << endl ;
      *outstream << "   <PIXEL_FED>"    	    << fedconfig_[i].getFEDNumber()             << "</PIXEL_FED>"    << endl ;
      *outstream << "   <CRATE_NUMBER>" 	    << fedconfig_[i].getCrate()	                << "</CRATE_NUMBER>" << endl ;
      *outstream << "   <VME_ADDR>"  << "0x" << hex << fedconfig_[i].getVMEBaseAddress() << dec << "</VME_ADDR>"     << endl ;
      *outstream << "  </DATA>"	 							 	      	     	     << endl ;
      *outstream << ""								         	      	     	     << endl ;
  }

}

//=============================================================================================
void PixelFEDConfig::writeXMLTrailer(std::ofstream *outstream,
                                     std::ofstream *out1stream,
                                     std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelFEDConfig::writeXMLTrailer()]\t\t\t    " ;
  
  *outstream << " </DATA_SET>" 						    	 	              	     	     << endl ;
  *outstream << "</ROOT> "								              	     	     << endl ;

  outstream->close() ;
}
