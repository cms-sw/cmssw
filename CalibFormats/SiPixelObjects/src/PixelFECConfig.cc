//
// This class stores the information about a FEC.
// This include the number, crate, and base address
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelFECConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include <fstream>
#include <sstream>
#include <map>
#include <assert.h>
#include <stdexcept>

using namespace pos;
using namespace std;



PixelFECConfig::PixelFECConfig(std::vector<std::vector<std::string> >& tableMat ) : PixelConfigBase(" "," "," "){

 std::map<std::string , int > colM;
 std::vector<std::string > colNames;
 /**
 
   EXTENSION_TABLE_NAME:  (VIEW: )
   
   CONFIG_KEY				     NOT NULL VARCHAR2(80)
   KEY_TYPE				     NOT NULL VARCHAR2(80)
   KEY_ALIAS				     NOT NULL VARCHAR2(80)
   VERSION					      VARCHAR2(40)
   KIND_OF_COND 			     NOT NULL VARCHAR2(40)
   PIXFEC_NAME  			     NOT NULL VARCHAR2(200)
   CRATE_NUMBER 			     NOT NULL NUMBER(38)
   SLOT_NUMBER  				      NUMBER(38)
   VME_ADDR				     NOT NULL VARCHAR2(200)
 */

 colNames.push_back("CONFIG_KEY"   );	    
 colNames.push_back("KEY_TYPE"     );	    
 colNames.push_back("KEY_ALIAS"    );	    
 colNames.push_back("VERSION"	   );	    
 colNames.push_back("KIND_OF_COND" ); 
 colNames.push_back("PIXFEC_NAME"  ); 
 colNames.push_back("CRATE_NUMBER" ); 
 colNames.push_back("SLOT_NUMBER"  ); 
 colNames.push_back("VME_ADDR"     );	    

 for(unsigned int c = 0 ; c < tableMat[0].size() ; c++)
   {
     for(unsigned int n=0; n<colNames.size(); n++)
       {
	 if(tableMat[0][c] == colNames[n]){
	   colM[colNames[n]] = c;
	   break;
	 }
       }
   }//end for
 /*
 for(unsigned int n=0; n<colNames.size(); n++)
   {
     if(colM.find(colNames[n]) == colM.end()){
       std::cerr << "[PixelFECConfig::PixelFECConfig()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
       assert(0);
     }
   }
 */

 fecconfig_.clear();
 for(unsigned int r = 1 ; r < tableMat.size() ; r++)    //Goes to every row of the Matrix
   {
     unsigned int fecnumber;
     unsigned int crate;
     unsigned int vme_base_address;
     
//      01234567890123
//      BPix_Pxl_FEC_1
//     string fullFECName = tableMat[r][colM["PIXEL_FEC"]] ;
//     fullFECName.replace(0,13,"") ;
     fecnumber = atoi(tableMat[r][colM["PIXFEC_NAME"]].c_str()) ;
     crate     = atoi(tableMat[r][colM["CRATE_NUMBER"]].c_str()) ;
     string hexVMEAddr = tableMat[r][colM["VME_ADDR"]] ;
     sscanf(hexVMEAddr.c_str(), "%x", &vme_base_address) ;
     PixelFECParameters tmp;
     
     tmp.setFECParameters(fecnumber , crate , vme_base_address);
     
     fecconfig_.push_back(tmp);
   }
 
}// end contructor

//****************************************************************************************

 
PixelFECConfig::PixelFECConfig(std::string filename):
    PixelConfigBase(" "," "," "){

    std::string mthn = "[[PixelFECConfig::PixelFECConfig()]\t\t\t   " ;
    
    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << __LINE__ << "]\t" << mthn << "Could not open: " << filename << std::endl;
	throw std::runtime_error("Failed to open file "+filename);
    }
    else {
	std::cout << __LINE__ << "]\t" << mthn <<" Opened: "         << filename << std::endl;
    }

    std::string dummy;

    in >> dummy;
    in >> dummy;
    in >> dummy;
    in >> dummy;
    in >> dummy;
    in >> dummy;

    do {
	
	unsigned int fecnumber;
	unsigned int crate;
	unsigned int vme_base_address;

	in >> fecnumber >> crate >> std::hex>> vme_base_address >>std::dec ;

	if (!in.eof() ){
	    //std::cout << __LINE__ << "]\t" << mthn << fecnumber <<" "<< crate << " "  
	    //      << std::hex << vme_base_address<<std::dec<<std::endl;
	    
	    PixelFECParameters tmp;
	    
	    tmp.setFECParameters(fecnumber , crate , vme_base_address);
	    
	    fecconfig_.push_back(tmp);
	}

    }
    while (!in.eof());
    in.close();
}
 

//std::ostream& operator<<(std::ostream& s, const PixelFECConfig& table){

    //for (unsigned int i=0;i<table.translationtable_.size();i++){
    //	s << table.translationtable_[i]<<std::endl;
    //   }
// return s;

//}


unsigned int PixelFECConfig::getNFECBoards() const{

    return fecconfig_.size();

}

unsigned int PixelFECConfig::getFECNumber(unsigned int i) const{

    assert(i<fecconfig_.size());
    return fecconfig_[i].getFECNumber();

}


unsigned int PixelFECConfig::getCrate(unsigned int i) const{

    assert(i<fecconfig_.size());
    return fecconfig_[i].getCrate();

}


unsigned int PixelFECConfig::getVMEBaseAddress(unsigned int i) const{

    assert(i<fecconfig_.size());
    return fecconfig_[i].getVMEBaseAddress();

}


unsigned int PixelFECConfig::crateFromFECNumber(unsigned int fecnumber) const{

    std::string mthn = "[PixelFECConfig::crateFromFECNumber()]\t\t\t    " ;
    for(unsigned int i=0;i<fecconfig_.size();i++){
	if (fecconfig_[i].getFECNumber()==fecnumber) return fecconfig_[i].getCrate();
    }

    std::cout << __LINE__ << "]\t" << mthn << "Could not find FEC number: " << fecnumber << std::endl;

    assert(0);

    return 0;

}

unsigned int PixelFECConfig::VMEBaseAddressFromFECNumber(unsigned int fecnumber) const{

    std::string mthn = "[PixelFECConfig::VMEBaseAddressFromFECNumber()]\t\t    " ;
    for(unsigned int i=0;i<fecconfig_.size();i++){
	if (fecconfig_[i].getFECNumber()==fecnumber) return fecconfig_[i].getVMEBaseAddress();
    }

    std::cout << __LINE__ << "]\t" << mthn << "Could not find FEC number: " << fecnumber << std::endl;

    assert(0);

    return 0;

}

//=============================================================================================
void PixelFECConfig::writeASCII(std::string dir) const {

  if (dir!="") dir+="/";
  std::string filename=dir+"fecconfig.dat";
  std::ofstream out(filename.c_str());

  std::vector< PixelFECParameters >::const_iterator i=fecconfig_.begin();

  out << "#FEC number     crate     vme base address" << endl;
  for(;i!=fecconfig_.end();++i){
    out << i->getFECNumber()<<"               "
        << i->getCrate()<<"         "
        << "0x"<<hex<<i->getVMEBaseAddress()<<dec<<endl;
  }

}

//=============================================================================================
void PixelFECConfig::writeXMLHeader(pos::PixelConfigKey key, 
                                    int version, 
                                    std::string path, 
                                    std::ofstream *outstream,
                                    std::ofstream *out1stream,
                                    std::ofstream *out2stream) const
{
  std::string mthn = "[PixelFECConfig::writeXMLHeader()]\t\t\t    " ;
  std::stringstream fullPath ;
  fullPath << path << "/Pixel_PixelFecParameters_" << PixelTimeFormatter::getmSecTime() << ".xml" ;
  cout << __LINE__ << "]\t" << mthn << "Writing to: " << fullPath.str() << endl ;
  
  outstream->open(fullPath.str().c_str()) ;
  
  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"			 	     << endl ;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 		 	             << endl ;
  *outstream << " <HEADER>"								         	     << endl ;
  *outstream << "  <TYPE>"								         	     << endl ;
  *outstream << "   <EXTENSION_TABLE_NAME>PIXEL_FEC_PARAMETERS</EXTENSION_TABLE_NAME>"          	     << endl ;
  *outstream << "   <NAME>Pixel FEC Parameters</NAME>"				         	             << endl ;
  *outstream << "  </TYPE>"								         	     << endl ;
  *outstream << "  <RUN>"								         	     << endl ;
  *outstream << "   <RUN_TYPE>Pixel FEC Parameters</RUN_TYPE>" 		                                     << endl ;
  *outstream << "   <RUN_NUMBER>1</RUN_NUMBER>"					         	             << endl ;
  *outstream << "   <RUN_BEGIN_TIMESTAMP>" << pos::PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << endl ;
  *outstream << "   <LOCATION>CERN P5</LOCATION>"                                                            << endl ; 
  *outstream << "  </RUN>"								         	     << endl ;
  *outstream << " </HEADER>"								         	     << endl ;
  *outstream << ""										 	     << endl ;
  *outstream << " <DATA_SET>"								         	     << endl ;
  *outstream << "  <PART>"                                                                                   << endl ;
  *outstream << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"                                                 << endl ;
  *outstream << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"                                              << endl ;
  *outstream << "  </PART>"                                                                                  << endl ;
  *outstream << "  <VERSION>"             << version      << "</VERSION>"				     << endl ;
  *outstream << "  <COMMENT_DESCRIPTION>" << getComment() << "</COMMENT_DESCRIPTION>"			     << endl ;
  *outstream << "  <CREATED_BY_USER>"     << getAuthor()  << "</CREATED_BY_USER>"  			     << endl ;
}

//=============================================================================================
void PixelFECConfig::writeXML( std::ofstream *outstream,
                               std::ofstream *out1stream,
                               std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelFECConfig::writeXML()]\t\t\t    " ;

  std::vector< PixelFECParameters >::const_iterator i=fecconfig_.begin();

  for(;i!=fecconfig_.end();++i){
   *outstream << ""                                                                           	             << endl ;
   *outstream << "  <DATA>"                                                                           	     << endl ;
   *outstream << "   <PIXFEC_NAME>"            << i->getFECNumber() 		<< "</PIXFEC_NAME>" 	     << endl ;
   *outstream << "   <CRATE_NUMBER>"           << i->getCrate()     		<< "</CRATE_NUMBER>" 	     << endl ;
   *outstream << "   <VME_ADDR>0x" << hex << i->getVMEBaseAddress() << dec      << "</VME_ADDR>" 	     << endl ;
   *outstream << "  </DATA>"                                                                          	     << endl ;
  }
}

//=============================================================================================
void PixelFECConfig::writeXMLTrailer(std::ofstream *outstream,
                                     std::ofstream *out1stream,
                                     std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelFECConfig::writeXMLTrailer()]\t\t\t    " ;
  
  *outstream << "" 						    	 	              	             << endl ;
  *outstream << " </DATA_SET>" 						    	 	              	     << endl ;
  *outstream << "</ROOT> "								              	     << endl ;

  outstream->close() ;
}

