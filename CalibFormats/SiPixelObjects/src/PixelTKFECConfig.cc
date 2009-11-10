//
// This class stores the information about a TKFEC.
// This include the number, crate, and base address
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelTKFECConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include <fstream>
#include <sstream>
#include <map>
#include <assert.h>
#include <stdexcept>

using namespace pos;
using namespace std;


PixelTKFECConfig::PixelTKFECConfig(std::vector<std::vector<std::string> >& tableMat ) : PixelConfigBase(" "," "," ")
{
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  /**

  EXTENSION_TABLE_NAME: TRACKER_FEC_PARAMETERS (VIEW: CONF_KEY_TRACKER_FEC_CONFIG_V)
  
  CONFIG_KEY				    NOT NULL VARCHAR2(80)
  KEY_TYPE				    NOT NULL VARCHAR2(80)
  KEY_ALIAS				    NOT NULL VARCHAR2(80)
  VERSION					     VARCHAR2(40)
  KIND_OF_COND  			    NOT NULL VARCHAR2(40)
  TRKFEC_NAME				    NOT NULL VARCHAR2(200)
  CRATE_LABEL					     VARCHAR2(200)
  CRATE_NUMBER  			    NOT NULL NUMBER(38)
  TYPE  					     VARCHAR2(200)
  SLOT_NUMBER					     NUMBER(38)
  VME_ADDR				    NOT NULL VARCHAR2(200)
  I2CSPEED					     NUMBER(38)

  */

  colNames.push_back("CONFIG_KEY"  ); 
  colNames.push_back("KEY_TYPE"    ); 
  colNames.push_back("KEY_ALIAS"   ); 
  colNames.push_back("VERSION"     ); 
  colNames.push_back("KIND_OF_COND"); 
  colNames.push_back("TRKFEC_NAME" ); 
  colNames.push_back("CRATE_LABEL" ); 
  colNames.push_back("CRATE_NUMBER"); 
  colNames.push_back("TYPE"	   );	      
  colNames.push_back("SLOT_NUMBER" ); 
  colNames.push_back("VME_ADDR"    );
  colNames.push_back("I2CSPEED"    );

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
	  std::cerr << "[PixelTKFECConfig::PixelTKFECConfig()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }
  */

  for(unsigned int r = 1 ; r < tableMat.size() ; r++)    //Goes to every row of the Matrix
    {
      std::string TKFECID  = tableMat[r][colM["TRKFEC_NAME"]] ;
      unsigned int crate   = atoi(tableMat[r][colM["CRATE_NUMBER"]].c_str()) ;
      std::string type     = "VME" ;
      unsigned int address = strtoul(tableMat[r][colM["VME_ADDR"]].c_str() , 0, 16);
      PixelTKFECParameters tmp;
      tmp.setTKFECParameters(TKFECID , crate , type, address);
      TKFECconfig_.push_back(tmp);
      //      cout << "[PixelTKFECConfig::PixelTKFECConfig()]\tID: " << TKFECID << " crate: " << crate << " address: " << address << endl;
    }
}// end contructor

//****************************************************************************************

 
PixelTKFECConfig::PixelTKFECConfig(std::string filename):
    PixelConfigBase(" "," "," "){

    std::string mthn ="]\t[PixelTKFECConfig::PixelTKFECConfig()]\t\t\t    " ;
    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << __LINE__ << mthn << "Could not open: " << filename << std::endl;
	throw std::runtime_error("Failed to open file "+filename);
    }
    else {
	std::cout << __LINE__ << mthn << "Opened: "         << filename << std::endl;
    }

    std::string dummy;

    getline(in, dummy); // skip the column headings

    do {
	
	std::string TKFECID;
	unsigned int crate;
	std::string type;
	unsigned int address;

	in >> TKFECID >> std::dec >> crate >> type;
	if (type=="VME" || type=="PCI")
	{
		in >> std::hex>> address >>std::dec ;
	}
	else // type not specified, default to "VME"
	{
		address = strtoul(type.c_str(), 0, 16); // convert string to integer using base 16
		type = "VME";
	}

	if (!in.eof() ){
	    //std::cout << TKFECID <<" "<< crate << " "  
	    //      << std::hex << vme_base_address<<std::dec<<std::endl;
	    
	    PixelTKFECParameters tmp;
	    
	    tmp.setTKFECParameters(TKFECID , crate , type, address);
	    
	    TKFECconfig_.push_back(tmp);
	}

    }
    while (!in.eof());
    in.close();

}
 
PixelTKFECConfig::~PixelTKFECConfig() {}

void PixelTKFECConfig::writeASCII(std::string dir) const {

  if (dir!="") dir+="/";
  string filename=dir+"tkfecconfig.dat";

  ofstream out(filename.c_str());
  if(!out.good()){
    cout << "Could not open file:"<<filename<<endl;
    assert(0);
  }

  out <<"#TKFEC ID     crate     VME/PCI    slot/address" <<endl;
  for(unsigned int i=0;i<TKFECconfig_.size();i++){
    out << TKFECconfig_[i].getTKFECID()<<"          "
	<< TKFECconfig_[i].getCrate()<<"          ";
    if (TKFECconfig_[i].getType()=="PCI") {
      out << "PCI       ";
    } else {
      out << "          ";
    }
    out << "0x"<<hex<<TKFECconfig_[i].getAddress()<<dec<<endl;
  }
  out.close();
}


//std::ostream& operator<<(std::ostream& s, const PixelTKFECConfig& table){

    //for (unsigned int i=0;i<table.translationtable_.size();i++){
    //	s << table.translationtable_[i]<<std::endl;
    //   }
// return s;

//}


unsigned int PixelTKFECConfig::getNTKFECBoards() const{

    return TKFECconfig_.size();

}

std::string PixelTKFECConfig::getTKFECID(unsigned int i) const{

    assert(i<TKFECconfig_.size());
    return TKFECconfig_[i].getTKFECID();

}


unsigned int PixelTKFECConfig::getCrate(unsigned int i) const{

    assert(i<TKFECconfig_.size());
    return TKFECconfig_[i].getCrate();

}

std::string PixelTKFECConfig::getType(unsigned int i) const{

    assert(i<TKFECconfig_.size());
    return TKFECconfig_[i].getType();

}

unsigned int PixelTKFECConfig::getAddress(unsigned int i) const{

    assert(i<TKFECconfig_.size());
    return TKFECconfig_[i].getAddress();

}


unsigned int PixelTKFECConfig::crateFromTKFECID(std::string TKFECID) const{

    for(unsigned int i=0;i<TKFECconfig_.size();i++){
	if (TKFECconfig_[i].getTKFECID()==TKFECID) return TKFECconfig_[i].getCrate();
    }

    std::cout << "Could not find TKFEC ID:"<<TKFECID<<std::endl;

    assert(0);

    return 0;

}

std::string PixelTKFECConfig::typeFromTKFECID(std::string TKFECID) const{

    for(unsigned int i=0;i<TKFECconfig_.size();i++){
	if (TKFECconfig_[i].getTKFECID()==TKFECID) return TKFECconfig_[i].getType();
    }

    std::cout << "Could not find TKFEC ID:"<<TKFECID<<std::endl;

    assert(0);

    return 0;

}

unsigned int PixelTKFECConfig::addressFromTKFECID(std::string TKFECID) const{

    for(unsigned int i=0;i<TKFECconfig_.size();i++){
	if (TKFECconfig_[i].getTKFECID()==TKFECID) return TKFECconfig_[i].getAddress();
    }

    std::cout << "Could not find TKFEC ID:"<<TKFECID<<std::endl;

    assert(0);

    return 0;

}

//=============================================================================================
void PixelTKFECConfig::writeXMLHeader(pos::PixelConfigKey key, 
                                      int version, 
                                      std::string path, 
                                      std::ofstream *outstream,
                                      std::ofstream *out1stream,
                                      std::ofstream *out2stream) const
{
  std::string mthn = "[PixelTKFECConfig::writeXMLHeader()]\t\t\t    " ;
  std::stringstream maskFullPath ;

  maskFullPath << path << "/Pixel_TrackerFecParameters_" << PixelTimeFormatter::getmSecTime() << ".xml";
  std::cout << mthn << "Writing to: " << maskFullPath.str() << std::endl ;

  outstream->open(maskFullPath.str().c_str()) ;
  
  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"                                 << std::endl ;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 		 	          << std::endl ;
  *outstream << ""                                                                                        << std::endl ; 
  *outstream << " <HEADER>"                                                                               << std::endl ; 
  *outstream << "  <TYPE>"                                                                                << std::endl ; 
  *outstream << "   <EXTENSION_TABLE_NAME>TRACKER_FEC_PARAMETERS</EXTENSION_TABLE_NAME>"                  << std::endl ; 
  *outstream << "   <NAME>Tracker FEC Parameters</NAME>"                                                  << std::endl ; 
  *outstream << "  </TYPE>"                                                                               << std::endl ; 
  *outstream << "  <RUN>"                                                                                 << std::endl ; 
  *outstream << "   <RUN_TYPE>Tracker FEC Parameters</RUN_TYPE>"                                          << std::endl ; 
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
  *outstream << "  <CREATED_BY_USER>"   << getAuthor()  << "</CREATED_BY_USER>"			          << std::endl ;
  *outstream << ""                                                                                        << std::endl ;
  *outstream << "  <PART>"                                                                                << std::endl ;
  *outstream << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"                                              << std::endl ;      
  *outstream << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"                                           << std::endl ;         
  *outstream << "  </PART>"                                                                               << std::endl ;

}

//=============================================================================================
void PixelTKFECConfig::writeXML( std::ofstream *outstream,
                            	 std::ofstream *out1stream,
                            	 std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelTKFECConfig::writeXML()]\t\t\t    " ;

  for(unsigned int i=0;i<TKFECconfig_.size();i++){
    *outstream << "  <DATA>"                                                                 		         << std::endl ;
    *outstream << "   <TRKFEC_NAME>"                  << TKFECconfig_[i].getTKFECID()        << "</TRKFEC_NAME>" << std::endl ;
    *outstream << "   <CRATE_NUMBER>"                 << TKFECconfig_[i].getCrate()          << "</CRATE_NUMBER>"<< std::endl ;
    *outstream << "   <VME_ADDR>"      << "0x" << hex << TKFECconfig_[i].getAddress() << dec << "</VME_ADDR>"    << std::endl ;
    *outstream << "  </DATA>"                                                                		         << std::endl ;
  }
}

//=============================================================================================
void PixelTKFECConfig::writeXMLTrailer(std::ofstream *outstream,
                             	       std::ofstream *out1stream,
                             	       std::ofstream *out2stream ) const 
{
  std::string mthn = "[PixelTKFECConfig::writeXMLTrailer()]\t\t\t    " ;
  
  *outstream << " </DATA_SET>"		 								  << std::endl ;
  *outstream << "</ROOT>"  		 								  << std::endl ;
  
  outstream->close() ;
  std::cout << mthn << "Data written "   								  << std::endl ;

}
