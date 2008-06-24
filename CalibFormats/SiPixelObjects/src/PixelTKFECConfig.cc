//
// This class stores the information about a TKFEC.
// This include the number, crate, and base address
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelTKFECConfig.h"
#include <fstream>
#include <map>
#include <assert.h>


using namespace pos;
using namespace std;


PixelTKFECConfig::PixelTKFECConfig(std::vector<std::vector<std::string> >& tableMat ) : PixelConfigBase(" "," "," ")
{
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  /**

  View's name: CONF_KEY_TRACKER_FEC_CONFIG_MV
  CONFIG_KEY_ID				   NOT NULL NUMBER(38)
  CONFIG_KEY				   NOT NULL VARCHAR2(80)
  VERSION					    VARCHAR2(40)
  KIND_OF_COND				   NOT NULL VARCHAR2(40)
  TRACKER_FEC				   NOT NULL VARCHAR2(200)
  CRATE					   NOT NULL NUMBER(38)
  SLOT_NUMBER				   NOT NULL NUMBER(38)
  VME_ADDRS_HEX					    VARCHAR2(17)

  */

  colNames.push_back("CONFIG_KEY_ID");
  colNames.push_back("CONFIG_KEY"   );
  colNames.push_back("VERSION"	    );
  colNames.push_back("KIND_OF_COND" );
  colNames.push_back("TRACKER_FEC"  );
  colNames.push_back("CRATE"	    );
  colNames.push_back("SLOT_NUMBER"  );
  colNames.push_back("VME_ADDRS_HEX");
  
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
	  std::cerr << "[PixelTKFECConfig::PixelTKFECConfig()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }
  
  for(unsigned int r = 1 ; r < tableMat.size() ; r++)    //Goes to every row of the Matrix
    {
      std::string TKFECID  = tableMat[r][colM["TRACKER_FEC"]] ;
      unsigned int crate   = atoi(tableMat[r][colM["CRATE"]].c_str()) ;
      std::string type     = "VME" ;
      unsigned int address = strtoul(tableMat[r][colM["VME_ADDRESS_HEX"]].c_str() , 0, 16);
      PixelTKFECParameters tmp;
      tmp.setTKFECParameters(TKFECID , crate , type, address);
      TKFECconfig_.push_back(tmp);
    }
}// end contructor

//****************************************************************************************

 
PixelTKFECConfig::PixelTKFECConfig(std::string filename):
    PixelConfigBase(" "," "," "){

    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << "Could not open:"<<filename<<std::endl;
	assert(0);
    }
    else {
	std::cout << "Opened:"<<filename<<std::endl;
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

