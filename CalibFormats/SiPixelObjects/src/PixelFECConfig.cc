//
// This class stores the information about a FEC.
// This include the number, crate, and base address
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelFECConfig.h"
#include <fstream>
#include <map>
#include <assert.h>

using namespace pos;
using namespace std;



PixelFECConfig::PixelFECConfig(std::vector<std::vector<std::string> >& tableMat ) : PixelConfigBase(" "," "," "){

 std::map<std::string , int > colM;
 std::vector<std::string > colNames;
 /**
    CONFIG_KEY_ID                             NOT NULL NUMBER(38)
    CONFIG_KEY                                NOT NULL VARCHAR2(80)
    VERSION                                            VARCHAR2(40)
    KIND_OF_COND                              NOT NULL VARCHAR2(40)
    PIXEL_FEC                                 NOT NULL VARCHAR2(200)
    CRATE                                     NOT NULL NUMBER(38)
    SLOT_NUMBER                               NOT NULL NUMBER(38)
    VME_ADDRS_HEX                                      VARCHAR2(17)
 */

 colNames.push_back("CONFIG_KEY_ID"  );
 colNames.push_back("CONFIG_KEY"     );
 colNames.push_back("VERSION"        );
 colNames.push_back("KIND_OF_COND"   );
 colNames.push_back("PIXEL_FEC"      );
 colNames.push_back("CRATE"          );
 colNames.push_back("SLOT_NUMBER"    );
 colNames.push_back("VME_ADDRS_HEX"  );


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
 for(unsigned int n=0; n<colNames.size(); n++)
   {
     if(colM.find(colNames[n]) == colM.end()){
       std::cerr << "[PixelFECConfig::PixelFECConfig()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
       assert(0);
     }
   }

 fecconfig_.clear();
 for(unsigned int r = 1 ; r < tableMat.size() ; r++)    //Goes to every row of the Matrix
   {
     unsigned int fecnumber;
     unsigned int crate;
     unsigned int vme_base_address;
     
//      01234567890123
//      BPix_Pxl_FEC_1
     string fullFECName = tableMat[r][colM["PIXEL_FEC"]] ;
     fullFECName.replace(0,13,"") ;
     fecnumber = atoi(fullFECName.c_str()) ;
     crate     = atoi(tableMat[r][colM["CRATE"]].c_str()) ;
     string hexVMEAddr = tableMat[r][colM["VME_ADDRS_HEX"]] ;
     sscanf(hexVMEAddr.c_str(), "%x", &vme_base_address) ;
     PixelFECParameters tmp;
     
     tmp.setFECParameters(fecnumber , crate , vme_base_address);
     
     fecconfig_.push_back(tmp);
   }
 
}// end contructor

//****************************************************************************************

 
PixelFECConfig::PixelFECConfig(std::string filename):
    PixelConfigBase(" "," "," "){

    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << "[PixelFECConfig::PixelFECConfig()]\t\t\t    Could not open: "<<filename<<std::endl;
	assert(0);
    }
    else {
	std::cout << "[PixelFECConfig::PixelFECConfig()]\t\t\t    Opened: "<<filename<<std::endl;
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
	    //std::cout << fecnumber <<" "<< crate << " "  
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

    for(unsigned int i=0;i<fecconfig_.size();i++){
	if (fecconfig_[i].getFECNumber()==fecnumber) return fecconfig_[i].getCrate();
    }

    std::cout << "Could not find FEC number:"<<fecnumber<<std::endl;

    assert(0);

    return 0;

}

unsigned int PixelFECConfig::VMEBaseAddressFromFECNumber(unsigned int fecnumber) const{

    for(unsigned int i=0;i<fecconfig_.size();i++){
	if (fecconfig_[i].getFECNumber()==fecnumber) return fecconfig_[i].getVMEBaseAddress();
    }

    std::cout << "Could not find FEC number:"<<fecnumber<<std::endl;

    assert(0);

    return 0;

}

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
