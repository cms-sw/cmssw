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

 std::vector< std::string > ins = tableMat[0];
 std::map<std::string , int > colM;
 std::vector<std::string > colNames;
 colNames.push_back("FEC_NAME");//0
 colNames.push_back("CRATE");//1
 colNames.push_back("VME_ADDR");//2

for(unsigned int c = 0 ; c < ins.size() ; c++){
   for(unsigned int n=0; n<colNames.size(); n++){
     if(tableMat[0][c] == colNames[n]){
       colM[colNames[n]] = c;
       break;
     }
   }
 }//end for
 for(unsigned int n=0; n<colNames.size(); n++){
   if(colM.find(colNames[n]) == colM.end()){
     std::cerr << "[PixelFECConfig::PixelFECConfig()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
     assert(0);
   }
 }

std::string fecname = "";
unsigned int fecnum = 0;
fecconfig_.clear();
bool flag = false;
for(unsigned int r = 1 ; r < tableMat.size() ; r++){    //Goes to every row of the Matrix
  

  fecname = tableMat[r][colM[colNames[0]]]; //This is not going to work if you change in the database "Pix Fec #"  Im removing "Pix Fec" and store the number
                        //becuase the PixelFecConfig class ask for the fec number not the name.  
  fecname.erase(0,8); 
  fecnum = (unsigned int)atoi(fecname.c_str()) ;
  
  if(fecconfig_.empty()){
  
  PixelFECParameters tmp;
  
  tmp.setFECParameters( fecnum, (unsigned int)atoi(tableMat[r][colM[colNames[1]]].c_str()) , 
  (unsigned int)atoi(tableMat[r][colM[colNames[2]]].c_str()));   
  
   fecconfig_.push_back(tmp);
  
  }
  else{
 
 for( unsigned int y = 0; y < fecconfig_.size() ; y++){
   if (fecconfig_[y].getFECNumber() == fecnum){    // This is for check is they are Pixel Fecs already in the vector because
                                                 // in the view of the database that I'm reading are repeated.
   flag =true;					// This ensure that the are no objects in the fecconfig vector with repeated values.
   break;
   }else flag= false;
  }
   
   if(flag == false){
  PixelFECParameters tmp;
  
  tmp.setFECParameters( fecnum, (unsigned int)atoi(tableMat[r][colM[colNames[1]]].c_str()) , 
  (unsigned int)atoi(tableMat[r][colM[colNames[2]]].c_str()));   
  
  fecconfig_.push_back(tmp); 
  }
  
  }//end else 
  
  }//end for r
  
std::cout<<std::endl;

for( unsigned int x = 0 ; x < fecconfig_.size() ; x++){
     std::cout<<fecconfig_[x]<<std::endl;

}

std::cout<<fecconfig_.size()<<std::endl;

}// end contructor

//****************************************************************************************

 
PixelFECConfig::PixelFECConfig(std::string filename):
    PixelConfigBase(" "," "," "){

    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << "[PixelFECConfig::PixelFECConfig()]\t\t\t   Could not open: "<<filename<<std::endl;
	assert(0);
    }
    else {
	std::cout << "[PixelFECConfig::PixelFECConfig()]\t\t\t   Opened: "<<filename<<std::endl;
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
