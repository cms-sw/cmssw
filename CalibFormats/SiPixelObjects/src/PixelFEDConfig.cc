//
// This class stores the information about a FED.
// This include the number, crate, and base address
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelFEDConfig.h"
#include <fstream>
#include <iostream>
#include <map>
#include <assert.h>

using namespace pos;
using namespace std;

PixelFEDConfig::PixelFEDConfig(std::vector<std::vector<std::string> >& tableMat ) : PixelConfigBase(" "," "," "){
  std::vector< std::string > ins = tableMat[0];
  std::map<std::string , int > colM;
   std::vector<std::string > colNames;
   colNames.push_back("FED");//0
   colNames.push_back("CRATE");//1
   colNames.push_back("VME_ADDRS");//2

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

  std::string fedname = "";
  unsigned int fednum = 0;
  fedconfig_.clear();
  bool flag = false;
  for(unsigned int r = 1 ; r < tableMat.size() ; r++){    //Goes to every row of the Matrix

    fedname = tableMat[r][colM[colNames[0]]]; //This is not going to work if you change in the database "PxlFed_#" in the FED column.Im removing "PlxFed_" and store the number
    //becuase the PixelFecConfig class ask for the fec number not the name.  
    fedname.erase(0,7); 
    fednum = (unsigned int)atoi(fedname.c_str()) ;
  
    if(fedconfig_.empty()){
  
      PixelFEDParameters tmp;
  
      tmp.setFEDParameters( fednum, (unsigned int)atoi(tableMat[r][colM[colNames[1]]].c_str()) , 
			    (unsigned int)atoi(tableMat[r][colM[colNames[2]]].c_str()));   
  
      fedconfig_.push_back(tmp);
  
    }
    else{
 
      for( unsigned int y = 0; y < fedconfig_.size() ; y++){
	if (fedconfig_[y].getFEDNumber() == fednum){    // This is for check is they are Pixel Feds already in the vector because
	  // in the view of the database that I'm reading are repeated.
	  flag =true;					// This ensure that the are no objects in the fecconfig vector with repeated values.
	  break;
	}else flag= false;
      }
   
      if(flag == false){
	PixelFEDParameters tmp;
  
	tmp.setFEDParameters( fednum, (unsigned int)atoi(tableMat[r][colM[colNames[1]]].c_str()) , 
			      (unsigned int)atoi(tableMat[r][colM[colNames[2]]].c_str()));   
  
	fedconfig_.push_back(tmp); 
      }
  
    }//end else 
  
  }//end for r
  
  std::cout<<std::endl;

  for( unsigned int x = 0 ; x < fedconfig_.size() ; x++){
    std::cout<<fedconfig_[x]<<std::endl;

  }

  std::cout<<fedconfig_.size()<<std::endl;

}//end Constructor



//*****************************************************************************************************




PixelFEDConfig::PixelFEDConfig(std::string filename):
  PixelConfigBase(" "," "," "){

    std::ifstream in(filename.c_str());

    if (!in.good()){
      std::cout << "Could not open:"<<filename.c_str()<<std::endl;
      assert(0);
    }
    else {
      std::cout << "Opened:"<<filename.c_str()<<std::endl;
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
	//	std::cout << std::dec << fednumber <<" "<< crate << " 0x"  
	//   << std::hex << vme_base_address<<std::dec<<std::endl;
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

  if (dir!="") dir+="/";
  string filename=dir+"fedconfig.dat";

  ofstream out(filename.c_str());
  if(!out.good()){
    cout << "Could not open file:"<<filename<<endl;
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

  for(unsigned int i=0;i<fedconfig_.size();i++){
    if (fedconfig_[i].getFEDNumber()==fednumber) return fedconfig_[i].getCrate();
  }

  std::cout << "Could not find FED number:"<<fednumber<<std::endl;

  assert(0);

  return 0;

}


unsigned int PixelFEDConfig::VMEBaseAddressFromFEDNumber(unsigned int fednumber) const{

  for(unsigned int i=0;i<fedconfig_.size();i++){
    if (fedconfig_[i].getFEDNumber()==fednumber) return fedconfig_[i].getVMEBaseAddress();
  }

  std::cout << "Could not find FED number:"<<fednumber<<std::endl;

  assert(0);

  return 0;

}

unsigned int PixelFEDConfig::FEDNumberFromCrateAndVMEBaseAddress(unsigned int crate, unsigned int vmebaseaddress) const {

  for(unsigned int i=0;i<fedconfig_.size();i++){
    if (fedconfig_[i].getCrate()==crate&&
        fedconfig_[i].getVMEBaseAddress()==vmebaseaddress) return fedconfig_[i].getFEDNumber();
  }

  std::cout << "Could not find FED crate and address:"<<crate<<", "<<vmebaseaddress<<std::endl;

  assert(0);

  return 0;

}
