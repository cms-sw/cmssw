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


PixelTKFECConfig::PixelTKFECConfig(std::vector<std::vector<std::string> >& tableMat ) : PixelConfigBase(" "," "," "){

assert(0); // This function needs to be updated.

 /*std::vector< std::string > ins = tableMat[0];
 std::map<std::string , int > colM;
 std::vector<std::string > colNames;
 colNames.push_back("TKFEC_NAME");//0
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
     std::cerr << "[PixelTKFECConfig::PixelTKFECConfig()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
     assert(0);
   }
 }

std::string TKFECname = "";
unsigned int TKFECnum = 0;
TKFECconfig_.clear();
bool flag = false;
for(unsigned int r = 1 ; r < tableMat.size() ; r++){    //Goes to every row of the Matrix
  

  TKFECname = tableMat[r][colM[colNames[0]]]; //This is not going to work if you change in the database "Pix TKFEC #"  Im removing "Pix TKFEC" and store the number
                        //becuase the PixelTKFECConfig class ask for the TKFEC number not the name.  
  TKFECname.erase(0,8); 
  TKFECnum = (unsigned int)atoi(TKFECname.c_str()) ;
  
  if(TKFECconfig_.empty()){
  
  PixelTKFECParameters tmp;
  
  tmp.setTKFECParameters( TKFECnum, (unsigned int)atoi(tableMat[r][colM[colNames[1]]].c_str()) , 
  (unsigned int)atoi(tableMat[r][colM[colNames[2]]].c_str()));   
  
   TKFECconfig_.push_back(tmp);
  
  }
  else{
 
 for( unsigned int y = 0; y < TKFECconfig_.size() ; y++){
   if (TKFECconfig_[y].getTKFECID() == TKFECnum){    // This is for check is they are Pixel TKFECs already in the vector because
                                                 // in the view of the database that I'm reading are repeated.
   flag =true;					// This ensure that the are no objects in the TKFECconfig vector with repeated values.
   break;
   }else flag= false;
  }
   
   if(flag == false){
  PixelTKFECParameters tmp;
  
  tmp.setTKFECParameters( TKFECnum, (unsigned int)atoi(tableMat[r][colM[colNames[1]]].c_str()) , 
  (unsigned int)atoi(tableMat[r][colM[colNames[2]]].c_str()));   
  
  TKFECconfig_.push_back(tmp); 
  }
  
  }//end else 
  
  }//end for r
  
std::cout<<std::endl;

for( unsigned int x = 0 ; x < TKFECconfig_.size() ; x++){
     std::cout<<TKFECconfig_[x]<<std::endl;

}

std::cout<<TKFECconfig_.size()<<std::endl;*/

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

