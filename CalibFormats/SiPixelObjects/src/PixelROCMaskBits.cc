//
// This class provide the data structure for the
// ROC DAC parameters
//
// At this point I do not see a reason to make an
// abstract layer for this code.
//

#include "CalibFormats/SiPixelObjects/interface/PixelROCMaskBits.h"
#include "CalibFormats/SiPixelObjects/interface/PixelBase64.h"
#include <iostream>
#include <cassert>
#include <typeinfo>

using namespace pos;

//=====================================================================/
PixelROCMaskBits::PixelROCMaskBits(){
}


/**********************Start Modification******************************/
void PixelROCMaskBits::setROCMaskBits(PixelROCName& rocid ,std::string bits)
{
  std::string mthn = "[PixelROCMaskBits::setROCMaskBits()]\t\t\t    " ;
  rocid_=rocid;
  char cpt[520] ;
  bits.copy( cpt , 520);
  for(unsigned int i = 0 ; i < bits.size(); i++)
    {
      bits_[i] = static_cast<unsigned char>(cpt[i]);
      //      std::cout<< "bits_[" << i << "]\t" << bits_[i] <<std::endl;
      //      std::cout<<rocid_<<std::endl;
      //      std::cout.flags(std::ios::hex)
    }
}
/**********************End Modification******************************/

// Added by Dario: handles the base_64-decoded strings from aDB read
int PixelROCMaskBits::read(const PixelROCName& rocid, std::string in){
 rocid_=rocid;
 for( int i=0; i<(int)sizeof(bits_); i++)
 {
  bits_[i] = in.at(i) ;
 }
 return 1 ;
}

//=====================================================================/
int PixelROCMaskBits::read(const PixelROCName& rocid, std::ifstream& in){

    rocid_=rocid;

    std::string tag;

    for (int i=0;i<52;i++){
    
      in >> tag;
      
      //std::cout << "Now reading col:"<<tag<<std::endl;

      std::string data;

      in >> data;

      //std::cout <<"data.size()" <<data.size()<<std::endl;

      unsigned char byte=0;

      for(int j=0;j<80;j++){
	 
	if (data[j]=='1') byte+=128;
	
	if ((j+1)%8==0) {
	    //std::cout << "Writing byte:"<<(int)byte<<std::endl;
	  bits_[i*10+(j+1)/8-1]=byte;
	  byte=0; 
	}
	else{
	  byte/=2;
	}
	

      }

    }

    return 1;

}

//=====================================================================/
// modified by MR on 23-06-2008 11:57:58
int PixelROCMaskBits::read(const PixelROCName& rocid, std::istringstream& in)
{
  rocid_=rocid;
  std::string tag;
  for (int i=0;i<52;i++)
    {
      in >> tag;
      //std::cout << "Now reading col:"<<tag<<std::endl;
      std::string data;
      in >> data;
      //std::cout <<"data.size()" <<data.size()<<std::endl;
      unsigned char byte=0;
      for(int j=0;j<80;j++)
	{
	if (data[j]=='1') byte+=128;
	if ((j+1)%8==0) 
	  {
	    //std::cout << "Writing byte:"<<(int)byte<<std::endl;
	    bits_[i*10+(j+1)/8-1]=byte;
	    byte=0; 
	  }
	else
	  {
	    byte/=2;
	  }
      }
    }
  return 1;
}

//=====================================================================/
int PixelROCMaskBits::readBinary(const PixelROCName& rocid, std::ifstream& in){

    rocid_=rocid;


    in.read((char*)bits_,520);


    return 1;
}

//=====================================================================/
void PixelROCMaskBits::writeBinary(std::ofstream& out) const{

    out << (char)rocid_.rocname().size();
    out.write(rocid_.rocname().c_str(),rocid_.rocname().size());

    for(unsigned int i=0;i<520;i++){
	out << bits_[i];
    }

}

//=====================================================================/
void PixelROCMaskBits::writeASCII(std::ofstream& out) const{

    out << "ROC:    "<<rocid_.rocname()<<std::endl;
 
    for(unsigned int col=0;col<52;col++){
	out << "col";
	if (col<10) out << "0";
	out <<col<<":  ";
	for (int row=0;row<80;row++){
	    out << mask(col,row);
	}
	out << std::endl;
    }

}

//=====================================================================/
unsigned int PixelROCMaskBits::mask(unsigned int col, unsigned int row) const{

  unsigned int tmp=bits_[col*10+row/8];
//   std::cout << "c =  " << col << "\tr = " << row << "\tbits_[" << (col*10+row/8) << "]=" << bits_[col*10+row/8] << std::endl ;
//   std::cout << "[PixelROCMaskBits::mask()] tmp iniziale " << tmp      << std::endl ;
  tmp=tmp>>(row%8);				        	
//   std::cout << "[PixelROCMaskBits::mask()] tmp finale   " << tmp      << std::endl ;
//   unsigned int res = tmp&0x01 ;			        	
//   std::cout << "[PixelROCMaskBits::mask()] return value " << res      << std::endl ;
  return tmp&0x01;

}

//=====================================================================/
void PixelROCMaskBits::setMask(unsigned int col, unsigned int row, unsigned int mask){

  assert(mask==0||mask==1);  

  unsigned int bit=1<<(row%8);
  if (mask) bits_[col*10+row/8]=bits_[col*10+row/8]|bit;  
  if (!mask) bits_[col*10+row/8]=bits_[col*10+row/8]&(0xff^bit);

}

//=====================================================================/
std::ostream& pos::operator<<(std::ostream& s, const PixelROCMaskBits& mask){

  s << "Dumping ROC masks" <<std::endl; 

  for(int i=0;i<52;i++){
    s<<"Col"<<i<<":";
    for(int j=0;j<10;j++){
      unsigned char bitmask=1;
      for(int k=0;k<8;k++){
	if(mask.bits_[i*10+j]&bitmask) {
	  s << "1";
	}
	else{
	  s << "0";
	}
	bitmask*=2;
      }
    }
    s<<std::endl;
  }


  return s;
  
}

//=============================================================================================
void PixelROCMaskBits::writeXML(std::ofstream * out) const
{
  std::string mthn = "[PixelROCMaskBits::writeXML()]\t\t\t\t" ;

  std::string encoded = base64_encode(bits_, sizeof(bits_));

  *out << "  <DATA>"						 << std::endl ;
  *out << "   <ROC_NAME>"  << rocid_.rocname() << "</ROC_NAME>"  << std::endl ;
  *out << "   <KILL_MASK>" << encoded	       << "</KILL_MASK>" << std::endl ;
  *out << "  </DATA>"						 << std::endl ;
  *out << " "  						         << std::endl ;
      
}


