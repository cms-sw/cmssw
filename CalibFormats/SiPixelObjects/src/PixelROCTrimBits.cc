//
// This class provide the data structure for the
// ROC DAC parameters
//
// At this point I do not see a reason to make an
// abstract layer for this code.
//

#include "CalibFormats/SiPixelObjects/interface/PixelROCTrimBits.h"
#include "CalibFormats/SiPixelObjects/interface/PixelBase64.h"
#include <iostream>
#include <sstream>
#include <cassert>
#include <typeinfo>

using namespace pos;

PixelROCTrimBits::PixelROCTrimBits()
{
}
//This part has been modified from the orignal
void PixelROCTrimBits::setROCTrimBits(PixelROCName rocid , std::string bits)
{

  rocid_=rocid;
  char cpt[2080] ;
  bits.copy( cpt , 2080);
  for(unsigned int i = 0 ; i < bits.size(); i++)
    bits_[i] = static_cast<unsigned char>(cpt[i]);
  
}
//End of part modified

// Added by Dario: handles the base_64-decoded strings from aDB read
int PixelROCTrimBits::read(const PixelROCName rocid, std::string in){
 rocid_=rocid;
 for( int i=0; i<(int)sizeof(bits_); i++)
 {
  bits_[i] = in.at(i) ;
 }
 return 1 ;
}

int PixelROCTrimBits::read(PixelROCName rocid,std::ifstream& in){
    
  std::string tag;

  //std::cout << "PixelROCTrimBits::read rocid:"<<rocid<<std::endl;

  rocid_=rocid;
  
  //std::cout << "PixelROCTrimBits::read rocid_:"<<rocid_<<std::endl;

  for (int i=0;i<52;i++){
    
    in >> tag;
    
    //std::cout << "Now reading col:"<<tag<<std::endl;
    
    std::string data;
    
    in >> data;
    
    //std::cout <<"data.size()" <<data.size()<<std::endl;

    unsigned char byte=0;

    for(int j=0;j<80;j++){

      unsigned char tmp=toupper(data[j])-48;
      if (tmp>9) tmp-=7;  //FIXME this is so ugly

      byte+=tmp;
      
      if ((j+1)%2==0) {
	  //std::cout << "Writing byte:"<<(int)byte<<std::endl;
	bits_[i*40+(j+1)/2-1]=byte;
	byte=0; 
      }
      else{
	byte*=16;
      }
	

    }

  }
  return 1;

}

int PixelROCTrimBits::read(PixelROCName rocid, std::istringstream& in)
{
  std::string tag;
  //std::cout << "PixelROCTrimBits::read rocid:"<<rocid<<std::endl;
  rocid_=rocid;
  //std::cout << "PixelROCTrimBits::read rocid_:"<<rocid_<<std::endl;
  for (int i=0;i<52;i++)
    {
      in >> tag;
//       std::cout << "Now reading col:"<<tag<<std::endl;
      std::string data;
      in >> data;
//       std::cout <<" data: " <<data<<std::endl;
      unsigned char byte=0;
      for(int j=0;j<80;j++)
	{
	  unsigned char tmp=toupper(data[j])-48;
	  if (tmp>9) tmp-=7;  //FIXME this is so ugly
	  byte+=tmp;
	  if ((j+1)%2==0) 
	    {
	      //std::cout << "Writing byte:"<<(int)byte<<std::endl;
	      bits_[i*40+(j+1)/2-1]=byte;
	      byte=0; 
	    }
	  else
	    {
	      byte*=16;
	    }
	}
    }
  return 1;
}



int PixelROCTrimBits::readBinary(PixelROCName rocid,std::ifstream& in){
    
  rocid_=rocid;

  in.read((char*)bits_,2080);

  return 1;

}


void PixelROCTrimBits::writeBinary(std::ofstream& out) const{

    out << (char)rocid_.rocname().size();
    out.write(rocid_.rocname().c_str(),rocid_.rocname().size());

    //std::cout << "PixelROCTrimBits::writeBinary:"<<rocid_.rocname().size()
    // << " " <<rocid_.rocname()<<std::endl;

    for(unsigned int i=0;i<2080;i++){
	out << bits_[i];
    }

}

void PixelROCTrimBits::writeASCII(std::ofstream& out) const{

    //std::cout << " PixelROCTrimBits::writeASCII rocid_.rocname():"<<rocid_.rocname()<<std::endl;


    out << "ROC:     "<<rocid_.rocname()<<std::endl;

    //std::cout << "PixelROCTrimBits::writeBinary:"<<rocid_.rocname().size()
    //	 << " " <<rocid_.rocname()<<std::endl;

    for(unsigned int col=0;col<52;col++){
	out << "col";
	if (col<10) out << "0";
	out <<col<<":   ";
	for (int row=0;row<80;row++){
	    out << std::hex<<std::uppercase<<trim(col,row)<<std::dec;
	}
	out << std::endl;
    }

}

unsigned int PixelROCTrimBits::trim(unsigned int col, unsigned int row) const{

  unsigned int tmp=bits_[col*40+row/2];
  if (row%2==0) tmp/=16;
  return tmp&0x0F;

}

void PixelROCTrimBits::setTrim(unsigned int col, unsigned int row, unsigned int trim){

  assert(trim<16);

  unsigned int mask=0xf0;
  if (row%2==0) {
    trim<<=4;
    mask>>=4;
  }
  unsigned int tmp=bits_[col*40+row/2];
  bits_[col*40+row/2]=(tmp&mask)|trim;

}




std::ostream& pos::operator<<(std::ostream& s, const PixelROCTrimBits& mask){

  s << "Dumping ROC masks" <<std::endl; 

  for(int i=0;i<52;i++){
    s<<"Col"<<i<<": ";
    for(int j=0;j<40;j++){
      unsigned char bitmask=15*16;
      for(int k=0;k<2;k++){
	unsigned int tmp=mask.bits_[i*40+j]&bitmask;
	if (tmp>15) tmp/=16;
	s << std::hex << tmp << std::dec;
	bitmask/=16;
      }
    }
    s<<std::endl;
  }


  return s;
  
}

//=============================================================================================
void PixelROCTrimBits::writeXML(std::ofstream * out) const
{
  std::string mthn = "[PixelROCTrimBits::writeXML()]\t\t\t\t" ;

  std::string encoded = base64_encode(bits_, sizeof(bits_));
  std::string decoded = base64_decode(encoded);

  *out << "  <DATA>"						 << std::endl ;
  *out << "   <ROC_NAME>"  << rocid_.rocname() << "</ROC_NAME>"  << std::endl ;
  *out << "   <TRIM_BITS>" << encoded	       << "</TRIM_BITS>" << std::endl ;
  *out << "  </DATA>"						 << std::endl ;
  *out << " "                                                    << std::endl ;
      
}

