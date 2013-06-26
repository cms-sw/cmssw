#include "CalibFormats/SiPixelObjects/interface/PixelFECParameters.h"
#include <ostream>


using namespace pos;

PixelFECParameters::PixelFECParameters()
{
fecnumber_=0;
crate_=0;
vmebaseaddress_=0;

}

PixelFECParameters::~PixelFECParameters()
{

}


unsigned int PixelFECParameters::getFECNumber() const{

  
    return fecnumber_;

}


unsigned int PixelFECParameters::getCrate() const{

    
    return crate_;

}


unsigned int PixelFECParameters::getVMEBaseAddress() const{

    
    return vmebaseaddress_;

}

void PixelFECParameters::setFECParameters( unsigned int fecnumber , unsigned int crate , unsigned int vmebaseaddress){

fecnumber_ = fecnumber;
crate_ = crate;
vmebaseaddress_ =vmebaseaddress;

}

void PixelFECParameters::setFECNumber(unsigned int fecnumber) {

fecnumber_ = fecnumber;

}


void PixelFECParameters::setCrate(unsigned int crate) {

crate_=crate;

}

void PixelFECParameters::setVMEBaseAddress(unsigned int vmebaseaddress){

vmebaseaddress_=vmebaseaddress;

}

std::ostream&  pos::operator <<(std::ostream& s ,const PixelFECParameters &pFECp){

s <<"FEC Number:"<<pFECp.fecnumber_<<std::endl;
s <<"Crate Number:"<<pFECp.crate_<<std::endl;
s <<"VME Base Address:"<<pFECp.vmebaseaddress_<<std::endl;

return s;

}



