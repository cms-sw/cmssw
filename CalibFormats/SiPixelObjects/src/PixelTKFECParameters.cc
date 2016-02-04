#include "CalibFormats/SiPixelObjects/interface/PixelTKFECParameters.h"
#include <ostream>

using namespace pos;

PixelTKFECParameters::PixelTKFECParameters()
{
TKFECID_="";
crate_=0;
type_="";
address_=0;

}

PixelTKFECParameters::~PixelTKFECParameters()
{

}


std::string PixelTKFECParameters::getTKFECID() const{
    return TKFECID_;
}

unsigned int PixelTKFECParameters::getCrate() const{
    return crate_;
}

std::string PixelTKFECParameters::getType() const{
    return type_;
}

unsigned int PixelTKFECParameters::getAddress() const{
    return address_;
}

void PixelTKFECParameters::setTKFECParameters( std::string TKFECID , unsigned int crate , std::string type, unsigned int address){

TKFECID_ = TKFECID;
crate_ = crate;
type_ = type;
address_ =address;

}

void PixelTKFECParameters::setTKFECID(std::string TKFECID) {

TKFECID_ = TKFECID;

}

void PixelTKFECParameters::setCrate(unsigned int crate) {

crate_=crate;

}

void PixelTKFECParameters::setType(std::string type) {

type_ = type;

}

void PixelTKFECParameters::setAddress(unsigned int address){

address_=address;

}

std::ostream&  pos::operator <<(std::ostream& s ,const PixelTKFECParameters &pTKFECp){

s <<"TKFEC ID:"<<pTKFECp.TKFECID_<<std::endl;
s <<"Crate Number:"<<pTKFECp.crate_<<std::endl;
s <<pTKFECp.type_<<std::endl;
s <<"Address:"<<pTKFECp.address_<<std::endl;

return s;

}



