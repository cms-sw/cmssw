#include "CondFormats/EcalObjects/interface/EcalTPGLut.h"

EcalTPGLut::EcalTPGLut()
{ }

EcalTPGLut::EcalTPGLut(const EcalTPGLut & tpgLut)
{ 
  const unsigned int * lut = tpgLut.getLut() ;
  for (unsigned int i=0 ; i<1024 ; i++) lut_[i] = lut[i] ;
}

EcalTPGLut::~EcalTPGLut()
{ }

const unsigned int * EcalTPGLut::getLut() const
{ 
  return lut_ ;
}

void EcalTPGLut::setLut(const unsigned int * lut) 
{
  for (unsigned int i=0 ; i<1024 ; i++) lut_[i] = lut[i] ;
}

EcalTPGLut & EcalTPGLut::operator=(const EcalTPGLut & tpgLut) {
  const unsigned int * lut = tpgLut.getLut() ;
  for (unsigned int i=0 ; i<1024 ; i++) lut_[i] = lut[i] ;
  return *this;
}
