#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameBase.h"

#include <sstream>
#include <iostream>

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

std::ostream & operator<<( std::ostream& out, const PixelBarrelNameBase::Shell& t)
{
  switch (t) {
    case(PixelBarrelNameBase::pI) : {out << "pI"; break;}
    case(PixelBarrelNameBase::pO) : {out << "pO"; break;}
    case(PixelBarrelNameBase::mI) : {out << "mI"; break;}
    case(PixelBarrelNameBase::mO) : {out << "mO"; break;}
    default: out << "unknown";
  };
  return out;
}


