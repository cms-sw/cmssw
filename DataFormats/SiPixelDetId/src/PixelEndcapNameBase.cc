#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameBase.h"

std::ostream & operator<<( std::ostream& out, const PixelEndcapNameBase::HalfCylinder& t)
{
  switch (t) {
    case(PixelEndcapNameBase::pI) : {out << "pI"; break;}
    case(PixelEndcapNameBase::pO) : {out << "pO"; break;}
    case(PixelEndcapNameBase::mI) : {out << "mI"; break;}
    case(PixelEndcapNameBase::mO) : {out << "mO"; break;}
    default: out << "unknown";
  };
  return out;
}

