#include "CondFormats/SiPixelObjects/interface/ModuleType.h"

using namespace sipixelobjects;
std::ostream & operator<<( std::ostream& out, const ModuleType & t)
{
  switch (t) {
    case(v1x2) : {out << "v1x2"; break;}
    case(v1x5) : {out << "v1x5"; break;}
    case(v1x8) : {out << "v1x8"; break;}
    case(v2x3) : {out << "v2x3"; break;}
    case(v2x4) : {out << "v2x4"; break;}
    case(v2x5) : {out << "v2x5"; break;}
    case(v2x8) : {out << "v2x8"; break;}
    default: out << "unknown"; 
  };
  return out;
}
