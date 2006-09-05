#ifndef SiPixelObjects_ModuleType_H 
#define SiPixelObjects_ModuleType_H

#include <iostream>
namespace sipixelobjects {
  enum ModuleType { v1x2, v1x5, v1x8, v2x3, v2x4, v2x5, v2x8 } ;
}
std::ostream & operator<<( std::ostream& out, const sipixelobjects::ModuleType & t);
#endif
