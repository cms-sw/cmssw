#ifndef Geometry_MTDCommonData_MTDTopologyMode_H
#define Geometry_MTDCommonData_MTDTopologyMode_H

#include "FWCore/Utilities/interface/Exception.h"
#include <map>
#include <string>
#include <algorithm>

template< typename T >
class MTDStringToEnumParser {
  std::map< std::string, T > enumMap;
 public:
     
  MTDStringToEnumParser( void );
 
  T parseString( const std::string &value )  { 
    typename std::map<std::string, T>::const_iterator iValue = enumMap.find( value );
    if (iValue  == enumMap.end())
      throw cms::Exception( "Configuration" )
        << "the value " << value << " is not defined.";
       
    return iValue->second;
  }
};
 
namespace MTDTopologyMode {
  enum Mode { tile=1, bar=2, barzflat=3 };
}
 
#endif // Geometry_MTDCommonData_MTDTopologyMode_H
