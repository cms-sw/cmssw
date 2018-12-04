#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include <string>

namespace MTDTopologyMode {

  Mode MTDStringToEnumParser( const std::string &value ) {

    std::string prefix("MTDTopologyMode::");
    if ( value == prefix+"tile" ) { return Mode::tile; }
    else if ( value == prefix+"bar" ) { return Mode::bar; }
    else if ( value == prefix+"barzflat" ) { return Mode::barzflat; }
    else { throw cms::Exception( "MTDTopologyModeError" ) 
        << "the value " << value << " is not defined."; 
      return Mode::undefined; }
    
  }

}
