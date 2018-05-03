#include "DetectorDescription/DDCMS/interface/DDParsingContext.h"

using namespace std;
using namespace cms;

void
DDParsingContext::addVector( const string& name, const VecDouble& value ) {
  numVectors.emplace( std::piecewise_construct,
		      std::forward_as_tuple( name ),
		      std::forward_as_tuple( value ));
}
