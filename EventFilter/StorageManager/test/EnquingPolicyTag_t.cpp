#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"

#include <sstream>
#include <string>
#include <vector>
#include <cassert>

using namespace std;

int main()
{

  typedef std::vector<std::string> SVec;
  SVec tags;
  tags.push_back( "Discard New" );
  tags.push_back( "Discard Old" );
  tags.push_back( "Fail If Full" );
  tags.push_back( "Max" );

  for( unsigned int i = 0; i < tags.size(); ++i )
    {
      ostringstream oss;
      oss << stor::enquing_policy::PolicyTag( i );
      assert( oss.str() == tags[i] );
    }

  return 0;
}
