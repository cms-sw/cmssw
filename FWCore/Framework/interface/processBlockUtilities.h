#ifndef FWCore_Framework_processBlockUtilities_h
#define FWCore_Framework_processBlockUtilities_h

/**

\author W. David Dagenhart, created 13 January, 2021

*/

#include <string>

namespace edm {

  class Event;

  unsigned int eventProcessBlockIndex(Event const& event, std::string const& processName);

}  // namespace edm
#endif
