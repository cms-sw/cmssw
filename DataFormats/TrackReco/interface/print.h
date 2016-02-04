#ifndef TrackReco_print_h
#define TrackReco_print_h
/* Track print utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: print.h,v 1.3 2006/05/08 07:52:54 llista Exp $
 * 
 */
#include "FWCore/Utilities/interface/Verbosity.h"
#include <string>

namespace reco {
  class Track;
  /// Track print utility
  std::string print( const Track &, edm::Verbosity = edm::Concise );
}

#endif
