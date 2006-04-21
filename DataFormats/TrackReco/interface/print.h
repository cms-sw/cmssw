#ifndef TrackReco_print_h
#define TrackReco_print_h
/* Track print utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: print.h,v 1.1 2006/04/21 05:50:46 llista Exp $
 * 
 */
#include <string>

namespace reco {
  class Track;
  /// Track print utility

  enum verbosity { concise, normal, detailed };

  std::string print( const Track &, verbosity = concise );
}

#endif
