#ifndef TrackReco_print_h
#define TrackReco_print_h
/* Track print utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Track.h,v 1.16 2006/04/19 13:35:05 llista Exp $
 * 
 */
#include <string>

namespace reco {
  class Track;
  /// Track print utility

  enum verbosity { coincise, normal, detailed };

  std::string print( const Track &, verbosity = detailed );
}

#endif
