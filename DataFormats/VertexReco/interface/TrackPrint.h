#ifndef TrackPrint_h
#define TrackPrint_h
#include "DataFormats/TrackReco/interface/Track.h"
#include <iostream>


std::ostream& operator<<(std::ostream& os, const reco::Track & tk);

#endif
