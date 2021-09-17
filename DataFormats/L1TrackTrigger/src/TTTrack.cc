#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void tttrack::errorSetTrackWordBits(unsigned int theNumFitPars) {
  edm::LogError("TTTrack") << " setTrackWordBits method is called with theNumFitPars_=" << theNumFitPars
                           << " only possible values are 4/5" << std::endl;
}
