#include <DataFormats/L1CSCTrackFinder/interface/L1Track.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>

typedef std::pair<csc::L1Track,CSCCorrelatedLCTDigiCollection> L1CSCTrack;
typedef std::vector<L1CSCTrack> L1CSCTrackCollection;
