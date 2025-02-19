#ifndef L1CSCTrackFinder_L1CSCTrackCollection_h
#define L1CSCTrackFinder_L1CSCTrackCollection_h

#include <DataFormats/L1CSCTrackFinder/interface/L1Track.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>

typedef std::pair<csc::L1Track,CSCCorrelatedLCTDigiCollection> L1CSCTrack;
typedef std::vector<L1CSCTrack> L1CSCTrackCollection;

#endif

