#ifndef BMTrackFinder_L1BMTrackCollection_h
#define BMTrackFinder_L1BMTrackCollection_h

#include <vector>
#include <L1Trigger/L1TMuonBarrel/interface/L1MuBMTrack.h>
#include <L1Trigger/L1TMuonBarrel/src/L1MuBMTrackSegEta.h>
#include <L1Trigger/L1TMuonBarrel/src/L1MuBMTrackSegPhi.h>

typedef std::pair<L1MuBMTrack, std::vector<L1MuBMTrackSegPhi> > L1BMTrack;
typedef std::vector<L1BMTrack> L1BMTrackCollection;

#endif

