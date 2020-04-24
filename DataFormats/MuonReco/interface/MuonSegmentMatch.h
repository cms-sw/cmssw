#ifndef MuonReco_MuonSegmentMatch_h
#define MuonReco_MuonSegmentMatch_h

#include <cmath>

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMSegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h"

namespace reco {
   class MuonSegmentMatch {
      public:
         /// segment mask flags
         static const unsigned int Arbitrated                = 1<<8;     // is arbitrated (multiple muons)
         static const unsigned int BestInChamberByDX         = 1<<9;     // best delta x in single muon chamber
         static const unsigned int BestInChamberByDR         = 1<<10;    // best delta r in single muon chamber
         static const unsigned int BestInChamberByDXSlope    = 1<<11;    // best delta dx/dz in single muon chamber
         static const unsigned int BestInChamberByDRSlope    = 1<<12;    // best delta dy/dz in single muon chamber
         static const unsigned int BestInStationByDX         = 1<<13;    // best delta x in single muon station
         static const unsigned int BestInStationByDR         = 1<<14;    // best delta r in single muon station
         static const unsigned int BestInStationByDXSlope    = 1<<15;    // best delta dx/dz in single muon station
         static const unsigned int BestInStationByDRSlope    = 1<<16;    // best delta dy/dz in single muon station
         static const unsigned int BelongsToTrackByDX        = 1<<17;    // best delta x of multiple muons
         static const unsigned int BelongsToTrackByDR        = 1<<18;    // best delta r of multiple muons
         static const unsigned int BelongsToTrackByDXSlope   = 1<<19;    // best delta dx/dz of multiple muons
         static const unsigned int BelongsToTrackByDRSlope   = 1<<20;    // best delta dy/dz of multiple muons
         static const unsigned int BelongsToTrackByME1aClean = 1<<21;    // won ME1a segment sharing cleaning
         static const unsigned int BelongsToTrackByOvlClean  = 1<<22;    // won chamber overlap segment sharing cleaning
         static const unsigned int BelongsToTrackByClusClean = 1<<23;    // won cluster sharing cleaning
         static const unsigned int BelongsToTrackByCleaning  = 1<<24;    // won any arbitration cleaning type, including defaults

         float x;              // X position of the matched segment
         float y;              // Y position of the matched segment
         float xErr;           // uncertainty in X
         float yErr;           // uncertainty in Y
         float dXdZ;           // dX/dZ of the matched segment
         float dYdZ;           // dY/dZ of the matched segment
         float dXdZErr;        // uncertainty in dX/dZ
         float dYdZErr;        // uncertainty in dY/dZ
         unsigned int mask;    // arbitration mask
         bool hasZed_;         // contains local y information (only relevant for segments in DT)
         bool hasPhi_;         // contains local x information (only relevant for segments in DT)

         bool isMask( unsigned int flag = Arbitrated ) const { return (mask & flag) == flag; }
         void setMask( unsigned int flag ) { mask |= flag; }
         float t0;

         DTRecSegment4DRef  dtSegmentRef;
         CSCSegmentRef      cscSegmentRef;
	 GEMSegmentRef      gemSegmentRef;
	 ME0SegmentRef      me0SegmentRef;
      MuonSegmentMatch():x(0),y(0),xErr(0),yErr(0),dXdZ(0),dYdZ(0),
      dXdZErr(0),dYdZErr(0) {}

         bool hasZed() const { return hasZed_; }
         bool hasPhi() const { return hasPhi_; }
   };
}

#endif
