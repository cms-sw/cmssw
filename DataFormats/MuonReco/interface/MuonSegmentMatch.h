#ifndef MuonReco_MuonSegmentMatch_h
#define MuonReco_MuonSegmentMatch_h

namespace reco {
   class MuonSegmentMatch {
      public:
         /// segment mask flags
         static const unsigned int Arbitrated              = 1<<8;     // is arbitrated (multiple muons)
         static const unsigned int BestInChamberByDX       = 1<<9;     // best delta x in single muon chamber
         static const unsigned int BestInChamberByDR       = 1<<10;    // best delta r in single muon chamber
         static const unsigned int BestInChamberByDXSlope  = 1<<11;    // best delta dx/dz in single muon chamber
         static const unsigned int BestInChamberByDRSlope  = 1<<12;    // best delta dy/dz in single muon chamber
         static const unsigned int BestInStationByDX       = 1<<13;    // best delta x in single muon station
         static const unsigned int BestInStationByDR       = 1<<14;    // best delta r in single muon station
         static const unsigned int BestInStationByDXSlope  = 1<<15;    // best delta dx/dz in single muon station
         static const unsigned int BestInStationByDRSlope  = 1<<16;    // best delta dy/dz in single muon station
         static const unsigned int BelongsToTrackByDX      = 1<<17;    // best delta x of multiple muons
         static const unsigned int BelongsToTrackByDR      = 1<<18;    // best delta r of multiple muons
         static const unsigned int BelongsToTrackByDXSlope = 1<<19;    // best delta dx/dz of multiple muons
         static const unsigned int BelongsToTrackByDRSlope = 1<<20;    // best delta dy/dz of multiple muons

         float x;              // X position of the matched segment
         float y;              // Y position of the matched segment
         float xErr;           // uncertainty in X
         float yErr;           // uncertainty in Y
         float dXdZ;           // dX/dZ of the matched segment
         float dYdZ;           // dY/dZ of the matched segment
         float dXdZErr;        // uncertainty in dX/dZ
         float dYdZErr;        // uncertainty in dY/dZ
         unsigned int mask;    // arbitration mask

         bool isMask( unsigned int flag = Arbitrated ) const { return mask & flag; }
         void setMask( unsigned int flag ) { if(!(mask & flag)) mask += flag; }
   };
}

#endif
