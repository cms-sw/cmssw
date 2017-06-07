#ifndef L1TMuonEndCap_PtAssignment_h
#define L1TMuonEndCap_PtAssignment_h

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"


class PtAssignmentEngine;
class PtAssignmentEngineAux;

class PtAssignment {
public:
  void configure(
      const PtAssignmentEngine* pt_assign_engine,
      int verbose, int endcap, int sector, int bx,
      int ptLUTVersion, bool readPtLUTFile, bool fixMode15HighPt,
      bool bug9BitDPhi, bool bugMode7CLCT, bool bugNegPt,
      bool bugGMTPhi
  );

  void process(
      EMTFTrackCollection& best_tracks
  );

  const PtAssignmentEngineAux& aux() const;

private:
  PtAssignmentEngine* pt_assign_engine_;

  int verbose_, endcap_, sector_, bx_;

  bool bugGMTPhi_;
};

#endif
