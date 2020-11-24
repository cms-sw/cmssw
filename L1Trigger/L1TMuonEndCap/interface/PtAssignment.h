#ifndef L1TMuonEndCap_PtAssignment_h
#define L1TMuonEndCap_PtAssignment_h

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"

class PtAssignmentEngine;
class PtAssignmentEngineDxy;
class PtAssignmentEngineAux;

class PtAssignment {
public:
  void configure(PtAssignmentEngine* pt_assign_engine,
                 PtAssignmentEngineDxy* pt_assign_engine_dxy,
                 int verbose,
                 int endcap,
                 int sector,
                 int bx,
                 bool readPtLUTFile,
                 bool fixMode15HighPt,
                 bool bug9BitDPhi,
                 bool bugMode7CLCT,
                 bool bugNegPt,
                 bool bugGMTPhi,
                 bool promoteMode7,
                 int modeQualVer);

  void process(EMTFTrackCollection& best_tracks);

  const PtAssignmentEngineAux& aux() const;

private:
  PtAssignmentEngine* pt_assign_engine_;

  PtAssignmentEngineDxy* pt_assign_engine_dxy_;

  int verbose_, endcap_, sector_, bx_;

  bool bugGMTPhi_, promoteMode7_;
  int modeQualVer_;
};

#endif
