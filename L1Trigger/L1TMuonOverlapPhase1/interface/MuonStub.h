/*
 * MuonStub.h
 *
 *  Created on: Dec 21, 2018
 *      Author: kbunkow
 *
 *      MuonStub - data structure for algorithm input
 */

#ifndef L1T_OmtfP1_MUONSTUB_H_
#define L1T_OmtfP1_MUONSTUB_H_

#include <vector>
#include <memory>

struct MuonStub {
public:
  enum Type {
    EMPTY,
    DT_PHI,
    DT_THETA,
    DT_PHI_ETA,
    DT_HIT,
    RPC,
    RPC_DROPPED,  //to mark that all clusters were dropped because there are more than 2 clusters or at least one too big cluster
    CSC_PHI,
    CSC_ETA,
    CSC_PHI_ETA,
    BARREL_SUPER_SEG,
  };

  MuonStub();

  MuonStub(int phiHw, int phiBHw) : phiHw(phiHw), phiBHw(phiBHw){};

  virtual ~MuonStub();

  Type type = EMPTY;

  int phiHw = 0;
  int phiBHw = 0;

  static const int EMTPY_PHI = 0xffffff;

  int etaHw = 0;
  int etaSigmaHw = 0;  ///error of the eta measurement
  int qualityHw = 0;

  int bx = 0;
  int timing = 0;

  //used to address LUTs
  unsigned int logicLayer = 0;

  //int roll = 0;  //TODO remove

  int detId = 0;

  friend std::ostream& operator<<(std::ostream& out, const MuonStub& stub);
};

typedef std::vector<MuonStub> MuonStubs1D;
typedef std::vector<MuonStubs1D> MuonStubs2D;

typedef std::shared_ptr<MuonStub> MuonStubPtr;
typedef std::vector<MuonStubPtr> MuonStubPtrs1D;
typedef std::vector<MuonStubPtrs1D> MuonStubPtrs2D;

#endif /* L1T_OmtfP1_MUONSTUB_H_ */
