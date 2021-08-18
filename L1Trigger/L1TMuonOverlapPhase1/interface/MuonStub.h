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
    CSC_PHI,
    CSC_ETA,
    CSC_PHI_ETA,
    BARREL_SUPER_SEG,
    TTTRACK_REF  //for ttTrack correlation algorithm with the reference stub
  };

  /*  enum EtaType {
    NO_ETA,
    CORSE,
    FINE,
    //add other if needed
  };*/

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
  //example use: 1 if phi segment uniquely assigned to eta segment
  //EtaType etaType = NO_ETA;
  //int isPhiFine = 0;

  //used to address LUTs
  unsigned int logicLayer = 0;

  //int inputNumHw = -1;

  //layer number in hardware convention
  //int layerHw = -1;
  //int subLayerHw = -1;

  //int station = -1;

  int roll = 0;

  int detId = 0;

  //float phi = 0; //radians
  //float eta = 0;

  friend std::ostream& operator<<(std::ostream& out, const MuonStub& stub);
};

typedef std::vector<MuonStub> MuonStubs1D;
typedef std::vector<MuonStubs1D> MuonStubs2D;

typedef std::shared_ptr<const MuonStub> MuonStubPtr;
typedef std::vector<MuonStubPtr> MuonStubPtrs1D;
typedef std::vector<MuonStubPtrs1D> MuonStubPtrs2D;

#endif /* L1T_OmtfP1_MUONSTUB_H_ */
