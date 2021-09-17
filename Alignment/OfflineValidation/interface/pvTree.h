#ifndef Alignment_OfflineValidation_pvTree_h
#define Alignment_OfflineValidation_pvTree_h

#include "TROOT.h"
#include "TMath.h"
#include <vector>
#include <string>

class pvCand {
public:
  int nTrks;
  int ipos;

  float x_origVtx;
  float y_origVtx;
  float z_origVtx;

  float xErr_origVtx;
  float yErr_origVtx;
  float zErr_origVtx;

  int n_subVtx1;
  float x_subVtx1;
  float y_subVtx1;
  float z_subVtx1;

  float xErr_subVtx1;
  float yErr_subVtx1;
  float zErr_subVtx1;
  float sumPt_subVtx1;

  int n_subVtx2;
  float x_subVtx2;
  float y_subVtx2;
  float z_subVtx2;

  float xErr_subVtx2;
  float yErr_subVtx2;
  float zErr_subVtx2;
  float sumPt_subVtx2;

  float CL_subVtx1;
  float CL_subVtx2;

  float minW_subVtx1;
  float minW_subVtx2;

  pvCand(){};
  virtual ~pvCand(){};

  ClassDef(pvCand, 1)
};

class pvEvent {
public:
  int runNumber;
  int luminosityBlockNumber;
  int eventNumber;

  int nVtx;

  std::vector<pvCand> pvs;

  pvEvent(){};
  virtual ~pvEvent(){};

  ClassDef(pvEvent, 1)
};

#endif
