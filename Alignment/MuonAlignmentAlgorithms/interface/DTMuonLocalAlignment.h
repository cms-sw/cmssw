#ifndef Alignment_MuonAlignmentAlgorithms_DTMuonLocalAlignment_h
#define Alignment_MuonAlignmentAlgorithms_DTMuonLocalAlignment_h

/** \class DTMuonLocalAlignment
 *  $Date: 2010/02/25 11:33:32 $
 *  $Revision: 1.2 $
 *  \author Luca Scodellaro <Luca.Scodellaro@cern.ch>
 */

#include "TFile.h"
#include "TTree.h"
#include <string>
#include "TChain.h"

#define MAX_SEGMENT 5
#define MAX_HIT_CHAM 14

class DTMuonLocalAlignment {
  
  public:

  DTMuonLocalAlignment();

  ~DTMuonLocalAlignment(); 

  void initNTuples(int );
  
  void setBranchAddressTree(); 
  
  std::string ntuplePath;

  int numberOfRootFiles;
 
  TChain *tali;

  TFile *f;



  //Block of variables for the tree 
  //---------------------------------------------------------
  float p, pt, eta, phi, charge;
  int nseg;
  int nphihits[MAX_SEGMENT];
  int nthetahits[MAX_SEGMENT];
  int nhits[MAX_SEGMENT];
  float xSl[MAX_SEGMENT];
  float dxdzSl[MAX_SEGMENT];
  float exSl[MAX_SEGMENT];
  float edxdzSl[MAX_SEGMENT];
  float exdxdzSl[MAX_SEGMENT];
  float ySl[MAX_SEGMENT];
  float dydzSl[MAX_SEGMENT];
  float eySl[MAX_SEGMENT];
  float edydzSl[MAX_SEGMENT];
  float eydydzSl[MAX_SEGMENT];
  float xSlSL1[MAX_SEGMENT];
  float dxdzSlSL1[MAX_SEGMENT];
  float exSlSL1[MAX_SEGMENT];
  float edxdzSlSL1[MAX_SEGMENT];
  float exdxdzSlSL1[MAX_SEGMENT];
  float xSL1SL3[MAX_SEGMENT];
  float xSlSL3[MAX_SEGMENT];
  float dxdzSlSL3[MAX_SEGMENT];
  float exSlSL3[MAX_SEGMENT];
  float edxdzSlSL3[MAX_SEGMENT];
  float exdxdzSlSL3[MAX_SEGMENT];
  float xSL3SL1[MAX_SEGMENT];
  float xc[MAX_SEGMENT][MAX_HIT_CHAM];
  float yc[MAX_SEGMENT][MAX_HIT_CHAM];
  float zc[MAX_SEGMENT][MAX_HIT_CHAM];
  float ex[MAX_SEGMENT][MAX_HIT_CHAM];
  float xcp[MAX_SEGMENT][MAX_HIT_CHAM];
  float ycp[MAX_SEGMENT][MAX_HIT_CHAM];
  float excp[MAX_SEGMENT][MAX_HIT_CHAM];
  float eycp[MAX_SEGMENT][MAX_HIT_CHAM];
  int wh[MAX_SEGMENT]; int st[MAX_SEGMENT]; int sr[MAX_SEGMENT];
  int sl[MAX_SEGMENT][MAX_HIT_CHAM];
  int la[MAX_SEGMENT][MAX_HIT_CHAM];
  //---------------------------------------------------------------

  private:


};

#endif
