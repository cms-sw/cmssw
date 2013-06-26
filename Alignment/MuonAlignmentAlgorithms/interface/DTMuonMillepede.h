#ifndef Alignment_MuonAlignmentAlgorithms_DTMuonMillepede_H
#define Alignment_MuonAlignmentAlgorithms_DTMuonMillepede_H

/** \class DTMuonMillepede
 *  $Date: 2011/09/15 08:52:14 $
 *  $Revision: 1.3 $
 *  \author Luca Scodellaro <Luca.Scodellaro@cern.ch>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/DTMuonLocalAlignment.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/ReadPGInfo.h"
#include "TMatrixD.h"
#include "TFile.h"
#include "TTree.h"
#include <string>
#include "TChain.h"
#include "math.h"


class DTMuonMillepede : public DTMuonLocalAlignment {
  
  public:

  DTMuonMillepede(std::string, int, float, float, int, int, int, int);

  ~DTMuonMillepede(); 

  void calculationMillepede(int);

  TMatrixD getCcsMatrix(int, int, int);
  
  TMatrixD getbcsMatrix(int, int, int);

  TMatrixD getMatrixFromFile(TString Code, int , int, int, int);

  TMatrixD getCqcMatrix(int, int, int);
  
  TMatrixD getbqcMatrix(int, int, int);
  
  TMatrixD getCsurveyMatrix(int, int, int);
  
  TMatrixD getbsurveyMatrix(int, int, int);

  TMatrixD getLagMatrix(int, int, int);
  
  TMatrixD prepareForLagrange(const TMatrixD &);

  void setBranchTree();

  private:

  ReadPGInfo *myPG;
  
  TFile *f;
  TTree *ttreeOutput;

  float ptMax, ptMin;

  int nPhiHits, nThetaHits;

  //Variables for the output tree
  //---------------------------------------------------------
  int whC, stC, srC;
  int slC[12], laC[12];
  float dx[12], dy[12], dz[12], phix[12], phiy[12], phiz[12];
  float cov[60][60];
  //---------------------------------------------------------
   
};

#endif
