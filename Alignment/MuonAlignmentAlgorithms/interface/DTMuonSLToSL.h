#ifndef Alignment_MuonAlignmentAlgorithms_DTMuonSLToSL_H
#define Alignment_MuonAlignmentAlgorithms_DTMuonSLToSL_H

/** \class DTMuonSLToSL
 *  $Date: 2010/02/25 11:33:32 $
 *  $Revision: 1.2 $
 *  \author Luca Scodellaro <Luca.Scodellaro@cern.ch>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/DTMuonLocalAlignment.h"
#include "TMatrixD.h"
#include "TFile.h"
#include "TTree.h"
#include <string>
#include "TChain.h"
#include "math.h"

class DTMuonSLToSL : public DTMuonLocalAlignment {
  
  public:

  DTMuonSLToSL(std::string, int, float, float, TFile *);

  ~DTMuonSLToSL(); 

  void calculationSLToSL();

  TMatrixD returnCSLMatrix(float, float, float);
  
  TMatrixD returnbSLMatrix(float, float, float);

  void setBranchTree();

  private:

  //Variables for the output tree
  //------------------------------------- 
  int whC, stC, srC;
  float dx, dz, phiy;
  float cov[3][3];
  //--------------------------------------

  float ptMax, ptMin;

  TTree *ttreeOutput;

};

#endif
