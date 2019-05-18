#ifndef Alignment_MuonAlignmentAlgorithms_ReadPGInfo_H
#define Alignment_MuonAlignmentAlgorithms_ReadPGInfo_H

/** \class ReadPGInfo
 *  $Date: 2010/02/25 11:33:32 $
 *  $Revision: 1.3 $
 *  \author Luca Scodellaro <Luca.Scodellaro@cern.ch>
 */

#include <TFile.h>
#include <TMatrixD.h>
#include <TMath.h>

#define TOTALCHAMBERS 264

class ReadPGInfo {
public:
  ReadPGInfo(const char *name);
  ~ReadPGInfo();
  char *getId(int, int, int);
  TMatrixD giveR(int, int, int);
  TMatrixD giveQC(int, int, int);
  TMatrixD giveSurvey(int, int, int);
  TMatrixD giveQCCal(int, int, int);

private:
  TFile *rootFile;
};

#endif
