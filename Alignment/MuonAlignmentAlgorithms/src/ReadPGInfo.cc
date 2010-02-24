#include "Alignment/MuonAlignmentAlgorithms/interface/ReadPGInfo.h"


ReadPGInfo::ReadPGInfo(char *name) {
  rootFile = new TFile(name);
}

ReadPGInfo::~ReadPGInfo(){delete rootFile;}

char * ReadPGInfo::getId(int wheel, int station, int sector) {
  for(int counter = 0; counter < TOTALCHAMBERS; ++counter) {
    if(wheel == position[counter][0] && sector == position[counter][1] && station == position[counter][2])
      return chambers[counter];
  }
  return NULL;
}


TMatrixD ReadPGInfo::giveR(int wheel, int station, int sector) {
  TMatrixD *empty = new TMatrixD(0,0);
  char *id = getId(wheel, station, sector);
  if(id == NULL) return *empty;
  TDirectoryFile *myDir = (TDirectoryFile *)rootFile->Get(id);
  TDirectoryFile *myR = (TDirectoryFile *)myDir->Get("R");
  TMatrixD *R = (TMatrixD *)myR->Get("matrix");
  return *R;
}
    

TMatrixD ReadPGInfo::giveQCCal(int wheel, int station, int sector) {
  TMatrixD *mat = new TMatrixD(0,0);
  TMatrixD qc = giveQC(wheel, station, sector);
  if(qc.GetNrows() == 0) return *mat;
  mat->ResizeTo(12,2);
  int maxCount = 12;
  if(station == 4) maxCount = 8;
  for(int c = 0; c < maxCount; ++c) {
    float error;
    if(qc(c,1) == 0 || qc(c,3) == 0) {
      (*mat)(c,0) = (qc(c,0)+qc(c,2)) /2.0;
      (*mat)(c,1) = 500;
    } else {  
      error = 1.0/(1.0/(qc(c,1)*qc(c,1))+1.0/(qc(c,3)*qc(c,3)));
     (*mat)(c, 0) = (qc(c,0)/(qc(c,1)*qc(c,1))+qc(c,2)/(qc(c,3)*qc(c,3)))*error;
     (*mat)(c, 1) = TMath::Sqrt(error);
    }
  }
  return *mat;
}

TMatrixD ReadPGInfo::giveQC(int wheel, int station, int sector) {
  TMatrixD *empty = new TMatrixD(0,0);
  char *id = getId(wheel, station, sector);
  if(id == NULL) return *empty;
  TDirectoryFile *myDir = (TDirectoryFile *)rootFile->Get(id);
  TDirectoryFile *myQC = (TDirectoryFile *)myDir->Get("QCW");
  TMatrixD *QC;
  if(myQC == NULL) {
    QC = new TMatrixD(0,0);
  } else {
    QC = (TMatrixD *)myQC->Get("matrix");
  }
  return *QC;
}


TMatrixD ReadPGInfo::giveSurvey(int wheel, int station, int sector) {
  TMatrixD *empty = new TMatrixD(0,0);
  char *id = getId(wheel, station, sector);
  if(id == NULL) return *empty;
  TDirectoryFile *myDir = (TDirectoryFile *)rootFile->Get(id);
  TDirectoryFile *mySur = (TDirectoryFile *)myDir->Get("Survey");
  TMatrixD *Survey = (TMatrixD *)mySur->Get("matrix_layer");
  return *Survey;
}





