// Aram Avetisyan; Brown University; February 15, 2008

#ifndef DBlmapReader_h
#define DBlmapReader_h

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <algorithm>

#include "occi.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/HCALConfigDB.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabase.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImplOracle.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImplXMLFile.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationItemNotFoundException.hh"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

using namespace std;

struct VectorLMAP{
  
  vector<int> sideC;
  vector<int> etaC;
  vector<int> phiC;
  
  vector<int> dphiC;
  vector<int> depthC;
  vector<string> detC;
  vector<string> rbxC;
  vector<int> wedgeC;
  
  vector<int> sectorC;
  vector<int> rmC;
  vector<int> pixelC;
  vector<int> qieC;
  vector<int> adcC;
  
  vector<int>  rm_fiC;
  vector<int> fi_chC;
  vector<string> let_codeC;
  vector<int> crateC;
  vector<int> htrC;
  
  vector<string> fpgaC;
  vector<int> htr_fiC;
  vector<int> dcc_slC;
  vector<int> spigoC;
  vector<int> dccC;
  
  vector<int> slbC;
  vector<string> slbinC;
  vector<string> slbin2C;
  vector<string> slnamC;
  vector<int> rctcraC;
  
  vector<int> rctcarC;
  vector<int> rctconC;
  vector<string> rctnamC;
  vector<int> fedidC;
  vector<int> geoC;

  vector<int> blockC;
  vector<int> lcC;
  
  vector<int> orderC;
  vector<int> versionC;
};

bool SortComp(int x, int y);
VectorLMAP* SortByHardware(VectorLMAP* lmapHBEFO);
VectorLMAP* SortByGeometry(VectorLMAP* lmapHBEFO);

void printHBHEHF(int channel, FILE * HBEFmap, VectorLMAP * lmap);
void printHO(int channel, FILE * HOmap, VectorLMAP * lmap);
void printEMAProw(int channel, FILE * emap, VectorLMAP * lmap);

class DBlmapReader{
 public:
  DBlmapReader(){ };
  void lrTestFunction(void);
  VectorLMAP* GetLMAP(int version);
  void PrintLMAP(FILE* HBEFmap, FILE* HOmap, VectorLMAP* lmapHBEFO);
  void PrintEMAPfromLMAP(FILE* emap, VectorLMAP* lmapHBEFO);
};

#endif
