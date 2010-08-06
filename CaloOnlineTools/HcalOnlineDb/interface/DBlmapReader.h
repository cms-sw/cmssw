// Aram Avetisyan; Brown University; February 15, 2008

#ifndef DBlmapReader_h
#define DBlmapReader_h

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <algorithm>

#include "OnlineDB/Oracle/interface/Oracle.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/HCALConfigDB.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabase.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImplOracle.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImplXMLFile.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationItemNotFoundException.hh"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

struct VectorLMAP{
  
  std::vector<int> sideC;
  std::vector<int> etaC;
  std::vector<int> phiC;
  
  std::vector<int> dphiC;
  std::vector<int> depthC;
  std::vector<std::string> detC;
  std::vector<std::string> rbxC;
  std::vector<int> wedgeC;
  
  std::vector<int> sectorC;
  std::vector<int> rmC;
  std::vector<int> pixelC;
  std::vector<int> qieC;
  std::vector<int> adcC;
  
  std::vector<int>  rm_fiC;
  std::vector<int> fi_chC;
  std::vector<std::string> let_codeC;
  std::vector<int> crateC;
  std::vector<int> htrC;
  
  std::vector<std::string> fpgaC;
  std::vector<int> htr_fiC;
  std::vector<int> dcc_slC;
  std::vector<int> spigoC;
  std::vector<int> dccC;
  
  std::vector<int> slbC;
  std::vector<std::string> slbinC;
  std::vector<std::string> slbin2C;
  std::vector<std::string> slnamC;
  std::vector<int> rctcraC;
  
  std::vector<int> rctcarC;
  std::vector<int> rctconC;
  std::vector<std::string> rctnamC;
  std::vector<int> fedidC;
  std::vector<int> geoC;

  std::vector<int> blockC;
  std::vector<int> lcC;
  
  std::vector<int> orderC;
  std::vector<int> versionC;
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
