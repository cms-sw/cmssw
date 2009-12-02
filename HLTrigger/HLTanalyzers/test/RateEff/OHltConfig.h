//////////////////////////////////////////////////////////
//
// Class to parse and store configuration file
//
//////////////////////////////////////////////////////////

#ifndef OHltConfig_h
#define OHltConfig_h

#include <iostream>
#include <vector>
#include <TString.h>
#include <libconfig.h++>
#include "OHltMenu.h"

using namespace std;
using namespace libconfig;

class OHltConfig {
 public:

  OHltConfig() {};
  OHltConfig(TString cfgfile,OHltMenu *omenu);
  virtual ~OHltConfig(){};
  void OHltConfig::print();
  void OHltConfig::fillMenu(OHltMenu *omenu);
  void OHltConfig::printMenu(OHltMenu *omenu);
  void OHltConfig::convert();
  void OHltConfig::fillRunBlockList();
  void OHltConfig::getPreFilter();

  // Data
  Config cfg;

  /**** General Menu & Run conditions ****/
  int nEntries;
  int nPrintStatusEvery;
  bool isRealData;
  TString menuTag;
  TString alcaCondition;
  TString versionTag;
  bool doPrintAll;
  bool doDeterministicPrescale; // default is random prescale
  TString dsList;
  /*************************/

  /**** Beam conditions ****/
  float iLumi;
  float bunchCrossingTime;
  int maxFilledBunches;
  int nFilledBunches;
  float cmsEnergy;
  /*************************/

  /**** Real data conditions ****/
  float liveTimeRun;
  int nL1AcceptsRun;
  float lumiSectionLength;
  int prescaleNormalization;
  vector < vector <int> > runLumiblockList; // format: (runnr, minLumiBlock, maxLumiBlock)

  
  /******************************/

  /**** Samples & processes ****/
  vector<TString> pnames;
  vector<TString> ppaths;
  vector<TString> pfnames;
  vector<bool> pdomucuts;
  vector<bool> pdoecuts;
  vector<float> psigmas;
  vector <int> pisPhysicsSample; // Is it a RATE sample (MB, QCD) or a PHYSICS sample (W,Z,top)
  /*****************************/

  /**** Menu ****/
  bool isL1Menu;
  bool doL1preloop;
  /**********************************/

  /****  ****/
  // Only for experts:
  // Select certain branches to speed up code.
  // Modify only if you know what you do!
  bool doSelectBranches;
  bool selectBranchL1;
  bool selectBranchHLT;
  bool selectBranchOpenHLT;
  bool selectBranchReco;
  bool selectBranchL1extra;
  bool selectBranchMC;

  /**********************************/
  TString preFilterLogicString;
};
#endif
