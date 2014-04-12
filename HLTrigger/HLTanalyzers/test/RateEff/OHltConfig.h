//////////////////////////////////////////////////////////
//
// Class to parse and store configuration file
//
//////////////////////////////////////////////////////////

#ifndef OHltConfig_h
#define OHltConfig_h

#include <iostream>
#include <cmath>
#include <vector>
#include <TString.h>
#include <libconfig.h++>
#include "OHltMenu.h"

class OHltConfig
{
public:

   OHltConfig() {}

   OHltConfig(TString cfgfile, OHltMenu *omenu);
   
   virtual ~OHltConfig() {}

   void print();
   void fillMenu(OHltMenu *omenu);
   void printMenu(OHltMenu *omenu);
   void convert();
   void fillRunBlockList();
   void getPreFilter();

   // Data
   libconfig::Config cfg;

   /**** General Menu & Run conditions ****/
   int nEntries;
   int nPrintStatusEvery;
   bool isRealData;
   TString menuTag;
   TString versionTag;
   bool doPrintAll;
   bool doDeterministicPrescale; // default is random prescale
   bool useNonIntegerPrescales; // default is integer prescales
   bool readRefPrescalesFromNtuple; // default is read prescales from config
   TString nonlinearPileupFit; // default is to do a linear extrapolation
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
   float lumiScaleFactor;
   int prescaleNormalization;
   std::vector < std::vector <int> > runLumiblockList; // format: (runnr, minLumiBlock, maxLumiBlock)


   /******************************/

   /**** Samples & processes ****/
   std::vector<TString> pnames;
   std::vector<TString> ppaths;
   std::vector<TString> pfnames;
   std::vector<bool> pdomucuts;
   std::vector<bool> pdoecuts;
   std::vector<float> psigmas;
   std::vector <int> pisPhysicsSample; // Is it a RATE sample (MB, QCD) or a PHYSICS sample (W,Z,top)
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
