//
// $Id: CSA07ProcessId.h,v 1.9 2008/04/04 14:20:55 lowette Exp $
//

#ifndef PhysicsTools_HepMCCandAlgos_CSA07ProcessId_h
#define PhysicsTools_HepMCCandAlgos_CSA07ProcessId_h

/* CSA07ProcessId

  This file contains two utilities to be used in the EDM framework:

  1/ csa07ProcessId
     Description: csa07ProcessId acts like a global function in the csa07
       namespace that returns you a unique consecutive id for each sample that
       went into the CSA07 soups. The definition of what id the various
       processes correspond to can be found in the code and on the
       CSA07ProcessId twiki page.
     Implemented to be used by the user as a function which looks like:
       int csa07::csa07ProcessId(const edm::Event & iEvent);

  2/ csa07ProcessName
     Description: csa07ProcessName acts like a global function in the csa07
       namespace that returns the name corresponding to the csa07ProcessId
       you give as input.
     Implemented to be used by the user as a function which looks like:
       char * csa07::csa07ProcessName(int csa07ProcessId);

  A note on the implementation: for reasons of ease of use and C++ limitations
  a free global function could not be used, since this gives conflicts when
  linking in separate compilation units that both include and use the function.
  Therefore a detour was chosen through a converting constructor: the interface
  to the user stays the same, and BuildFile-dependency is avoided.

*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"


namespace csa07 {

  class csa07ProcessId {
    private:
      int csa07ProcId_;
    public:
      operator int() const { return csa07ProcId_; }
      csa07ProcessId(const edm::Event & iEvent) {
    	// get process Id
    	bool runOnChowder = false; // check if this is a chowder sample
    	edm::Handle<int> procIdH;
    	iEvent.getByLabel("genEventProcID", procIdH);
    	int procId = *procIdH;
    	if (procId == 4) { // it's chowder!
    	  runOnChowder = true;
    	  iEvent.getByLabel("csa07EventWeightProducer", "AlpgenProcessID", procIdH);
    	  procId = *procIdH;
    	}
    	// get generator event scale
    	edm::Handle<double> scale;
    	iEvent.getByLabel("genEventScale", scale);
    	double ptHat = *scale;
    	// get generated filter efficiency
    	double filterEff;
    	if (runOnChowder) {
    	  filterEff = -1; // not available for alpgen samples
    	} else {
    	  edm::Handle<double> filterEffH;
    	  iEvent.getByLabel("genEventRunInfo", "FilterEfficiency", filterEffH);
    	  filterEff = *filterEffH;
    	}
    	// get csa07 weight
    	edm::Handle<double> weightH;
        iEvent.getByLabel("csa07EventWeightProducer", "weight", weightH);
    	double weight = *weightH;
    	// get the csa07 process id
    	int csa07ProcId;
        // chowder processes
    	if (procId == 1000)
    	  csa07ProcId = 0;  // W+0jet		 (weight ~ 5.15)
    	else if (procId == 1001 && weight < 1.03)
    	  csa07ProcId = 1;  // W+1jet 0<pTW<100   (weight ~ 1.02)
    	else if (procId == 1001 && weight > 1.03)
    	  csa07ProcId = 2;  // W+1jet 100<pTW<300 (weight ~ 1.04)
    	else if (procId == 1002 && weight > 1.)
    	  csa07ProcId = 3;  // W+2jet 0<pTW<100   (weight ~ 1.07)
    	else if (procId == 1002 && weight < 1.)
    	  csa07ProcId = 4;  // W+2jet 100<pTW<300 (weight ~ 0.78)
    	else if (procId == 1003 && weight > 1.)
    	  csa07ProcId = 5;  // W+3jet 0<pTW<100   (weight ~ 1.67)
    	else if (procId == 1003 && weight < 1.)
    	  csa07ProcId = 6;  // W+3jet 100<pTW<300 (weight ~ 0.91)
    	else if (procId == 1004 && weight > 0.96)
    	  csa07ProcId = 7;  // W+4jet 0<pTW<100   (weight ~ 0.98)
    	else if (procId == 1004 && weight < 0.96)
    	  csa07ProcId = 8;  // W+4jet 100<pTW<300 (weight ~ 0.95)
    	else if (procId == 1005 && weight > 1.)
    	  csa07ProcId = 9;  // W+5jet 0<pTW<100   (weight ~ 1.35)
    	else if (procId == 1005 && weight < 1.)
    	  csa07ProcId = 10; // W+5jet 100<pTW<300 (weight ~ 0.90)
    	else if (procId == 2000)
    	  csa07ProcId = 11; // Z+0jet		  (weight ~ 1.38)
    	else if (procId == 2001 && weight > 0.9)
    	  csa07ProcId = 12; // Z+1jet 0<pTZ<100   (weight ~ 0.98)
    	else if (procId == 2001 && weight < 0.9)
    	  csa07ProcId = 13; // Z+1jet 100<pTZ<300 (weight ~ 0.83)
    	else if (procId == 2002 && weight > 0.9)
    	  csa07ProcId = 14; // Z+2jet 0<pTZ<100   (weight ~ 0.93)
    	else if (procId == 2002 && weight < 0.9)
    	  csa07ProcId = 15; // Z+2jet 100<pTZ<300 (weight ~ 0.80)
    	else if (procId == 2003 && weight > 0.9)
    	  csa07ProcId = 16; // Z+3jet 0<pTZ<100   (weight ~ 0.94)
    	else if (procId == 2003 && weight < 0.9)
    	  csa07ProcId = 17; // Z+3jet 100<pTZ<300 (weight ~ 0.53)
    	else if (procId == 2004 && weight < 0.5)
    	  csa07ProcId = 18; // Z+4jet 0<pTZ<100   (weight ~ 0.42)
    	else if (procId == 2004 && weight > 0.5)
    	  csa07ProcId = 19; // Z+4jet 100<pTZ<300 (weight ~ 0.63)
    	else if (procId == 2005 && weight < 0.8)
    	  csa07ProcId = 20; // Z+5jet 0<pTZ<100   (weight ~ 0.72)
    	else if (procId == 2005 && weight > 0.8)
    	  csa07ProcId = 21; // Z+5jet 100<pTZ<300 (weight ~ 0.85)
    	else if (procId == 3000)
    	  csa07ProcId = 22; // tt+0jet
    	else if (procId == 3001)
    	  csa07ProcId = 23; // tt+1jet
    	else if (procId == 3002)
    	  csa07ProcId = 24; // tt+2jet
    	else if (procId == 3003)
    	  csa07ProcId = 25; // tt+3jet
    	else if (procId == 3004)
    	  csa07ProcId = 26; // tt+4jet
        // gumbo processes
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (15 < ptHat && ptHat < 20) && filterEff == 1.)
    	  csa07ProcId = 28; // QCD_Pt_15_20
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (20 < ptHat && ptHat < 30) && filterEff == 1.)
    	  csa07ProcId = 29; // QCD_Pt_20_30
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (30 < ptHat && ptHat < 50) && filterEff == 1.)
    	  csa07ProcId = 30; // QCD_Pt_30_50
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (50 < ptHat && ptHat < 80) && filterEff == 1.)
    	  csa07ProcId = 31; // QCD_Pt_50_80
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (80 < ptHat && ptHat < 120) && filterEff == 1.)
    	  csa07ProcId = 32; // QCD_Pt_80_120
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (120 < ptHat && ptHat < 170) && filterEff == 1.)
    	  csa07ProcId = 33; // QCD_Pt_120_170
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (170 < ptHat && ptHat < 230) && filterEff == 1.)
    	  csa07ProcId = 34; // QCD_Pt_170_230
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (230 < ptHat && ptHat < 300) && filterEff == 1.)
    	  csa07ProcId = 35; // QCD_Pt_230_300
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (300 < ptHat && ptHat < 380) && filterEff == 1.)
    	  csa07ProcId = 36; // QCD_Pt_300_380
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (380 < ptHat && ptHat < 470) && filterEff == 1.)
    	  csa07ProcId = 37; // QCD_Pt_380_470
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (470 < ptHat && ptHat < 600) && filterEff == 1.)
    	  csa07ProcId = 38; // QCD_Pt_470_600
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (600 < ptHat && ptHat < 800) && filterEff == 1.)
    	  csa07ProcId = 39; // QCD_Pt_600_800
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (800 < ptHat && ptHat < 1000) && filterEff == 1.)
    	  csa07ProcId = 40; // QCD_Pt_800_1000
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (1000 < ptHat && ptHat < 1400) && filterEff == 1.)
    	  csa07ProcId = 41; // QCD_Pt_1000_1400
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (1400 < ptHat && ptHat < 1800) && filterEff == 1.)
    	  csa07ProcId = 42; // QCD_Pt_1400_1800
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (1800 < ptHat && ptHat < 2200) && filterEff == 1.)
    	  csa07ProcId = 43; // QCD_Pt_1800_2200
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (2200 < ptHat && ptHat < 2600) && filterEff == 1.)
    	  csa07ProcId = 44; // QCD_Pt_2200_2600
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (2600 < ptHat && ptHat < 3000) && filterEff == 1.)
    	  csa07ProcId = 45; // QCD_Pt_2600_3000
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (3000 < ptHat && ptHat < 3500) && filterEff == 1.)
    	  csa07ProcId = 46; // QCD_Pt_3000_3500
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (3500 < ptHat) && filterEff == 1.)
    	  csa07ProcId = 47; // QCD_Pt_3500_inf
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68 || procId == 92 || procId == 93 || procId == 94 || procId == 95) &&
                 (filterEff == 1.))
    	  csa07ProcId = 27; // min bias - MUST COME AFTER ALL QCD
    	else if ((procId == 14 || procId == 18 || procId == 29) &&
    		 (ptHat < 15))
    	  csa07ProcId = 48; // PhotonJets_Pt_0_15
    	else if ((procId == 14 || procId == 18 || procId == 29) &&
    		 (15 < ptHat && ptHat < 20))
    	  csa07ProcId = 49; // PhotonJets_Pt_15_20
    	else if ((procId == 14 || procId == 18 || procId == 29) &&
    		 (20 < ptHat && ptHat < 30))
    	  csa07ProcId = 50; // PhotonJets_Pt_20_30
    	else if ((procId == 14 || procId == 18 || procId == 29) &&
    		 (30 < ptHat && ptHat < 50))
    	  csa07ProcId = 51; // PhotonJets_Pt_30_50
    	else if ((procId == 14 || procId == 18 || procId == 29) &&
    		 (50 < ptHat && ptHat < 80))
    	  csa07ProcId = 52; // PhotonJets_Pt_50_80
    	else if ((procId == 14 || procId == 18 || procId == 29) &&
    		 (80 < ptHat && ptHat < 120))
    	  csa07ProcId = 53; // PhotonJets_Pt_80_120
    	else if ((procId == 14 || procId == 18 || procId == 29) &&
    		 (120 < ptHat && ptHat < 170))
    	  csa07ProcId = 54; // PhotonJets_Pt_120_170
    	else if ((procId == 14 || procId == 18 || procId == 29) &&
    		 (170 < ptHat && ptHat < 300))
    	  csa07ProcId = 55; // PhotonJets_Pt_170_300
    	else if ((procId == 14 || procId == 18 || procId == 29) &&
    		 (300 < ptHat && ptHat < 500))
    	  csa07ProcId = 56; // PhotonJets_Pt_300_500
    	else if ((procId == 14 || procId == 18 || procId == 29) &&
    		 (500 < ptHat && ptHat < 7000))
    	  csa07ProcId = 57; // PhotonJets_Pt_500_7000
    	else if (procId == 102 || procId == 123 || procId == 124)
    	  csa07ProcId = 58; // higgs signal
    	else if (procId == 141)
    	  csa07ProcId = 59; // Zprime signal
        // stew processes
    	else if (filterEff == 0.00013)
    	  csa07ProcId = 60; // B(bar)->JPsi
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68) &&
    		 (filterEff == 1. || filterEff == 0.964))
    	  csa07ProcId = 61; // QCD_Pt_0_15
    	else if ((460 < procId && procId < 480) &&
    		 (0 < ptHat && ptHat < 20))
    	  csa07ProcId = 62; // Bottomonium Pt_0_20
    	else if ((460 < procId && procId < 480) &&
    		 (20 < ptHat))
    	  csa07ProcId = 63; // Bottomonium Pt_20_inf
    	else if ((420 < procId && procId < 440) &&
    		 (0 < ptHat && ptHat < 20))
    	  csa07ProcId = 64; // Charmonium Pt_0_20
    	else if ((420 < procId && procId < 440) &&
    		 (20 < ptHat))
    	  csa07ProcId = 65; // Charmonium Pt_20_inf
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68 || procId == 95) &&
    		 (filterEff == 0.00019))
    	  csa07ProcId = 66; // bbe Pt_5_50
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68 || procId == 95) &&
    		 (filterEff == 0.0068))
    	  csa07ProcId = 67; // bbe Pt_50_170
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68 || procId == 95) &&
    		 (filterEff == 0.0195))
    	  csa07ProcId = 68; // bbe Pt_170_up
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68 || procId == 95) &&
    		 (filterEff == 0.0097))
    	  csa07ProcId = 69; // ppEleX
    	else if ((procId == 11 || procId == 12 || procId == 13 || procId == 28 || procId == 53 || procId == 68 || procId == 95) &&
    		 (filterEff == 0.0008))
    	  csa07ProcId = 70; // ppMuX
    	else {
    	  csa07ProcId = -1; // unknown process
    	  throw cms::Exception("Unknown CSA07 process")
    	    << "OUCH! Unknown CSA07 process with: \n"
    	    << "  ID	    : " << procId    << "\n"
    	    << "  scale     : " << ptHat     << "\n"
    	    << "  filter eff: " << filterEff << "\n"
    	    << "  weight    : " << weight    << "\n";
    	}
    	csa07ProcId_ = csa07ProcId;
      }
  };

  class csa07ProcessName {
    private:
      char * csa07ProcName_;
    public:
      operator char *() const { return csa07ProcName_; }
      csa07ProcessName(int csa07ProcessId) {
  	switch (csa07ProcessId) {
  	  case 0: 
  	    csa07ProcName_ = "W+0jets";
  	    break;
  	  case 1: 
  	    csa07ProcName_ = "W+1jets 0 < pTW < 100";
  	    break;
  	  case 2: 
  	    csa07ProcName_ = "W+1jets 100 < pTW < 300";
  	    break;
  	  case 3: 
  	    csa07ProcName_ = "W+2jets 0 < pTW < 100";
  	    break;
  	  case 4: 
  	    csa07ProcName_ = "W+2jets 100 < pTW < 300";
  	    break;
  	  case 5: 
  	    csa07ProcName_ = "W+3jets 0 < pTW < 100";
  	    break;
  	  case 6: 
  	    csa07ProcName_ = "W+3jets 100 < pTW < 300";
  	    break;
  	  case 7: 
  	    csa07ProcName_ = "W+4jets 0 < pTW < 100";
  	    break;
  	  case 8: 
  	    csa07ProcName_ = "W+4jets 100 < pTW < 300";
  	    break;
  	  case 9: 
  	    csa07ProcName_ = "W+5jets 0 < pTW < 100";
  	    break;
  	  case 10:
  	    csa07ProcName_ = "W+5jets 100 < pTW < 300";
  	    break;
  	  case 11:
  	    csa07ProcName_ = "Z+0jets";
  	    break;
  	  case 12:
  	    csa07ProcName_ = "Z+1jet 0 < pTZ < 100";
  	    break;
  	  case 13:
  	    csa07ProcName_ = "Z+1jet 100 < pTZ < 300 ";
  	    break;
  	  case 14:
  	    csa07ProcName_ = "Z+2jets 0 < pTZ < 100";
  	    break;
  	  case 15:
  	    csa07ProcName_ = "Z+2jets 100 < pTZ < 300";
  	    break;
  	  case 16:
  	    csa07ProcName_ = "Z+3jets 0 < pTZ < 100";
  	    break;
  	  case 17:
  	    csa07ProcName_ = "Z+3jets 100 < pTZ < 300";
  	    break;
  	  case 18:
  	    csa07ProcName_ = "Z+4jets 0 < pTZ < 100";
  	    break;
  	  case 19:
  	    csa07ProcName_ = "Z+4jets 100 < pTZ < 300";
  	    break;
  	  case 20:
  	    csa07ProcName_ = "Z+5jets 0 < pTZ < 100";
  	    break;
  	  case 21:
  	    csa07ProcName_ = "Z+5jets 100 < pTZ < 300";
  	    break;
  	  case 22:
  	    csa07ProcName_ = "tt+0jets";
  	    break;
  	  case 23:
  	    csa07ProcName_ = "tt+1jets";
  	    break;
  	  case 24:
  	    csa07ProcName_ = "tt+2jets";
  	    break;
  	  case 25:
  	    csa07ProcName_ = "tt+3jets";
  	    break;
  	  case 26:
  	    csa07ProcName_ = "tt+4jets";
  	    break;
  	  case 27:
  	    csa07ProcName_ = "Minimum bias";
  	    break;
  	  case 28:
  	    csa07ProcName_ = "QCD Pt_15_20";
  	    break;
  	  case 29:
  	    csa07ProcName_ = "QCD Pt_20_30";
  	    break;
  	  case 30:
  	    csa07ProcName_ = "QCD Pt_30_50";
  	    break;
  	  case 31:
  	    csa07ProcName_ = "QCD Pt_50_80";
  	    break;
  	  case 32:
  	    csa07ProcName_ = "QCD Pt_80_120";
  	    break;
  	  case 33:
  	    csa07ProcName_ = "QCD Pt_120_170";
  	    break;
  	  case 34:
  	    csa07ProcName_ = "QCD Pt_170_230";
  	    break;
  	  case 35:
  	    csa07ProcName_ = "QCD Pt_230_300";
  	    break;
  	  case 36:
  	    csa07ProcName_ = "QCD Pt_300_380";
  	    break;
  	  case 37:
  	    csa07ProcName_ = "QCD Pt_380_470";
  	    break;
  	  case 38:
  	    csa07ProcName_ = "QCD Pt_470_600";
  	    break;
  	  case 39:
  	    csa07ProcName_ = "QCD Pt_600_800";
  	    break;
  	  case 40:
  	    csa07ProcName_ = "QCD Pt_800_1000";
  	    break;
  	  case 41:
  	    csa07ProcName_ = "QCD Pt_1000_1400";
  	    break;
  	  case 42:
  	    csa07ProcName_ = "QCD Pt_1400_1800";
  	    break;
  	  case 43:
  	    csa07ProcName_ = "QCD Pt_1800_2200";
  	    break;
  	  case 44:
  	    csa07ProcName_ = "QCD Pt_2200_2600";
  	    break;
  	  case 45:
  	    csa07ProcName_ = "QCD Pt_2600_3000";
  	    break;
  	  case 46:
  	    csa07ProcName_ = "QCD Pt_3000_3500";
  	    break;
  	  case 47:
  	    csa07ProcName_ = "QCD Pt_3500_inf";
  	    break;
  	  case 48:
  	    csa07ProcName_ = "PhotonJets_Pt_0_15";
  	    break;
  	  case 49:
  	    csa07ProcName_ = "PhotonJets_Pt_15_20";
  	    break;
  	  case 50:
  	    csa07ProcName_ = "PhotonJets_Pt_20_30";
  	    break;
  	  case 51:
  	    csa07ProcName_ = "PhotonJets_Pt_30_50";
  	    break;
  	  case 52:
  	    csa07ProcName_ = "PhotonJets_Pt_50_80";
  	    break;
  	  case 53:
  	    csa07ProcName_ = "PhotonJets_Pt_80_120";
  	    break;
  	  case 54:
  	    csa07ProcName_ = "PhotonJets_Pt_120_170";
  	    break;
  	  case 55:
  	    csa07ProcName_ = "PhotonJets_Pt_170_300";
  	    break;
  	  case 56:
  	    csa07ProcName_ = "PhotonJets_Pt_300_500";
  	    break;
  	  case 57:
  	    csa07ProcName_ = "PhotonJets_Pt_500_700";
  	    break;
  	  case 58:
  	    csa07ProcName_ = "Higgs M150";
  	    break;
  	  case 59:
  	    csa07ProcName_ = "Zprime M1000";
  	    break;
  	  case 60:
  	    csa07ProcName_ = "B(bar) to Jpsi";
  	    break;
  	  case 61:
  	    csa07ProcName_ = "QCD Pt_0_15";
  	    break;
  	  case 62:
  	    csa07ProcName_ = "Bottomonium Pt_0_20";
  	    break;
  	  case 63:
  	    csa07ProcName_ = "Bottomonium Pt_20_inf";
  	    break;
  	  case 64:
  	    csa07ProcName_ = "Charmonium Pt_0_20";
  	    break;
  	  case 65:
  	    csa07ProcName_ = "Charmonium Pt_20_inf";
  	    break;
  	  case 66:
  	    csa07ProcName_ = "Electron enriched bbe Pt_5_50";
  	    break;
  	  case 67:
  	    csa07ProcName_ = "Electron enriched bbe Pt_50_170";
  	    break;
  	  case 68:
  	    csa07ProcName_ = "Electron enriched bbe Pt_170_up";
  	    break;
  	  case 69:
  	    csa07ProcName_ = "Electron enriched ppEleX";
  	    break;
  	  case 70:
  	    csa07ProcName_ = "Muon enriched ppMuX";
  	    break;
  	  default:
  	    csa07ProcName_ = "Unknown CSA07ProcessId";
  	}
      }
  };

}


#endif
