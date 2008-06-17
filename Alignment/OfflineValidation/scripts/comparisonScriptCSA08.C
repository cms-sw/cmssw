#include <string>
#include <sstream>

#include "TFile.h"
#include "TList.h"
#include "TNtuple.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCut.h"

void comparisonScriptCSA08(){

	
	gStyle->SetOptStat("emr");
	gSystem->Load("comparisonPlots_cc.so");
	
	// create plots object for given input
	// arg1 = input ROOT comparison, arg2 = output directory, arg3 = name of output ROOT file
	comparisonPlots c1("../test/myComparison_commonTracker.root","outputDirTracker/");
	comparisonPlots c2("../test/myComparison_commonSubdets.root","outputDirSubdets/");
	
	// ------------ COMMON CUTS -----------
	// LEVEL CUT - which hierarchy to plot
	// for convention, see: http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Alignment/CommonAlignment/interface/StructureType.h?view=log
	TCut levelCut = "(level == 1)"; // plotting DetUnits
	// SUBLEVEL CUT - plot only alignables belongnig to this subdetector
	TCut PXBCut = "(sublevel == 1)"; // PXB
	TCut PXFCut = "(sublevel == 2)"; // PXF
	TCut TIBCut = "(sublevel == 3)"; // TIB
	TCut TIDCut = "(sublevel == 4)"; // TID
	TCut TOBCut = "(sublevel == 5)"; // TOB
	TCut TECCut = "(sublevel == 6)"; // TEC
		
	// for plot3x5:
	// arg1 = cuts, arg2 = bool to save the 3x5 plot, arg3 = name of saved plot
	// for plot3x5Profile:
	// arg1 = cuts, arg2 = nBins for profile, arg3 = bool to save 3x5 profile plot, arg4 = name of saved plot
	
	// plots the normal 3x3 plus dx/dy vs. r/z/phi
	// stores each histogram to output file (including dr/dz/r*dphi 1D plots)
	c1.plot3x5( levelCut, "Tracker", true, "Tracker.eps");
	c1.plot3x5( levelCut+PXBCut, "PXB", true, "PXB.eps" );
	c1.plot3x5( levelCut+PXFCut, "PXF", true, "PXF.eps" );
	c1.plot3x5( levelCut+TIBCut, "TIB", true, "TIB.eps" );
	c1.plot3x5( levelCut+TIDCut, "TID", true, "TID.eps" );
	c1.plot3x5( levelCut+TOBCut, "TOB", true, "TOB.eps" );
	c1.plot3x5( levelCut+TECCut, "TEC", true, "TEC.eps" );
	
	// plots the normal 3x3 plus dx/dy vs. r/z/phi 2D profile plots
	// second argument is the nBinsX for profile plot
	// all arguments are stored to output file
	c1.plot3x5Profile( levelCut, "Tracker", 30, true, "Tracker.eps" );
	c1.plot3x5Profile( levelCut+PXBCut, "PXB", 30, true, "PXB.eps" );
	c1.plot3x5Profile( levelCut+PXFCut, "PXF", 30, true, "PXF.eps" );
	c1.plot3x5Profile( levelCut+TIBCut, "TIB", 30, true, "TIB.eps" );
	c1.plot3x5Profile( levelCut+TIDCut, "TID", 30, true, "TID.eps" ) ;
	c1.plot3x5Profile( levelCut+TOBCut, "TOB", 30, true, "TOB.eps" );
	c1.plot3x5Profile( levelCut+TECCut, "TEC", 30, true, "TEC.eps" );
	
	// plots the normal 3x3 plus dx/dy vs. r/z/phi
	// stores each histogram to output file (including dr/dz/r*dphi 1D plots)
	c2.plot3x5( levelCut, "Tracker", true, "Tracker.eps");
	c2.plot3x5( levelCut+PXBCut, "PXB", true, "PXB.eps" );
	c2.plot3x5( levelCut+PXFCut, "PXF", true, "PXF.eps" );
	c2.plot3x5( levelCut+TIBCut, "TIB", true, "TIB.eps" );
	c2.plot3x5( levelCut+TIDCut, "TID", true, "TID.eps" );
	c2.plot3x5( levelCut+TOBCut, "TOB", true, "TOB.eps" );
	c2.plot3x5( levelCut+TECCut, "TEC", true, "TEC.eps" );
	
	// plots the normal 3x3 plus dx/dy vs. r/z/phi 2D profile plots
	// second argument is the nBinsX for profile plot
	// all arguments are stored to output file
	c2.plot3x5Profile( levelCut, "Tracker", 30, true, "Tracker.eps" );
	c2.plot3x5Profile( levelCut+PXBCut, "PXB", 30, true, "PXB.eps" );
	c2.plot3x5Profile( levelCut+PXFCut, "PXF", 30, true, "PXF.eps" );
	c2.plot3x5Profile( levelCut+TIBCut, "TIB", 30, true, "TIB.eps" );
	c2.plot3x5Profile( levelCut+TIDCut, "TID", 30, true, "TID.eps" ) ;
	c2.plot3x5Profile( levelCut+TOBCut, "TOB", 30, true, "TOB.eps" );
	c2.plot3x5Profile( levelCut+TECCut, "TEC", 30, true, "TEC.eps" );
	
	
}
