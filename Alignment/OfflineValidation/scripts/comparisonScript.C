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

void comparisonScript(){

	
	gStyle->SetOptStat("emr");
	gSystem->Load("comparisonPlots_cc.so");
	
	// create plots object for given input
	// arg1 = input ROOT comparison, arg2 = output directory, arg3 = name of output ROOT file
	comparisonPlots c1("../test/comparisonSurvey.root","outputDir/");
	
	// ------------ COMMON CUTS -----------
	// LEVEL CUT - which hierarchy to plot
	// for convention, see: http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Alignment/CommonAlignment/interface/StructureType.h?view=log
	TCut levelCut = "(level == 2)";
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
	c1.plot3x5( levelCut, true );
	//c1.plot3x5( levelCut+PXBCut );
	//c1.plot3x5( levelCut+PXFCut );
	//c1.plot3x5( levelCut+TIBCut );
	//c1.plot3x5( levelCut+TIDCut );
	//c1.plot3x5( levelCut+TOBCut );
	//c1.plot3x5( levelCut+TECCut );
	
	// plots the normal 3x3 plus dx/dy vs. r/z/phi 2D profile plots
	// second argument is the nBinsX for profile plot
	// all arguments are stored to output file
	c1.plot3x5Profile( levelCut, 30 );
	//c1.plot3x5Profile( levelCut+PXBCut, 30 );
	//c1.plot3x5Profile( levelCut+PXFCut, 30 );
	//c1.plot3x5Profile( levelCut+TIBCut, 30 );
	//c1.plot3x5Profile( levelCut+TIDCut, 30) ;
	//c1.plot3x5Profile( levelCut+TOBCut, 30 );
	//c1.plot3x5Profile( levelCut+TECCut, 30 );
	
	
}
