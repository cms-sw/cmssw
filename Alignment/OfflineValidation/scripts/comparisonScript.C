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

void comparisonScript(string inFile="../test/testComparison.root",string outDir="outputDir/")
{
	gStyle->SetOptStat("emr");
	gROOT->ProcessLine(".L comparisonPlots.cc+");
		
	// create plots object for given input
	// arg1 = input ROOT comparison, arg2 = output directory, arg3 = name of output ROOT file
	comparisonPlots c1(inFile.c_str(),outDir.c_str());
	
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

	TCut Det1dCut = "(detDim==1)";
	TCut Det2dCut = "(detDim==2)";  

		
	// for plot3x5:
	// arg1 = cuts, arg2 = bool to save the 3x5 plot, arg3 = name of saved plot
	// for plot3x5Profile:
	// arg1 = cuts, arg2 = nBins for profile, arg3 = bool to save 3x5 profile plot, arg4 = name of saved plot
	
	// plots the normal 3x3 plus dx/dy vs. r/z/phi
	// stores each histogram to output file (including dr/dz/r*dphi 1D plots)
	//syntax (TCut Cut, dirrectory name, bool savePlot, std::string plotName, bool autolimits, int ColorCode (0=z/-z separation| 1= subdetector seperation ))
	
	c1.plot3x5( levelCut, "Tracker",true,"Tracker.pdf", true,1 );
	c1.plot3x5( levelCut+PXBCut, "PXB", true,  "PXB.pdf", true ,0);
	c1.plot3x5( levelCut+PXFCut, "PXF", true,  "PXF.pdf", true ,0);
	c1.plot3x5( levelCut+TIBCut, "TIB", true,  "TIB.pdf", true ,0);
	c1.plot3x5( levelCut+TIDCut, "TID", true,  "TID.pdf", true ,0);
	c1.plot3x5( levelCut+TOBCut, "TOB", true,  "TOB.pdf", true ,0);
	c1.plot3x5( levelCut+TECCut, "TEC", true,  "TEC.pdf", true ,0);

	c1.plot3x3Rot( levelCut,        "Tracker", true, "Tracker.pdf", true, 1);
	c1.plot3x3Rot( levelCut+PXBCut, "PXB",     true, "PXB.pdf",     true ,0);
	c1.plot3x3Rot( levelCut+PXFCut, "PXF",     true, "PXF.pdf",     true ,0);
	c1.plot3x3Rot( levelCut+TIBCut, "TIB",     true, "TIB.pdf",     true ,0);
	c1.plot3x3Rot( levelCut+TIDCut, "TID",     true, "TID.pdf",     true ,0);
	c1.plot3x3Rot( levelCut+TOBCut, "TOB",     true, "TOB.pdf",     true ,0);
	c1.plot3x3Rot( levelCut+TECCut, "TEC",     true, "TEC.pdf",     true ,0);

        c1.plotTwist( levelCut, "TwistTracker",true,"Tracker.pdf", true,1 );
	c1.plotTwist( levelCut+PXBCut, "TwistPXB", true,  "PXB.pdf", true ,0);
	c1.plotTwist( levelCut+PXFCut, "TwistPXF", true,  "PXF.pdf", true ,0);
	c1.plotTwist( levelCut+TIBCut, "TwistTIB", true,  "TIB.pdf", true ,0);
	c1.plotTwist( levelCut+TIDCut, "TwistTID", true,  "TID.pdf", true ,0);
	c1.plotTwist( levelCut+TOBCut, "TwistTOB", true,  "TOB.pdf", true ,0);
	c1.plotTwist( levelCut+TECCut, "TwistTEC", true,  "TEC.pdf", true ,0);

	//again this time only for 2D modules

	c1.plot3x5( levelCut+Det2dCut,        "Tracker2D",true,"Tracker2D.pdf", true ,1);
	c1.plot3x5( levelCut+PXBCut+Det2dCut, "PXB2D", true,  "PXB2D.pdf", true );
	c1.plot3x5( levelCut+PXFCut+Det2dCut, "PXF2D", true,  "PXF2D.pdf", true );
	c1.plot3x5( levelCut+TIBCut+Det2dCut, "TIB2D", true,  "TIB2D.pdf", true ,0);
	c1.plot3x5( levelCut+TIDCut+Det2dCut, "TID2D", true,  "TID2D.pdf", true ,0);
	c1.plot3x5( levelCut+TOBCut+Det2dCut, "TOB2D", true,  "TOB2D.pdf", true,0 );
	c1.plot3x5( levelCut+TECCut+Det2dCut, "TEC2D", true,  "TEC2D.pdf", true ,0);
	

	c1.plotTwist( levelCut+Det2dCut,        "TwistTracker2D",true,"Tracker2D.pdf", true ,1);
	c1.plotTwist( levelCut+PXBCut+Det2dCut, "TwistPXB2D", true,  "PXB2D.pdf", true );
	c1.plotTwist( levelCut+PXFCut+Det2dCut, "TwistPXF2D", true,  "PXF2D.pdf", true );
	c1.plotTwist( levelCut+TIBCut+Det2dCut, "TwistTIB2D", true,  "TIB2D.pdf", true ,0);
	c1.plotTwist( levelCut+TIDCut+Det2dCut, "TwistTID2D", true,  "TID2D.pdf", true ,0);
	c1.plotTwist( levelCut+TOBCut+Det2dCut, "TwistTOB2D", true,  "TOB2D.pdf", true,0 );
	c1.plotTwist( levelCut+TECCut+Det2dCut, "TwistTEC2D", true,  "TEC2D.pdf", true ,0);
	
	// plots the normal 3x3 plus dx/dy vs. r/z/phi 2D profile plots
	// second argument is the nBinsX for profile plot
	// all arguments are stored to output file
	//syntax (TCut Cut, dirrectory name, bool savePlot, std::string plotName, bool autolimits, int ColorCode (0=z/-z separation| 1= subdetector seperation ))
	c1.plot3x5Profile( levelCut,       "Tracker", 50, true, "Tracker.pdf", true, 1);
	c1.plot3x5Profile( levelCut+PXBCut,"PXB",     50, true, "PXB.pdf",     true, 0);
	c1.plot3x5Profile( levelCut+PXFCut,"PXF",     50, true, "PXF.pdf",     true, 0);
	c1.plot3x5Profile( levelCut+TIBCut,"TIB",     50 ,true, "TIB.pdf",     true, 0);
	c1.plot3x5Profile( levelCut+TIDCut,"TID",     50, true, "TID.pdf",     true, 0);
	c1.plot3x5Profile( levelCut+TOBCut,"TOB",     50, true, "TOB.pdf",     true, 0);
	c1.plot3x5Profile( levelCut+TECCut,"TEC",     50, true, "TEC.pdf",     true, 0);
	
	return ; 
	
}
