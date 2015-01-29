#include <TROOT.h>
#include <TApplication.h>

#include <sys/stat.h>

#include "GeometryComparisonPlotter.cc" // TO DO: only include the header and provide the .o file

///////////////////////////////////////////////////////////////////////////////////////
// README:                                                                           //
///////////////////////////////////////////////////////////////////////////////////////
// This script is an example highly documented to present the production             //
// of comparison plots between to geometries of the tracker.                         //
// The main idea is to provide some input file and output destination                //
// and to provide the couples to plot. Each combination gives rise to a single plot  //
// and within a global output.                                                       //
// Some options/cuts/specifications may be added.                                    //
///////////////////////////////////////////////////////////////////////////////////////
// Any question can be asked to Patrick Connor at the address patrick.connor@desy.de //
///////////////////////////////////////////////////////////////////////////////////////

int GeometryComparisonPlotter::canvas_index = 0;


void comparisonScript (TString inFile="mp1510_vs_mp1509.Comparison_commonTracker.root",
                       TString outDir="outputDir/")
{
    // the output directory is created if it does not exist
    mkdir(outDir, S_IRWXU);
    // display canvases: be careful, as there are many many ... canvases

    // the name of the variables are the names of the branches
    // REMARK: a supplementary branch for rdphi is calculated automatically
    //         from r and dphi

    // now the object to produce the comparison plots is created
    GeometryComparisonPlotter * cp = new GeometryComparisonPlotter (inFile, outDir);
    // x and y contain the couples to plot
    // -> every combination possible will be performed
    // /!\ always give units (otherwise, unexpected bug from root...)
    vector<TString> x,y;
    x.push_back("r");                                           cp->SetBranchUnits("r",     "cm");
    x.push_back("phi");                                         cp->SetBranchUnits("phi",   "rad");
    x.push_back("z");                                           cp->SetBranchUnits("z",     "cm");
    y.push_back("dr");		cp->SetBranchSF("dr", 	10000);     cp->SetBranchUnits("dr",    "#mu m");
    y.push_back("dz");		cp->SetBranchSF("dz", 	10000);     cp->SetBranchUnits("dz",    "#mu m");
    y.push_back("rdphi");	cp->SetBranchSF("rdphi",10000);     cp->SetBranchUnits("rdphi", "#mu m rad");
    y.push_back("dx");		cp->SetBranchSF("dx", 	10000);     cp->SetBranchUnits("dx",    "#mu m");
    y.push_back("dy");		cp->SetBranchSF("dy", 	10000);     cp->SetBranchUnits("dy",    "#mu m");
    cp->SetPrintOption("png");
    cp->MakePlots(x, y);
    // remark: what takes the more time is the creation of the output files,
    // not the looping on the tree (because the code is perfect, of course :p)
    cp->SetPrintOption("pdf");
    cp->MakePlots(x, y);

    // now the same object can be reused with other specifications/cuts
    //SetPrint               (const bool);      // option to produce output files
    //SetWrite               (const bool);      // option to also produce a root file
    //Set1dModule            (const bool);      // "false" cuts on 1d modules
    //Set2dModule            (const bool);      // id for 2d
    //SetLevelCut            (const int);       // corresponds to the branch level
    //SetBatchMode           (const bool);      // display option
    //SetBranchMax           (const TString,    // set fixed maximum
    //                        const float);     
    //SetBranchMin           (const TString,    // id for min
    //                        const float);
    //SetBranchUnits         (const TString,    // set branch units
    //                        const float);
    //SetBranchSF            (const TString,    // rescaling factor (i.e change units)
    //                        const float);
    //SetOutputDirectoryName (const TString);   // change the destination
    //SetOutputFileName      (const TString);   // change the output filename
    //SetPrintOption         (const Option_t *);// see TPad::Print() for possible options
}

// the following line is necessary for standalone applications
// so in this case just run the makefile and the standalone executable with right arguments:
// - root file containing the tree
// - name of the output directory
// otherwise, juste ignore this part of the code
#ifndef __CINT__
int main (int argc, char * argv[])
{
    TApplication * app = new TApplication ("comparisonScript", &argc, argv);
    comparisonScript(app->Argv(1),
                     app->Argv(2));
    app->Run();
    // ask René Brun if you wonder why it is needed, I have no damned idea :p
#ifndef DEBUG
    delete app;
#endif
    return 0;
}
#endif
