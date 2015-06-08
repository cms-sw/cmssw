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


void comparisonScript (TString inFile,//="mp1510_vs_mp1509.Comparison_commonTracker.root",
                       TString outDir="outputDir/")
{
    // the output directory is created if it does not exist
    mkdir(outDir, S_IRWXU);
    // display canvases: be careful, as there are many many ... canvases

    // the name of the variables are the names of the branches
    // REMARK: a supplementary branch for rdphi is calculated automatically
    //         from r and dphi

    // now the object to produce the comparison plots is created
    
    // Plot Translations
    GeometryComparisonPlotter * trans = new GeometryComparisonPlotter (inFile, outDir);
    // x and y contain the couples to plot
    // -> every combination possible will be performed
    // /!\ always give units (otherwise, unexpected bug from root...)
    vector<TString> x,y;
    x.push_back("r");                                           	trans->SetBranchUnits("r",     "cm");  
    x.push_back("phi");                                         	trans->SetBranchUnits("phi",   "rad");
    x.push_back("z");                                           	trans->SetBranchUnits("z",     "cm");      //trans->SetBranchMax("z", 100); trans->SetBranchMin("z", -100);
    y.push_back("dr");		trans->SetBranchSF("dr", 	10000);     trans->SetBranchUnits("dr",    "#mum");
    y.push_back("dz");		trans->SetBranchSF("dz", 	10000);     trans->SetBranchUnits("dz",    "#mum");
    y.push_back("rdphi");	trans->SetBranchSF("rdphi",10000);      trans->SetBranchUnits("rdphi", "#mum rad");
    y.push_back("dx");		trans->SetBranchSF("dx", 	10000);     trans->SetBranchUnits("dx",    "#mum");    //trans->SetBranchMax("dx", 10); trans->SetBranchMin("dx", -10);
    y.push_back("dy");		trans->SetBranchSF("dy", 	10000);     trans->SetBranchUnits("dy",    "#mum");    //trans->SetBranchMax("dy", 10); trans->SetBranchMin("dy", -10);
    trans->MakePlots(x, y); // default output is pdf, but png gives a nicer result, so we use it as well
    // remark: what takes the more time is the creation of the output files,
    //         not the looping on the tree (because the code is perfect, of course :p)
    trans->SetGrid(1,1);
    trans->SetPrintOption("png");
    trans->MakePlots(x, y);

    
    // Plot Rotations
    GeometryComparisonPlotter * rot = new GeometryComparisonPlotter (inFile, outDir);
    // x and y contain the couples to plot
    // -> every combination possible will be performed
    // /!\ always give units (otherwise, unexpected bug from root...)
    vector<TString> a,b;
    a.push_back("alpha");       									rot->SetBranchUnits("alpha",    "rad");  
    a.push_back("beta");        									rot->SetBranchUnits("beta",   "rad");
    a.push_back("gamma");       									rot->SetBranchUnits("gamma",   "rad");
    b.push_back("dalpha");	rot->SetBranchSF("dalpha", 	1000);      rot->SetBranchUnits("dalpha",    "mrad");      
    b.push_back("dbeta");   rot->SetBranchSF("dbeta", 	1000);    	rot->SetBranchUnits("dbeta",    "mrad");     
    b.push_back("dgamma");  rot->SetBranchSF("dgamma", 	1000);    	rot->SetBranchUnits("dgamma",    "mrad");    
    rot->MakePlots(a, b); // default output is pdf, but png gives a nicer result, so we use it as well
    // remark: what takes the more time is the creation of the output files,
    //         not the looping on the tree (because the code is perfect, of course :p)
    rot->SetGrid(1,1);
    rot->SetPrintOption("png");
    rot->MakePlots(a, b);
    
    // Plot cross talk
    GeometryComparisonPlotter * cross = new GeometryComparisonPlotter (inFile, outDir);
    // x and y contain the couples to plot
    // -> every combination possible will be performed
    // /!\ always give units (otherwise, unexpected bug from root...)
    vector<TString> dx,dy;
    dx.push_back("dalpha"); cross->SetBranchSF("dalpha", 1000);     cross->SetBranchUnits("dalpha", "mrad");      
    dx.push_back("dbeta");  cross->SetBranchSF("dbeta", 1000);     	cross->SetBranchUnits("dbeta",  "mrad");     
    dx.push_back("dgamma"); cross->SetBranchSF("dgamma", 1000);     cross->SetBranchUnits("dgamma", "mrad"); 
    dy.push_back("dr");		cross->SetBranchSF("dr", 	10000);     cross->SetBranchUnits("dr",    "#mum");
    dy.push_back("dz");		cross->SetBranchSF("dz", 	10000);     cross->SetBranchUnits("dz",    "#mum");
    dy.push_back("rdphi");	cross->SetBranchSF("rdphi",10000);      cross->SetBranchUnits("rdphi", "#mum rad");
    dy.push_back("dx");		cross->SetBranchSF("dx", 	10000);     cross->SetBranchUnits("dx",    "#mum");  
    dy.push_back("dy");		cross->SetBranchSF("dy", 	10000);     cross->SetBranchUnits("dy",    "#mum");     
    cross->MakePlots(dx,dy); // default output is pdf, but png gives a nicer result, so we use it as well
    // remark: what takes the more time is the creation of the output files,
    //         not the looping on the tree (because the code is perfect, of course :p)
    cross->SetGrid(1,1);
    cross->SetPrintOption("png");
    cross->MakePlots(dx, dy);

    // now the same object can be reused with other specifications/cuts
    //void SetPrint               (const bool);           // activates the printing of the individual and global pdf
    //void SetLegend              (const bool);           // activates the legends
    //void SetWrite               (const bool);           // activates the writing into a Root file
    //void Set1dModule            (const bool);           // cut to include 1D modules
    //void Set2dModule            (const bool);           // cut to include 2D modules
    //void SetLevelCut            (const int);            // module level: level=1 (default)
    //void SetBatchMode           (const bool);           // activates the display of the canvases
    //void SetGrid                (const int,             // activates the display of the grids
    //                             const int);
    //void SetBranchMax           (const TString,         // sets a max value for the variable
    //                             const float);          // by giving the name and the value
    //void SetBranchMin           (const TString,         // sets a min value for the variable
    //                             const float);          // by giving the name and the value
    //void SetBranchSF            (const TString,         // sets a rescaling factor for the variable
    //                             const float);          // by giving the name and the value
    //void SetBranchUnits         (const TString,         // writes de units next on the axis
    //                             const TString);
    //void SetOutputDirectoryName (const TString);        // sets the output name of the directory
    //void SetOutputFileName      (const TString);        // sets the name of the root file (if applicable)
    //void SetPrintOption         (const Option_t *);     // litteraly the print option of the TPad::Print()
    //void SetCanvasSize          (const int window_width  = DEFAULT_WINDOW_WIDTH,
    //                             const int window_height = DEFAULT_WINDOW_HEIGHT);
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
