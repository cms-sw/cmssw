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
                       TString outDir="outputDir/",
                       TString modulesToPlot="all",
                       TString alignmentName="Alignment",
                       TString referenceName="Ideal",
                       TString useDefaultRange= "false",
                       float dx_min = -99999,
                       float dx_max = -99999,
                       float dy_min = -99999,
                       float dy_max = -99999,
                       float dz_min = -99999,
                       float dz_max = -99999,
                       float dr_min = -99999,
                       float dr_max = -99999,
                       float rdphi_min = -99999,
                       float rdphi_max = -99999,
                       float dalpha_min = -99999,
                       float dalpha_max = -99999,
                       float dbeta_min = -99999,
                       float dbeta_max = -99999,
                       float dgamma_min = -99999,
                       float dgamma_max = -99999
                       )
{
    // the output directory is created if it does not exist
    mkdir(outDir, S_IRWXU);
    
    // store y-ranges temporaly for the case when the default range is used together with individual ranges
    float dx_min_temp = dx_min;
    float dx_max_temp = dx_max;
    float dy_min_temp = dy_min;
    float dy_max_temp = dy_max;
    float dz_min_temp = dz_min;
    float dz_max_temp = dz_max;
    float dr_min_temp = dr_min;
    float dr_max_temp = dr_max;
    float rdphi_min_temp = rdphi_min;
    float rdphi_max_temp = rdphi_max;
    float dalpha_min_temp = dalpha_min;
    float dalpha_max_temp = dalpha_max;
    float dbeta_min_temp = dbeta_min;
    float dbeta_max_temp = dbeta_max;
    float dgamma_min_temp = dgamma_min;
    float dgamma_max_temp = dgamma_max;
    
    if (useDefaultRange != "false"){
		dx_min = -200;
		dx_max = 200;
		dy_min = -200;
		dy_max = 200;
		dz_min = -200;
		dz_max = 200;
		dr_min = -200;
		dr_max = 200;
		rdphi_min = -200;
		rdphi_max = 200;
		
		dalpha_min = -100;
		dalpha_max = 100;
		dbeta_min = -100;
		dbeta_max = 100;
		dgamma_min = -100;
		dgamma_max = 100;
		
		// Overwrite single default values if individual ones are given
		if (dx_min_temp != -99999){dx_min = dx_min_temp;}
		if (dx_max_temp != -99999){dx_max = dx_max_temp;}
		if (dy_min_temp != -99999){dy_min = dy_min_temp;}
		if (dy_max_temp != -99999){dy_max = dy_max_temp;}
		if (dz_min_temp != -99999){dz_min = dz_min_temp;}
		if (dz_max_temp != -99999){dz_max = dz_max_temp;}
		if (dr_min_temp != -99999){dr_min = dr_min_temp;}
		if (dr_max_temp != -99999){dr_max = dr_max_temp;}
		if (rdphi_min_temp != -99999){rdphi_min = rdphi_min_temp;}
		if (rdphi_max_temp != -99999){rdphi_max = rdphi_max_temp;}
		
		if (dalpha_min_temp != -99999){dalpha_min = dalpha_min_temp;}
		if (dalpha_max_temp != -99999){dalpha_max = dalpha_max_temp;}
		if (dbeta_min_temp != -99999){dbeta_min = dbeta_min_temp;}
		if (dbeta_max_temp != -99999){dbeta_max = dbeta_max_temp;}
		if (dgamma_min_temp != -99999){dgamma_min = dgamma_min_temp;}
		if (dgamma_max_temp != -99999){dgamma_max = dgamma_max_temp;}
	}
	
	
    // display canvases: be careful, as there are many many ... canvases

    // the name of the variables are the names of the branches
    // REMARK: a supplementary branch for rdphi is calculated automatically
    //         from r and dphi

    // now the object to produce the comparison plots is created
    
    // Plot Translations
    GeometryComparisonPlotter * trans = new GeometryComparisonPlotter (inFile, outDir,modulesToPlot,alignmentName,referenceName);
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
    if (dx_min != -99999){trans->SetBranchMin("dx", 	dx_min);}
    if (dx_max != -99999){trans->SetBranchMax("dx", 	dx_max);}
    if (dy_min != -99999){trans->SetBranchMin("dy", 	dy_min);}
    if (dy_max != -99999){trans->SetBranchMax("dy", 	dy_max);}
    if (dz_min != -99999){trans->SetBranchMin("dz", 	dz_min);}
    if (dz_max != -99999){trans->SetBranchMax("dz", 	dz_max);}
    if (dr_min != -99999){trans->SetBranchMin("dr", 	dr_min);}
    if (dr_max != -99999){trans->SetBranchMax("dr", 	dr_max);}
    if (rdphi_min != -99999){trans->SetBranchMin("rdphi", 	rdphi_min);}
    if (rdphi_max != -99999){trans->SetBranchMax("rdphi", 	rdphi_max);}
    trans->SetGrid(1,1);
    trans->MakePlots(x, y); // default output is pdf, but png gives a nicer result, so we use it as well
    // remark: what takes the more time is the creation of the output files,
    //         not the looping on the tree (because the code is perfect, of course :p)
    trans->SetPrintOption("png");
    trans->MakePlots(x, y);

    
    // Plot Rotations
    GeometryComparisonPlotter * rot = new GeometryComparisonPlotter (inFile, outDir,modulesToPlot,alignmentName,referenceName);
    // x and y contain the couples to plot
    // -> every combination possible will be performed
    // /!\ always give units (otherwise, unexpected bug from root...)
    vector<TString> b;
    //a.push_back("alpha");       									rot->SetBranchUnits("alpha",    "rad");  
    //a.push_back("beta");        									rot->SetBranchUnits("beta",   "rad");
    //a.push_back("gamma");       									rot->SetBranchUnits("gamma",   "rad");
    rot->SetBranchUnits("r",     "cm");  
    rot->SetBranchUnits("phi",   "rad");
    rot->SetBranchUnits("z",     "cm"); 
    b.push_back("dalpha");	rot->SetBranchSF("dalpha", 	1000);      rot->SetBranchUnits("dalpha",    "mrad");      
    b.push_back("dbeta");   rot->SetBranchSF("dbeta", 	1000);    	rot->SetBranchUnits("dbeta",    "mrad");     
    b.push_back("dgamma");  rot->SetBranchSF("dgamma", 	1000);    	rot->SetBranchUnits("dgamma",    "mrad");
    if (dalpha_min != -99999){rot->SetBranchMin("dalpha", 	dalpha_min);}
    if (dalpha_max != -99999){rot->SetBranchMax("dalpha", 	dalpha_max);}   
    if (dbeta_min != -99999){rot->SetBranchMin("dbeta", 	dbeta_min);}
    if (dbeta_max != -99999){rot->SetBranchMax("dbeta", 	dbeta_max);}   
    if (dgamma_min != -99999){rot->SetBranchMin("dgamma", 	dgamma_min);}
    if (dgamma_max != -99999){rot->SetBranchMax("dgamma", 	dgamma_max);}
    rot->SetGrid(1,1);
    rot->SetPrintOption("pdf");
    rot->MakePlots(x, b);
    rot->SetPrintOption("png");
    rot->MakePlots(x, b);
    

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
