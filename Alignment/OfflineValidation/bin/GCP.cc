#include <cstdlib>
#include <iostream>
#include <vector>

#include <TString.h>

#include "exceptions.h"
#include "toolbox.h"
#include "Options.h"

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "GeometryComparisonPlotter.h"

using namespace std;
using namespace AllInOneConfig;

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;

void comparisonScript (TString inFile,//="mp1510_vs_mp1509.Comparison_commonTracker.root", // TODO: get ROOT file
                       TString outDir="outputDir/",
                       TString modulesToPlot="all",
                       TString alignmentName="Alignment",
                       TString referenceName="Ideal",
                       bool useDefaultRange= false,
                       bool plotOnlyGlobal= false,
                       bool plotPng= true,
                       bool makeProfilePlots= true,
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
    fs::create_directories(outDir.Data());
    
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
    
    if (useDefaultRange){
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
    // REMARK: an additional branch for rdphi is calculated automatically from r and dphi

    // now the object to produce the comparison plots is created
    
    // Plot Translations
    GeometryComparisonPlotter * trans = new GeometryComparisonPlotter(inFile, outDir,modulesToPlot,alignmentName,referenceName,plotOnlyGlobal,makeProfilePlots);
    // x and y contain the couples to plot
    // -> every combination possible will be performed
    // /!\ always give units (otherwise, unexpected bug from root...)
    vector<TString> x,y, xmean;
    vector<float> dyMin,dyMax;
    x.push_back("r");                                           	trans->SetBranchUnits("r",     "cm");
    x.push_back("phi");                                         	trans->SetBranchUnits("phi",   "rad");
    x.push_back("z");                                           	trans->SetBranchUnits("z",     "cm");      //trans->SetBranchMax("z", 100); trans->SetBranchMin("z", -100);
    y.push_back("dr");		trans->SetBranchSF("dr", 	10000);     trans->SetBranchUnits("dr",    "#mum");
    dyMin.push_back(dr_min);
    dyMax.push_back(dr_max);
    y.push_back("dz");		trans->SetBranchSF("dz", 	10000);     trans->SetBranchUnits("dz",    "#mum");
    dyMin.push_back(dz_min);
    dyMax.push_back(dz_max);
    y.push_back("rdphi");	trans->SetBranchSF("rdphi", 10000);      trans->SetBranchUnits("rdphi", "#mum rad");
    dyMin.push_back(rdphi_min);
    dyMax.push_back(rdphi_max);
    y.push_back("dx");		trans->SetBranchSF("dx", 	10000);     trans->SetBranchUnits("dx",    "#mum");    //trans->SetBranchMax("dx", 10); trans->SetBranchMin("dx", -10);
    dyMin.push_back(dx_min);
    dyMax.push_back(dx_max);
    y.push_back("dy");		trans->SetBranchSF("dy", 	10000);     trans->SetBranchUnits("dy",    "#mum");    //trans->SetBranchMax("dy", 10); trans->SetBranchMin("dy", -10);
    dyMin.push_back(dy_min);
    dyMax.push_back(dy_max);
    
    xmean.push_back("x");                                         	
    trans->SetBranchUnits("x",     "cm");
    xmean.push_back("y");                                           	
    trans->SetBranchUnits("y",   "cm");
    xmean.push_back("z");                         
    xmean.push_back("r");                     
    
    
    trans->SetGrid(1,1);
    trans->MakePlots(x, y, dyMin, dyMax); // default output is pdf, but png gives a nicer result, so we use it as well
    // remark: what takes the more time is the creation of the output files,
    //         not the looping on the tree (because the code is perfect, of course :p)
    if (plotPng){
	    trans->SetPrintOption("png");
	    trans->MakePlots(x, y, dyMin, dyMax);
	}
	
	trans->MakeTables(xmean,y,dyMin,dyMax);

    
    // Plot Rotations
    GeometryComparisonPlotter * rot = new GeometryComparisonPlotter(inFile, outDir,modulesToPlot,alignmentName,referenceName,plotOnlyGlobal,makeProfilePlots);
    // x and y contain the couples to plot
    // -> every combination possible will be performed
    // /!\ always give units (otherwise, unexpected bug from root...)
    vector<TString> b;
    vector<float> dbMin,dbMax;
    rot->SetBranchUnits("r",     "cm");  
    rot->SetBranchUnits("phi",   "rad");
    rot->SetBranchUnits("z",     "cm"); 
    b.push_back("dalpha");	rot->SetBranchSF("dalpha", 	1000);      rot->SetBranchUnits("dalpha",    "mrad");
    dbMin.push_back(dalpha_min);
    dbMax.push_back(dalpha_max);      
    b.push_back("dbeta");   rot->SetBranchSF("dbeta", 	1000);    	rot->SetBranchUnits("dbeta",    "mrad");
    dbMin.push_back(dbeta_min);
    dbMax.push_back(dbeta_max);     
    b.push_back("dgamma");  rot->SetBranchSF("dgamma", 	1000);    	rot->SetBranchUnits("dgamma",    "mrad");
    dbMin.push_back(dgamma_min);
    dbMax.push_back(dgamma_max);
 
    rot->SetGrid(1,1);
    rot->SetPrintOption("pdf");
    rot->MakePlots(x, b,dbMin, dbMax);
    if (plotPng){
	    rot->SetPrintOption("png");
	    rot->MakePlots(x, b,dbMin, dbMax);
	}	
}

int GCP (int argc, char * argv[])
{
    // parse the command line
    Options options;
    options.helper(argc, argv);
    options.parser(argc, argv);

    pt::ptree main_tree;
    pt::read_json(options.config, main_tree);

    // TODO
    // - comments à la doxygen
    // - get ROOT file (look into All-In-One Tool)
    // - use Boost to read config file
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main (int argc, char * argv[])
{
    return exceptions<GCP>(argc, argv);
}
#endif
