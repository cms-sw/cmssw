#include <cstdlib>
#include <iostream>
#include <vector>

#include <TString.h>

#include "exceptions.h"
#include "toolbox.h"
#include "Options.h"

#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <Alignment/OfflineValidation/interface/GeometryComparisonPlotter.h>
#include <Alignment/OfflineValidation/scripts/visualizationTracker.C>
#include <Alignment/OfflineValidation/macros/makeArrowPlots.C>
//#include "GeometryComparisonPlotter.h"

// for debugging
#include <TObject.h>

using namespace std;
using namespace AllInOneConfig;

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;

void comparisonScript(
        TString inFile,  //="mp1510_vs_mp1509.Comparison_commonTracker.root", // TODO: get ROOT file
        TString outDir = "outputDir/",
        TString modulesToPlot = "all",
        TString alignmentName = "Alignment",
        TString referenceName = "Ideal",
        bool useDefaultRange = false,
        bool plotOnlyGlobal = false,
        bool plotPng = true,
        bool makeProfilePlots = true,
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
        float dgamma_max = -99999) 
{
  // the output directory is created if it does not exist
  fs::create_directories(outDir.Data());

  TString transDir = outDir+"/Translations";
  TString rotDir = outDir+"/Rotations";
  fs::create_directories(transDir.Data());
  fs::create_directories(rotDir.Data());

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

  if (useDefaultRange) {
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
    if (dx_min_temp != -99999) {
      dx_min = dx_min_temp;
    }
    if (dx_max_temp != -99999) {
      dx_max = dx_max_temp;
    }
    if (dy_min_temp != -99999) {
      dy_min = dy_min_temp;
    }
    if (dy_max_temp != -99999) {
      dy_max = dy_max_temp;
    }
    if (dz_min_temp != -99999) {
      dz_min = dz_min_temp;
    }
    if (dz_max_temp != -99999) {
      dz_max = dz_max_temp;
    }
    if (dr_min_temp != -99999) {
      dr_min = dr_min_temp;
    }
    if (dr_max_temp != -99999) {
      dr_max = dr_max_temp;
    }
    if (rdphi_min_temp != -99999) {
      rdphi_min = rdphi_min_temp;
    }
    if (rdphi_max_temp != -99999) {
      rdphi_max = rdphi_max_temp;
    }

    if (dalpha_min_temp != -99999) {
      dalpha_min = dalpha_min_temp;
    }
    if (dalpha_max_temp != -99999) {
      dalpha_max = dalpha_max_temp;
    }
    if (dbeta_min_temp != -99999) {
      dbeta_min = dbeta_min_temp;
    }
    if (dbeta_max_temp != -99999) {
      dbeta_max = dbeta_max_temp;
    }
    if (dgamma_min_temp != -99999) {
      dgamma_min = dgamma_min_temp;
    }
    if (dgamma_max_temp != -99999) {
      dgamma_max = dgamma_max_temp;
    }
  }

  // display canvases: be careful, as there are many many ... canvases

  // the name of the variables are the names of the branches
  // REMARK: an additional branch for rdphi is calculated automatically from r and dphi

  // now the object to produce the comparison plots is created

  // Plot Translations
  GeometryComparisonPlotter* trans = new GeometryComparisonPlotter(
      inFile, transDir, modulesToPlot, alignmentName, referenceName, plotOnlyGlobal, makeProfilePlots, 0);
  // x and y contain the couples to plot
  // -> every combination possible will be performed
  // /!\ always give units (otherwise, unexpected bug from root...)
  vector<TString> x, y, xmean;
  vector<float> dyMin, dyMax;
  x.push_back("r");
  trans->SetBranchUnits("r", "cm");
  x.push_back("phi");
  trans->SetBranchUnits("phi", "rad");
  x.push_back("z");
  trans->SetBranchUnits("z", "cm");  //trans->SetBranchMax("z", 100); trans->SetBranchMin("z", -100);
  y.push_back("dr");
  trans->SetBranchSF("dr", 10000);
  trans->SetBranchUnits("dr", "#mum");
  dyMin.push_back(dr_min);
  dyMax.push_back(dr_max);
  y.push_back("dz");
  trans->SetBranchSF("dz", 10000);
  trans->SetBranchUnits("dz", "#mum");
  dyMin.push_back(dz_min);
  dyMax.push_back(dz_max);
  y.push_back("rdphi");
  trans->SetBranchSF("rdphi", 10000);
  trans->SetBranchUnits("rdphi", "#mum rad");
  dyMin.push_back(rdphi_min);
  dyMax.push_back(rdphi_max);
  y.push_back("dx");
  trans->SetBranchSF("dx", 10000);
  trans->SetBranchUnits("dx", "#mum");  //trans->SetBranchMax("dx", 10); trans->SetBranchMin("dx", -10);
  dyMin.push_back(dx_min);
  dyMax.push_back(dx_max);
  y.push_back("dy");
  trans->SetBranchSF("dy", 10000);
  trans->SetBranchUnits("dy", "#mum");  //trans->SetBranchMax("dy", 10); trans->SetBranchMin("dy", -10);
  dyMin.push_back(dy_min);
  dyMax.push_back(dy_max);

  xmean.push_back("x");
  trans->SetBranchUnits("x", "cm");
  xmean.push_back("y");
  trans->SetBranchUnits("y", "cm");
  xmean.push_back("z");
  xmean.push_back("r");

  trans->SetGrid(1, 1);
  trans->MakePlots(x, y, dyMin, dyMax);  // default output is pdf, but png gives a nicer result, so we use it as well
  // remark: what takes the more time is the creation of the output files,
  //         not the looping on the tree (because the code is perfect, of course :p)
  if (plotPng) {
    trans->SetPrintOption("png");
    trans->MakePlots(x, y, dyMin, dyMax);
  }

  trans->MakeTables(xmean, y, dyMin, dyMax);

  // Plot Rotations
  GeometryComparisonPlotter* rot = new GeometryComparisonPlotter(
      inFile, rotDir, modulesToPlot, alignmentName, referenceName, plotOnlyGlobal, makeProfilePlots, 2);
  // x and y contain the couples to plot
  // -> every combination possible will be performed
  // /!\ always give units (otherwise, unexpected bug from root...)
  vector<TString> b;
  vector<float> dbMin, dbMax;
  rot->SetBranchUnits("r", "cm");
  rot->SetBranchUnits("phi", "rad");
  rot->SetBranchUnits("z", "cm");
  b.push_back("dalpha");
  rot->SetBranchSF("dalpha", 1000);
  rot->SetBranchUnits("dalpha", "mrad");
  dbMin.push_back(dalpha_min);
  dbMax.push_back(dalpha_max);
  b.push_back("dbeta");
  rot->SetBranchSF("dbeta", 1000);
  rot->SetBranchUnits("dbeta", "mrad");
  dbMin.push_back(dbeta_min);
  dbMax.push_back(dbeta_max);
  b.push_back("dgamma");
  rot->SetBranchSF("dgamma", 1000);
  rot->SetBranchUnits("dgamma", "mrad");
  dbMin.push_back(dgamma_min);
  dbMax.push_back(dgamma_max);

  rot->SetGrid(1, 1);
  rot->SetPrintOption("pdf");
  rot->MakePlots(x, b, dbMin, dbMax);
  if (plotPng) {
    rot->SetPrintOption("png");
    rot->MakePlots(x, b, dbMin, dbMax);
  }
}

void vizualizationScript(
        TString inFile,
        TString outDir,
        TString alignmentName, 
        TString referenceName)
{
  TString outputFileName = outDir+"/Visualization"; 
  fs::create_directories(outputFileName.Data());
  //title
  std::string line1 = alignmentName.Data();
  std::string line2 = referenceName.Data();
  //set subdetectors to see
  int subdetector1 = 1;
  int subdetector2 = 2;
  //translation scale factor
  int sclftr = 50;
  //rotation scale factor
  int sclfrt = 1;
  //module size scale factor
  float sclfmodulesizex = 1;
  float sclfmodulesizey = 1;
  float sclfmodulesizez = 1;
  //beam pipe radius
  float piperadius = 2.25;
  //beam pipe xy coordinates
  float pipexcoord = 0;
  float pipeycoord = 0;
  //beam line xy coordinates
  float linexcoord = 0;
  float lineycoord = 0;
  runVisualizer(inFile,
                outputFileName.Data(),
                line1,
                line2,
                subdetector1,
                subdetector2,
                sclftr,
                sclfrt,
                sclfmodulesizex,
                sclfmodulesizey,
                sclfmodulesizez,
                piperadius,
                pipexcoord,
                pipeycoord,
                linexcoord,
                lineycoord );

}

int GCP(int argc, char* argv[]) {
  /*
  TObject* printer = new TObject();
  printer->Info("GCPvalidation", "Hello!");
  // Hack to push through messages even without -v running
  // Very ugly coding, to run with std::cout -> run with -v option (GCP cfg.json -v)
  */


  std::cout << " ----- GCP validation plots -----" << std::endl;
  std::cout << " --- Digesting configuration" << std::endl;

  // parse the command line
  Options options;
  options.helper(argc, argv);
  options.parser(argc, argv);

  pt::ptree main_tree;
  pt::read_json(options.config, main_tree);

  pt::ptree alignments = main_tree.get_child("alignments");
  pt::ptree validation = main_tree.get_child("validation");

  pt::ptree GCPoptions = validation.get_child("GCP");

  pt::ptree comAl = alignments.get_child("comp");
  pt::ptree refAl = alignments.get_child("ref");

  // Read the options
  TString inFile = main_tree.get<std::string>("output") + "/GCPtree.root";
  TString outDir = main_tree.get<std::string>("output");
  TString modulesToPlot = "all";
  TString alignmentName = comAl.get<std::string>("title");
  TString referenceName = refAl.get<std::string>("title");
  bool useDefaultRange  = GCPoptions.get_child_optional("useDefaultRange") ? GCPoptions.get<bool>("useDefaultRange") : false;
  bool plotOnlyGlobal   = GCPoptions.get_child_optional("plotOnlyGlobal") ? GCPoptions.get<bool>("plotOnlyGlobal") : false;
  bool plotPng          = GCPoptions.get_child_optional("plotPng") ? GCPoptions.get<bool>("plotPng") : false;
  bool makeProfilePlots = GCPoptions.get_child_optional("makeProfilePlots") ? GCPoptions.get<bool>("makeProfilePlots") : true;
  float dx_min = GCPoptions.get_child_optional("dx_min") ? GCPoptions.get<float>("dx_min") : -99999;
  float dx_max = GCPoptions.get_child_optional("dx_max") ? GCPoptions.get<float>("dx_max") : -99999;
  float dy_min = GCPoptions.get_child_optional("dy_min") ? GCPoptions.get<float>("dy_min") : -99999;
  float dy_max = GCPoptions.get_child_optional("dy_max") ? GCPoptions.get<float>("dy_max") : -99999;
  float dz_min = GCPoptions.get_child_optional("dz_min") ? GCPoptions.get<float>("dz_min") : -99999;
  float dz_max = GCPoptions.get_child_optional("dz_max") ? GCPoptions.get<float>("dz_max") : -99999;
  float dr_min = GCPoptions.get_child_optional("dr_min") ? GCPoptions.get<float>("dr_min") : -99999;
  float dr_max = GCPoptions.get_child_optional("dr_max") ? GCPoptions.get<float>("dr_max") : -99999;
  float rdphi_min  = GCPoptions.get_child_optional("rdphi_min") ? GCPoptions.get<float>("rdphi_min") : -99999;
  float rdphi_max  = GCPoptions.get_child_optional("rdphi_max") ? GCPoptions.get<float>("rdphi_max") : -99999;
  float dalpha_min = GCPoptions.get_child_optional("dalpha_min") ? GCPoptions.get<float>("dalpha_min") : -99999;
  float dalpha_max = GCPoptions.get_child_optional("dalpha_max") ? GCPoptions.get<float>("dalpha_max") : -99999;
  float dbeta_min  = GCPoptions.get_child_optional("dbeta_min") ? GCPoptions.get<float>("dbeta_min") : -99999;
  float dbeta_max  = GCPoptions.get_child_optional("dbeta_max") ? GCPoptions.get<float>("dbeta_max") : -99999;
  float dgamma_min = GCPoptions.get_child_optional("dgamma_min") ? GCPoptions.get<float>("dgamma_min") : -99999;
  float dgamma_max = GCPoptions.get_child_optional("dgamma_max") ? GCPoptions.get<float>("dgamma_max") : -99999;


  std::cout << " --- Running comparison script" << std::endl;
  // Compare script
  comparisonScript(
          inFile,
          outDir,
          modulesToPlot,
          alignmentName,
          referenceName,
          useDefaultRange,
          plotOnlyGlobal,
          plotPng,
          makeProfilePlots,
          dx_min,
          dx_max,
          dy_min,
          dy_max,
          dz_min,
          dz_max,
          dr_min,
          dr_max,
          rdphi_min,
          rdphi_max,
          dalpha_min,
          dalpha_max,
          dbeta_min,
          dbeta_max,
          dgamma_min,
          dgamma_max);


  std::cout << " --- Running visualization script" << std::endl;
  // Visualization script
  vizualizationScript(
        inFile,
        outDir,
        alignmentName, 
        referenceName);

  std::cout << " --- Running arrow plot script" << std::endl;
  // Arrow plot
  TString arrowDir = outDir+"/ArrowPlots"; 
  makeArrowPlots(
        inFile.Data(), 
        arrowDir.Data());

  // TODO
  // - comments à la doxygen
  // - get ROOT file (look into All-In-One Tool)
  // - use Boost to read config file
  return EXIT_SUCCESS;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char* argv[]) { return exceptions<GCP>(argc, argv); }
#endif
