//#include "GeometryComparisonPlotter.h"
#include "Alignment/OfflineValidation/interface/GeometryComparisonPlotter.h"

/***********************************************************************************/
/*                            GEOMETRY COMPARISON PLOTTER                          */
/* See the talk of 15 January 2015 for short documentation and the example script. */
/* This code is highly commented if need be to upgrade it.                         */
/* Any further question is to be asked to Patrick Connor (patrick.connor@desy.de). */
/*                                             Thanks a million <3                 */
/***********************************************************************************/

// NOTE: look for "TO DO" as a keyword to now what should be upgraded in later versions....

// modes
//#define TALKATIVE  // get some comments while processing
//#define DEBUG     // get a lot of comments while processing + canvases -> resource-consuming!

// MACROS
#define INSIDE_VECTOR(vector)                          \
  std::cout << #vector << "={";                        \
  for (unsigned int i = 0; i < vector.size() - 1; i++) \
    std::cout << vector[i] << ",";                     \
  std::cout << vector.back() << "}";
#define CHECK_MAP_CONTENT(m, type)                                            \
  for (std::map<TString, type>::iterator it = m.begin(); it != m.end(); it++) \
    std::cout << __FILE__ << ":" << __LINE__ << ":Info: " << #m << "[" << it->first << "]=" << it->second << std::endl;
/*
    TString _sublevel_names[NB_SUBLEVELS],
            _output_directory,
            _output_filename,
            _print_option,
            _module_plot_option,
            _alignment_name,
            _reference_name;
    bool _print,
         _legend,
         _write,
         _print_only_global,
         _make_profile_plots,
         _batchMode,
         _1dModule,
         _2dModule;
    int _levelCut,
        _grid_x,
        _grid_y,
        _window_width,
        _window_height;
 */

// CONSTRUCTOR AND DESTRUCTOR
GeometryComparisonPlotter::GeometryComparisonPlotter(TString tree_file_name,
                                                     TString output_directory,
                                                     TString modulesToPlot,
                                                     TString alignmentName,
                                                     TString referenceName,
                                                     bool printOnlyGlobal,
                                                     bool makeProfilePlots,
                                                     int canvas_idx)
    : _output_directory(output_directory + TString(output_directory.EndsWith("/") ? "" : "/")),
      _output_filename("comparison.root"),
      _print_option("pdf"),
      _module_plot_option(modulesToPlot),
      _alignment_name(alignmentName),
      _reference_name(referenceName),
      _print(true),   // print the graphs in a file (e.g. pdf)
      _legend(true),  // print the graphs in a file (e.g. pdf)
      _write(true),   // write the graphs in a root file
      _print_only_global(printOnlyGlobal),
      _make_profile_plots(makeProfilePlots),
      _batchMode(
#ifdef DEBUG
          false  // false = display canvases (very time- and resource-consuming)
#else
          true  // true = no canvases
#endif
          ),
      _1dModule(true),           // cut on 1d modules
      _2dModule(true),           // cut on 2d modules
      _levelCut(DEFAULT_LEVEL),  // module level (see branch of same name)
      _grid_x(0),                // by default no display the grid in the canvases
      _grid_y(0),                // by default no display the grid in the canvases
      _window_width(DEFAULT_WINDOW_WIDTH),
      _window_height(DEFAULT_WINDOW_HEIGHT),
      _canvas_index(canvas_idx) {
#ifdef TALKATIVE
  std::cout << ">>> TALKATIVE MODE ACTIVATED <<<" << std::endl;
#endif
#ifdef DEBUG
  std::cout << ">>> DEBUG MODE ACTIVATED <<<" << std::endl;
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: inside constructor of GeometryComparisonPlotter utility"
            << std::endl;
#endif
  // _canvas_index
  //_canvas_index = 0;

  //_sublevel_names = {"PXB", "PXF", "TIB", "TID", "TOB", "TEC"}; // C++11
  _sublevel_names[0] = TString("PXB");
  _sublevel_names[1] = TString("PXF");
  _sublevel_names[2] = TString("TIB");
  _sublevel_names[3] = TString("TID");
  _sublevel_names[4] = TString("TOB");
  _sublevel_names[5] = TString("TEC");
  // TO DO: handle other structures

  // read tree
  tree_file = new TFile(tree_file_name, "UPDATE");
  data = (TTree *)tree_file->Get("alignTree");
  // int branches
  data->SetBranchAddress("id", &branch_i["id"]);
  data->SetBranchAddress("inModuleList", &branch_i["inModuleList"]);
  data->SetBranchAddress("badModuleQuality", &branch_i["badModuleQuality"]);
  data->SetBranchAddress("mid", &branch_i["mid"]);
  data->SetBranchAddress("level", &branch_i["level"]);
  data->SetBranchAddress("mlevel", &branch_i["mlevel"]);
  data->SetBranchAddress("sublevel", &branch_i["sublevel"]);
  data->SetBranchAddress("useDetId", &branch_i["useDetId"]);
  data->SetBranchAddress("detDim", &branch_i["detDim"]);
  // float branches
  data->SetBranchAddress("x", &branch_f["x"]);
  data->SetBranchAddress("y", &branch_f["y"]);
  data->SetBranchAddress("z", &branch_f["z"]);
  data->SetBranchAddress("alpha", &branch_f["alpha"]);
  data->SetBranchAddress("beta", &branch_f["beta"]);
  data->SetBranchAddress("gamma", &branch_f["gamma"]);
  data->SetBranchAddress("phi", &branch_f["phi"]);
  data->SetBranchAddress("eta", &branch_f["eta"]);
  data->SetBranchAddress("r", &branch_f["r"]);
  data->SetBranchAddress("dx", &branch_f["dx"]);
  data->SetBranchAddress("dy", &branch_f["dy"]);
  data->SetBranchAddress("dz", &branch_f["dz"]);
  data->SetBranchAddress("dphi", &branch_f["dphi"]);
  data->SetBranchAddress("dr", &branch_f["dr"]);
  data->SetBranchAddress("dalpha", &branch_f["dalpha"]);
  data->SetBranchAddress("dbeta", &branch_f["dbeta"]);
  data->SetBranchAddress("dgamma", &branch_f["dgamma"]);
  if (data->GetBranch("rdphi") ==
      nullptr)  // in the case of rdphi branch not existing, it is created from r and dphi branches
  {
#ifdef TALKATIVE
    std::cout << __FILE__ << ":" << __LINE__
              << ":Info: computing the rdphi branch from r and dphi branches (assuming they exist...)" << std::endl;
#endif
    TBranch *br_rdphi = data->Branch("rdphi", &branch_f["rdphi"], "rdphi/F");
    for (unsigned int ientry = 0; ientry < data->GetEntries(); ientry++) {
      data->GetEntry(ientry);
      branch_f["rdphi"] = branch_f["r"] * branch_f["dphi"];
      br_rdphi->Fill();
    }
  } else
    data->SetBranchAddress("rdphi", &branch_f["rdphi"]);

#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: branch addresses set" << std::endl;
#endif

  // style
  data->SetMarkerSize(0.5);
  data->SetMarkerStyle(6);

  gStyle->SetOptStat("emr");
  gStyle->SetTitleAlign(22);
  gStyle->SetTitleX(0.5);
  gStyle->SetTitleY(0.97);
  gStyle->SetTitleFont(62);
  //gStyle->SetOptTitle(0);

  gStyle->SetTextFont(132);
  gStyle->SetTextSize(0.08);
  gStyle->SetLabelFont(132, "x");
  gStyle->SetLabelFont(132, "y");
  gStyle->SetLabelFont(132, "z");
  gStyle->SetTitleSize(0.08, "x");
  gStyle->SetTitleSize(0.08, "y");
  gStyle->SetTitleSize(0.08, "z");
  gStyle->SetLabelSize(0.08, "x");
  gStyle->SetLabelSize(0.08, "y");
  gStyle->SetLabelSize(0.08, "z");

  gStyle->SetMarkerStyle(8);
  gStyle->SetHistLineWidth(2);
  gStyle->SetLineStyleString(2, "[12 12]");  // postscript dashes

  gStyle->SetFrameBorderMode(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetTitleColor(1);
  gStyle->SetStatColor(0);
  gStyle->SetStatBorderSize(1);
  gStyle->SetFrameFillColor(0);

  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.18);

#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: end of constructor" << std::endl;
#endif
}

GeometryComparisonPlotter::~GeometryComparisonPlotter() {
#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: in destructor of the GeometryComparisonPlotter utility"
            << std::endl;
#endif
  tree_file->Close();
#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: ending." << std::endl;
#endif
}

// MAIN METHOD
void GeometryComparisonPlotter::MakePlots(
    std::vector<TString> x,  // axes to combine to plot
    std::vector<TString> y,  // every combination (except the ones such that x=y) will be perfomed
    pt::ptree CFG)           // property tree storing the min and max values under <var>_min and <var>_max
{
  /// -1) check that only existing branches are called
  // (we use a macro to avoid copy/paste)
#define CHECK_BRANCHES(branchname_vector)                                                       \
  for (unsigned int i = 0; i < branchname_vector.size(); i++) {                                 \
    if (branch_f.find(branchname_vector[i]) == branch_f.end()) {                                \
      std::cout << __FILE__ << ":" << __LINE__ << ":Error: The branch " << branchname_vector[i] \
                << " is not recognised." << std::endl;                                          \
      return;                                                                                   \
    }                                                                                           \
  }
  CHECK_BRANCHES(x);
  CHECK_BRANCHES(y);

  const unsigned int nentries = data->GetEntries();

#ifdef TALKATIVE
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: ";
  INSIDE_VECTOR(x);
  std::cout << std::endl << __FILE__ << ":" << __LINE__ << ":Info: ";
  INSIDE_VECTOR(y);
  std::cout << std::endl;
#endif

  /// 0) min and max values
  // the max and min of the graphs are computed from the tree if they have not been manually input yet
  // (we use a macro to avoid copy/paste)
#define LIMITS(axes_vector)                                                          \
  for (unsigned int i = 0; i < axes_vector.size(); i++) {                            \
    if (_SF.find(axes_vector[i]) == _SF.end())                                       \
      _SF[axes_vector[i]] = 1.;                                                      \
    if (_min.find(axes_vector[i]) == _min.end())                                     \
      _min[axes_vector[i]] = _SF[axes_vector[i]] * data->GetMinimum(axes_vector[i]); \
    if (_max.find(axes_vector[i]) == _max.end())                                     \
      _max[axes_vector[i]] = _SF[axes_vector[i]] * data->GetMaximum(axes_vector[i]); \
  }
  LIMITS(x);
  LIMITS(y);

#ifdef TALKATIVE
  CHECK_MAP_CONTENT(_min, float);
  CHECK_MAP_CONTENT(_max, float);
  CHECK_MAP_CONTENT(_SF, float);
#endif

  /// 1) declare TGraphs and Histograms for profile plots if these are to be plotted
  // the idea is to produce at the end a table of 8 TMultiGraphs and histograms:
  // - 0=Tracker, with color code for the different sublevels
  // - 1..6=different sublevels, with color code for z < or > 0
  // - 7=only pixel with color code for BPIX and FPIX

  // (convention: the six first (resp. last) correspond to z>0 (resp. z<0))
  // Modules with bad quality and in a list of modules that is given
  // by the user (e.g. list of bad/untouched modules, default: empty list)
  // are stored in seperate graphs and might be plotted (depends on the module
  // plot option, default: all modules plotted)
  // This means that 3*2*6 TGraphs will be filled during the loop on the TTree,
  // and will be arranged differently with different color codes in the TMultiGraphs

  // For the profile plots
  // Either all modules, only good modules or good modules + those in a given list will be plotted
  // This means that 2*6 TH2F will be filled during the loop on the TTree,
  // and will be arranged differently with different color codes in the Histograms
#ifndef NB_SUBLEVELS
#define NB_SUBLEVELS 6
#endif
#define NB_Z_SLICES 2
#define NB_MODULE_QUALITY 3
#define COLOR_CODE(icolor) int(icolor / 4) + icolor + 1

  TGraph *graphs[x.size()][y.size()][NB_SUBLEVELS * NB_Z_SLICES * NB_MODULE_QUALITY];
  long int ipoint[x.size()][y.size()][NB_SUBLEVELS * NB_Z_SLICES * NB_MODULE_QUALITY];

  TMultiGraph
      *mgraphs[x.size()][y.size()]
              [2 + NB_SUBLEVELS];  // the 0th is for global plots, the 1..6th for sublevel plots, 7th for pixel only
  TCanvas *c[x.size()][y.size()][2 + NB_SUBLEVELS], *c_global[2 + NB_SUBLEVELS];
  _canvas_index++;  // this static index is a safety used in case the MakePlots method is used several times to avoid overloading

  // histograms for profile plots,
  // 2D-hists to store the data
  // 1D-hists to calculate mean and sigma of y-values for each x-bin of the 2D-hists and for the final profile hist
  TH2F *histos2D[x.size()][y.size()][NB_SUBLEVELS * NB_Z_SLICES];
  TH1F *histos[x.size()][y.size()][NB_SUBLEVELS * NB_Z_SLICES];
  TH1F *histosYValues[x.size()][y.size()]
                     [NB_SUBLEVELS * NB_Z_SLICES];  // Used to calculate the mean and RMS for each x-bin of the 2D-hist
  TH1F *histosTracker
      [x.size()][y.size()]
      [NB_SUBLEVELS *
       NB_Z_SLICES];  // for the tracker plots all histos are copied to avoid using the same hists in different canvas

  TCanvas *c_hist[x.size()][y.size()][2 + NB_SUBLEVELS], *c_global_hist[2 + NB_SUBLEVELS];

  unsigned int nXBins;  // Sensible number of x-bins differs depending on the variable

  for (unsigned int ic = 0; ic <= NB_SUBLEVELS + 1; ic++) {
    c_global[ic] = new TCanvas(
        TString::Format(
            "global_%s_%d", ic == 0 ? "tracker" : (ic == 7 ? "pixel" : _sublevel_names[ic - 1].Data()), _canvas_index),
        TString::Format("Global overview of the %s variables",
                        ic == 0 ? "tracker" : (ic == 7 ? "pixel" : _sublevel_names[ic - 1].Data())),
        _window_width,
        _window_height);
    c_global[ic]->Divide(x.size(), y.size());

    if (_make_profile_plots) {
      c_global_hist[ic] =
          new TCanvas(TString::Format("global_profile_plots_%s_%d",
                                      ic == 0 ? "tracker" : (ic == 7 ? "pixel" : _sublevel_names[ic - 1].Data()),
                                      _canvas_index),
                      TString::Format("Global overview profile plots of the %s variables",
                                      ic == 0 ? "tracker" : (ic == 7 ? "pixel" : _sublevel_names[ic - 1].Data())),
                      _window_width,
                      _window_height);
      c_global_hist[ic]->Divide(x.size(), y.size());
    }
  }

  for (unsigned int ix = 0; ix < x.size(); ix++) {
    for (unsigned int iy = 0; iy < y.size(); iy++) {
      //if (x[ix] == y[iy]) continue;       // do not plot graphs like (r,r) or (phi,phi)
      for (unsigned int igraph = 0; igraph < NB_SUBLEVELS * NB_Z_SLICES * NB_MODULE_QUALITY; igraph++) {
        // declaring
        ipoint[ix][iy][igraph] =
            0;  // the purpose of an index for every graph is to avoid thousands of points at the origin of each
        graphs[ix][iy][igraph] = new TGraph();

        graphs[ix][iy][igraph]->SetMarkerColor(COLOR_CODE(igraph));
        graphs[ix][iy][igraph]->SetMarkerStyle(6);
        // pimping
        graphs[ix][iy][igraph]->SetName(
            x[ix] + y[iy] + _sublevel_names[igraph % NB_SUBLEVELS] +
            TString(igraph % (NB_SUBLEVELS * NB_Z_SLICES) >= NB_SUBLEVELS ? "n"
                                                                          : "p")  // graphs for negative/positive  z
            + TString(igraph >= NB_SUBLEVELS * NB_Z_SLICES ? (igraph >= 2 * NB_SUBLEVELS * NB_Z_SLICES ? "bad" : "list")
                                                           : "good"));  // graphs for good, bad modules and from a list
        graphs[ix][iy][igraph]->SetTitle(
            _sublevel_names[igraph % NB_SUBLEVELS] +
            TString(igraph % (NB_SUBLEVELS * NB_Z_SLICES) >= NB_SUBLEVELS ? " at z<0" : " at z>=0") +
            TString(igraph >= NB_SUBLEVELS * NB_Z_SLICES
                        ? (igraph >= 2 * NB_SUBLEVELS * NB_Z_SLICES ? " bad modules" : " in list")
                        : " good modules") +
            TString(";") + LateXstyle(x[ix]) + " /" + _units[x[ix]] + TString(";") + LateXstyle(y[iy]) + " /" +
            _units[y[iy]]);
        graphs[ix][iy][igraph]->SetMarkerStyle(
            igraph >= NB_SUBLEVELS * NB_Z_SLICES
                ? (igraph >= 2 * NB_SUBLEVELS * NB_Z_SLICES ? 4 : 5)
                : 6);  // empty circle for bad modules, X for those in list, dot for good ones
      }
    }
  }

  // Use seperate loop for the profile histograms since we do not produce histograms for the different module qualities
  if (_make_profile_plots) {
    for (unsigned int ix = 0; ix < x.size(); ix++) {
      if (x[ix] == "phi")
        nXBins = 10;
      else
        nXBins = 40;

      for (unsigned int iy = 0; iy < y.size(); iy++) {
        for (unsigned int igraph = 0; igraph < NB_SUBLEVELS * NB_Z_SLICES; igraph++) {
          // declaring
          histos2D[ix][iy][igraph] =
              new TH2F("2Dhist" + x[ix] + y[iy] + _sublevel_names[igraph % NB_SUBLEVELS] +
                           TString(igraph % (NB_SUBLEVELS * NB_Z_SLICES) >= NB_SUBLEVELS ? "n" : "p") +
                           std::to_string(_canvas_index),
                       "",
                       nXBins,
                       _min[x[ix]],
                       _max[x[ix]],
                       1000,
                       _min[y[iy]],
                       _max[y[iy]] + 1.);
        }
      }
    }
  }

#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: Creation of the TGraph[" << x.size() << "][" << y.size() << "]["
            << NB_SUBLEVELS * NB_Z_SLICES * NB_MODULE_QUALITY << "] ended." << std::endl;
#endif

  /// 2) loop on the TTree data
#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: Looping on the TTree" << std::endl;
#endif
#ifdef TALKATIVE
  unsigned int progress = 0;
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: 0%" << std::endl;
#endif
  for (unsigned int ientry = 0; ientry < nentries; ientry++) {
#ifdef TALKATIVE
    if (10 * ientry / nentries != progress) {
      progress = 10 * ientry / nentries;
      std::cout << __FILE__ << ":" << __LINE__ << ":Info: " << 10 * progress << "%" << std::endl;
    }
#endif
    // load current tree entry
    data->GetEntry(ientry);

    // CUTS on entry
    if (branch_i["level"] != _levelCut)
      continue;
    if (!_1dModule && branch_i["detDim"] == 1)
      continue;
    if (!_2dModule && branch_i["detDim"] == 2)
      continue;

    // loop on the different couples of variables to plot in a graph
    for (unsigned int ix = 0; ix < x.size(); ix++) {
      // CUTS on x[ix]
      if (_SF[x[ix]] * branch_f[x[ix]] > _max[x[ix]] || _SF[x[ix]] * branch_f[x[ix]] < _min[x[ix]]) {
        //#ifdef DEBUG
        //                std::cout << "branch_f[x[ix]]=" << branch_f[x[ix]] << std::endl;
        //#endif
        continue;
      }

      for (unsigned int iy = 0; iy < y.size(); iy++) {
        // CUTS on y[iy]
        //if (x[ix] == y[iy])                                                   continue; // TO DO: handle display when such a case occurs
        if (branch_i["sublevel"] < 1 || branch_i["sublevel"] > NB_SUBLEVELS)
          continue;

        // FILLING histograms take even those outside the plotted range into account
        if (_make_profile_plots) {
          if (_module_plot_option == "all") {
            const short int igraph = (branch_i["sublevel"] - 1) + (branch_f["z"] >= 0 ? 0 : NB_SUBLEVELS);
            histos2D[ix][iy][igraph]->Fill(_SF[x[ix]] * branch_f[x[ix]], _SF[y[iy]] * branch_f[y[iy]]);
          } else if (_module_plot_option == "good" && branch_i["badModuleQuality"] == 0) {
            const short int igraph = (branch_i["sublevel"] - 1) + (branch_f["z"] >= 0 ? 0 : NB_SUBLEVELS);
            histos2D[ix][iy][igraph]->Fill(_SF[x[ix]] * branch_f[x[ix]], _SF[y[iy]] * branch_f[y[iy]]);
          } else if (_module_plot_option == "list" &&
                     (branch_i["inModuleList"] == 1 || branch_i["badModuleQuality"] == 0)) {
            const short int igraph = (branch_i["sublevel"] - 1) + (branch_f["z"] >= 0 ? 0 : NB_SUBLEVELS);
            histos2D[ix][iy][igraph]->Fill(_SF[x[ix]] * branch_f[x[ix]], _SF[y[iy]] * branch_f[y[iy]]);
          }
        }

        // restrict scatter plots to chosen range
        if (_SF[y[iy]] * branch_f[y[iy]] > _max[y[iy]] || _SF[y[iy]] * branch_f[y[iy]] < _min[y[iy]]) {
          //#ifdef DEBUG
          //                    std::cout << "branch_f[y[iy]]=" << branch_f[y[iy]] << std::endl;
          //#endif
          continue;
        }

        // FILLING GRAPH
        if (y.size() >= x.size()) {
          if (branch_i["inModuleList"] == 0 && branch_i["badModuleQuality"] == 0) {
            const short int igraph = (branch_i["sublevel"] - 1) + (branch_f["z"] >= 0 ? 0 : NB_SUBLEVELS);
            graphs[ix][iy][igraph]->SetPoint(
                ipoint[ix][iy][igraph], _SF[x[ix]] * branch_f[x[ix]], _SF[y[iy]] * branch_f[y[iy]]);
            ipoint[ix][iy][igraph]++;
          }
          if (branch_i["inModuleList"] > 0) {
            const short int igraph =
                (branch_i["sublevel"] - 1) + (branch_f["z"] >= 0 ? 0 : NB_SUBLEVELS) + NB_SUBLEVELS * NB_Z_SLICES;
            graphs[ix][iy][igraph]->SetPoint(
                ipoint[ix][iy][igraph], _SF[x[ix]] * branch_f[x[ix]], _SF[y[iy]] * branch_f[y[iy]]);
            ipoint[ix][iy][igraph]++;
          }
          if (branch_i["badModuleQuality"] > 0) {
            const short int igraph =
                (branch_i["sublevel"] - 1) + (branch_f["z"] >= 0 ? 0 : NB_SUBLEVELS) + 2 * NB_SUBLEVELS * NB_Z_SLICES;
            graphs[ix][iy][igraph]->SetPoint(
                ipoint[ix][iy][igraph], _SF[x[ix]] * branch_f[x[ix]], _SF[y[iy]] * branch_f[y[iy]]);
            ipoint[ix][iy][igraph]++;
          }
        } else {
          if (branch_i["inModuleList"] == 0 && branch_i["badModuleQuality"] == 0) {
            const short int igraph = (branch_i["sublevel"] - 1) + (branch_f["z"] >= 0 ? 0 : NB_SUBLEVELS);
            graphs[iy][ix][igraph]->SetPoint(
                ipoint[iy][ix][igraph], _SF[x[ix]] * branch_f[x[ix]], _SF[y[iy]] * branch_f[y[iy]]);
            ipoint[iy][ix][igraph]++;
          }
          if (branch_i["inModuleList"] > 0) {
            const short int igraph =
                (branch_i["sublevel"] - 1) + (branch_f["z"] >= 0 ? 0 : NB_SUBLEVELS) + NB_SUBLEVELS * NB_Z_SLICES;
            graphs[iy][ix][igraph]->SetPoint(
                ipoint[iy][ix][igraph], _SF[x[ix]] * branch_f[x[ix]], _SF[y[iy]] * branch_f[y[iy]]);
            ipoint[iy][ix][igraph]++;
          }
          if (branch_i["badModuleQuality"] > 0) {
            const short int igraph =
                (branch_i["sublevel"] - 1) + (branch_f["z"] >= 0 ? 0 : NB_SUBLEVELS) + 2 * NB_SUBLEVELS * NB_Z_SLICES;
            graphs[iy][ix][igraph]->SetPoint(
                ipoint[ix][iy][igraph], _SF[x[ix]] * branch_f[x[ix]], _SF[y[iy]] * branch_f[y[iy]]);
            ipoint[iy][ix][igraph]++;
          }
        }
      }
    }
  }
#ifdef TALKATIVE
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: 100%\tLoop ended" << std::endl;
#endif

  /// 3) merge TGraph objects into TMultiGraph objects, then draw, print and write (according to the options _batchMode, _print and _write respectively)
  gROOT->SetBatch(_batchMode);  // if true, then equivalent to "root -b", i.e. no canvas
  if (_write) {                 // opening the file to write the graphs
    output = new TFile(_output_directory + TString(_output_filename),
                       "UPDATE");  // possibly existing file will be updated, otherwise created
    if (output->IsZombie()) {
      std::cout << __FILE__ << ":" << __LINE__ << ":Error: Opening of " << _output_directory + TString(_output_filename)
                << " failed" << std::endl;
      exit(-1);
    }
#ifdef TALKATIVE
    std::cout << __FILE__ << ":" << __LINE__ << ":Info: output file is "
              << _output_directory + TString(_output_filename) << std::endl;
#endif
  }
  // declaring TMultiGraphs and TCanvas
  // Usually more y variables than x variables
  // creating TLegend
  TLegend *legend = MakeLegend(.1, .92, .9, 1., NB_SUBLEVELS);
  if (_write)
    legend->Write();

  // check which modules are supposed to be plotted
  unsigned int n_module_types = 1;
  if (_module_plot_option == "all") {
    n_module_types = 3;  //plot all modules (good, list and bad )
  } else if (_module_plot_option == "list") {
    n_module_types = 2;  // plot good modules and those in the list
  } else if (_module_plot_option == "good") {
    n_module_types = 1;  // only plot the modules that are neither bad or in the list
  }

#define INDEX_IN_GLOBAL_CANVAS(i1, i2) 1 + i1 + i2 *x.size()
  // running on the TGraphs to produce the TMultiGraph and draw/print them
  for (unsigned int ix = 0; ix < x.size(); ix++) {
#ifdef DEBUG
    std::cout << __FILE__ << ":" << __LINE__ << ":Info: x[" << ix << "]=" << x[ix] << std::endl;
#endif

    // looping on Y axes
    for (unsigned int iy = 0; iy < y.size(); iy++) {
      TString min_branch = TString::Format("%s_min", y[iy].Data());
      TString max_branch = TString::Format("%s_max", y[iy].Data());

#ifdef DEBUG
      std::cout << __FILE__ << ":" << __LINE__ << ":Info: x[" << ix << "]=" << x[ix] << " and y[" << iy << "]=" << y[iy]
                << "\t-> creating TMultiGraph" << std::endl;
#endif
      mgraphs[ix][iy][0] = new TMultiGraph(
          TString::Format("mgr_%s_vs_%s_tracker_%d",
                          x[ix].Data(),
                          y[iy].Data(),
                          _canvas_index),  // name
          //LateXstyle(x[ix]) + TString(" vs. ") + LateXstyle(y[iy]) + TString(" for Tracker") // graph title
          TString(";") + LateXstyle(x[ix]) + " /" + _units[x[ix]]          // x axis title
              + TString(";") + LateXstyle(y[iy]) + " /" + _units[y[iy]]);  // y axis title

      mgraphs[ix][iy][7] = new TMultiGraph(
          TString::Format("mgr_%s_vs_%s_pixel_%d",
                          x[ix].Data(),
                          y[iy].Data(),
                          _canvas_index),  // name
          //LateXstyle(x[ix]) + TString(" vs. ") + LateXstyle(y[iy]) + TString(" for Tracker") // graph title
          TString(";") + LateXstyle(x[ix]) + " /" + _units[x[ix]]          // x axis title
              + TString(";") + LateXstyle(y[iy]) + " /" + _units[y[iy]]);  // y axis title

      /// TRACKER and PIXEL
      // fixing ranges and filling TMultiGraph
      // for (unsigned short int jgraph = NB_SUBLEVELS*NB_Z_SLICES-1 ; jgraph >= 0 ; --jgraph)
      for (unsigned short int jgraph = 0; jgraph < NB_SUBLEVELS * NB_Z_SLICES * n_module_types; jgraph++) {
        unsigned short int igraph =
            NB_SUBLEVELS * NB_Z_SLICES * n_module_types - jgraph -
            1;  // reverse counting for humane readability (one of the sublevel takes much more place than the others)

#ifdef DEBUG
        std::cout << __FILE__ << ":" << __LINE__ << ":Info: writing TGraph to file" << std::endl;
#endif
        // write into root file
        if (_write)
          graphs[ix][iy][igraph]->Write();
        if (graphs[ix][iy][igraph]->GetN() == 0) {
#ifdef TALKATIVE
          std::cout << __FILE__ << ":" << __LINE__ << ":Info: " << graphs[ix][iy][igraph]->GetName() << " is empty."
                    << std::endl;
#endif
          continue;
        }
#ifdef DEBUG
        std::cout << __FILE__ << ":" << __LINE__ << ":Info: cloning, coloring and adding TGraph "
                  << _sublevel_names[igraph % NB_SUBLEVELS] << (igraph >= NB_SUBLEVELS ? "(z<0)" : "(z>0)")
                  << " to global TMultiGraph" << std::endl;
#endif
        // clone to prevent any injure on the graph
        TGraph *gr = (TGraph *)graphs[ix][iy][igraph]->Clone();
        // color
        gr->SetMarkerColor(COLOR_CODE(igraph % NB_SUBLEVELS));
        mgraphs[ix][iy][0]->Add(gr, "P");  //, (mgraphs[ix][iy][0]->GetListOfGraphs()==0?"AP":"P"));

        if (igraph % NB_SUBLEVELS == 0 || igraph % NB_SUBLEVELS == 1)
          mgraphs[ix][iy][7]->Add(gr, "P");  // Add BPIX (0) and FPIX (1) to pixel plot
      }

      /// SUBLEVELS (1..6)
      for (unsigned int isublevel = 1; isublevel <= NB_SUBLEVELS; isublevel++) {
#ifdef DEBUG
        std::cout << __FILE__ << ":" << __LINE__ << ":Info: cloning, coloring and adding TGraph "
                  << _sublevel_names[isublevel - 1] << " to sublevel TMultiGraph" << std::endl;
#endif
        mgraphs[ix][iy][isublevel] =
            new TMultiGraph(TString::Format("%s_vs_%s_%s_%d",
                                            x[ix].Data(),
                                            y[iy].Data(),
                                            _sublevel_names[isublevel - 1].Data(),
                                            _canvas_index),  // name
                            //LateXstyle(x[ix]) + TString(" vs. ") + LateXstyle(y[iy]) + TString(" for ") +
                            _sublevel_names[isublevel - 1]                                   // graph title
                                + TString(";") + LateXstyle(x[ix]) + " /" + _units[x[ix]]    // x axis title
                                + TString(";") + LateXstyle(y[iy]) + " /" + _units[y[iy]]);  // y axis title

        graphs[ix][iy][isublevel - 1]->SetMarkerColor(kBlack);
        graphs[ix][iy][NB_SUBLEVELS + isublevel - 1]->SetMarkerColor(kRed);
        graphs[ix][iy][2 * NB_SUBLEVELS + isublevel - 1]->SetMarkerColor(kGray + 1);
        graphs[ix][iy][3 * NB_SUBLEVELS + isublevel - 1]->SetMarkerColor(kRed - 7);
        graphs[ix][iy][4 * NB_SUBLEVELS + isublevel - 1]->SetMarkerColor(kGray + 1);
        graphs[ix][iy][5 * NB_SUBLEVELS + isublevel - 1]->SetMarkerColor(kRed - 7);
        if (graphs[ix][iy][isublevel - 1]->GetN() > 0)
          mgraphs[ix][iy][isublevel]->Add(
              graphs[ix][iy][isublevel - 1],
              "P");  //(mgraphs[ix][iy][isublevel-1]->GetListOfGraphs()==0?"AP":"P")); // z>0
#ifdef TALKATIVE
        else
          std::cout << __FILE__ << ":" << __LINE__
                    << ":Info: graphs[ix][iy][isublevel-1]=" << graphs[ix][iy][isublevel - 1]->GetName()
                    << " is empty -> not added into " << mgraphs[ix][iy][isublevel]->GetName() << std::endl;
#endif
        if (graphs[ix][iy][NB_SUBLEVELS + isublevel - 1]->GetN() > 0)
          mgraphs[ix][iy][isublevel]->Add(
              graphs[ix][iy][NB_SUBLEVELS + isublevel - 1],
              "P");  //(mgraphs[ix][iy][isublevel-1]->GetListOfGraphs()==0?"AP":"P")); // z<0
#ifdef TALKATIVE
        else
          std::cout << __FILE__ << ":" << __LINE__ << ":Info: graphs[ix][iy][NB_SUBLEVEL+isublevel-1]="
                    << graphs[ix][iy][NB_Z_SLICES + isublevel - 1]->GetName() << " is empty -> not added into "
                    << mgraphs[ix][iy][isublevel]->GetName() << std::endl;
#endif
#if NB_Z_SLICES != 2
        std::cout << __FILE__ << ":" << __LINE__ << ":Error: color code incomplete for Z slices..." << std::endl;
#endif
        if (_module_plot_option == "all") {
          if (graphs[ix][iy][2 * NB_SUBLEVELS + isublevel - 1]->GetN() > 0)
            mgraphs[ix][iy][isublevel]->Add(graphs[ix][iy][2 * NB_SUBLEVELS + isublevel - 1], "P");
          if (graphs[ix][iy][3 * NB_SUBLEVELS + isublevel - 1]->GetN() > 0)
            mgraphs[ix][iy][isublevel]->Add(graphs[ix][iy][3 * NB_SUBLEVELS + isublevel - 1], "P");
          if (graphs[ix][iy][4 * NB_SUBLEVELS + isublevel - 1]->GetN() > 0)
            mgraphs[ix][iy][isublevel]->Add(graphs[ix][iy][4 * NB_SUBLEVELS + isublevel - 1], "P");
          if (graphs[ix][iy][5 * NB_SUBLEVELS + isublevel - 1]->GetN() > 0)
            mgraphs[ix][iy][isublevel]->Add(graphs[ix][iy][5 * NB_SUBLEVELS + isublevel - 1], "P");
        }
        if (_module_plot_option == "list") {
          if (graphs[ix][iy][2 * NB_SUBLEVELS + isublevel - 1]->GetN() > 0)
            mgraphs[ix][iy][isublevel]->Add(graphs[ix][iy][2 * NB_SUBLEVELS + isublevel - 1], "P");
          if (graphs[ix][iy][3 * NB_SUBLEVELS + isublevel - 1]->GetN() > 0)
            mgraphs[ix][iy][isublevel]->Add(graphs[ix][iy][3 * NB_SUBLEVELS + isublevel - 1], "P");
        }
      }

      // fixing ranges, saving, and drawing of TMultiGraph (tracker AND sublevels AND pixel, i.e. 2+NB_SUBLEVELS objects)
      // the individual canvases are saved, but the global are just drawn and will be saved later
      for (unsigned short int imgr = 0; imgr <= NB_SUBLEVELS + 1; imgr++) {
#ifdef DEBUG
        std::cout << __FILE__ << ":" << __LINE__ << ":Info: treating individual canvases." << std::endl;
#endif
        // drawing into individual canvas and printing it (including a legend for the tracker canvas)
        c[ix][iy][imgr] = new TCanvas(
            TString::Format("c_%s_vs_%s_%s_%d",
                            x[ix].Data(),
                            y[iy].Data(),
                            imgr == 0 ? "tracker" : (imgr == 7 ? "pixel" : _sublevel_names[imgr - 1].Data()),
                            _canvas_index),
            TString::Format("%s vs. %s at %s level",
                            x[ix].Data(),
                            y[iy].Data(),
                            imgr == 0 ? "tracker" : (imgr == 7 ? "pixel" : _sublevel_names[imgr - 1].Data())),
            _window_width,
            _window_height);
        c[ix][iy][imgr]->SetGrid(_grid_x, _grid_y);  // grid

        if (mgraphs[ix][iy][imgr]->GetListOfGraphs() != nullptr) {
          if (CFG.count(min_branch.Data()) > 0) {
            mgraphs[ix][iy][imgr]->SetMinimum(CFG.get<float>(min_branch.Data()));
          }
          if (CFG.count(max_branch.Data()) > 0) {
            mgraphs[ix][iy][imgr]->SetMaximum(CFG.get<float>(max_branch.Data()));
          }
          mgraphs[ix][iy][imgr]->Draw("A");
        }
        if (imgr == 0 && _legend)
          legend->Draw();  // only for the tracker
        if (_print && !_print_only_global)
          c[ix][iy][imgr]->Print(
              _output_directory + mgraphs[ix][iy][imgr]->GetName() + ExtensionFromPrintOption(_print_option),
              _print_option);

        // writing into root file
        if (_write)
          mgraphs[ix][iy][imgr]->Write();

        // drawing into global canvas
        c_global[imgr]->cd(INDEX_IN_GLOBAL_CANVAS(ix, iy));
        c_global[imgr]->GetPad(INDEX_IN_GLOBAL_CANVAS(ix, iy))->SetFillStyle(4000);         //  make the pad transparent
        c_global[imgr]->GetPad(INDEX_IN_GLOBAL_CANVAS(ix, iy))->SetGrid(_grid_x, _grid_y);  // grid
        if (mgraphs[ix][iy][imgr]->GetListOfGraphs() != nullptr) {
          if (CFG.count(min_branch.Data()) > 0) {
            mgraphs[ix][iy][imgr]->SetMinimum(CFG.get<float>(min_branch.Data()));
          }
          if (CFG.count(max_branch.Data()) > 0) {
            mgraphs[ix][iy][imgr]->SetMaximum(CFG.get<float>(max_branch.Data()));
          }
          mgraphs[ix][iy][imgr]->Draw("A");
        }
        // printing will be performed after customisation (e.g. legend or title) just after the loops on ix and iy
      }
    }  // end of loop on y
  }    // end of loop on x

  // CUSTOMISATION
  gStyle->SetOptTitle(0);  // otherwise, the title is repeated in every pad of the global canvases
                           // -> instead, we will write it in the upper part in a TPaveText or in a TLegend
  for (unsigned int ic = 0; ic <= NB_SUBLEVELS + 1; ic++) {
    c_global[ic]->Draw();

    // setting legend to tracker canvases
    if (!_legend)
      break;
    TCanvas *c_temp = (TCanvas *)c_global[ic]->Clone(c_global[ic]->GetTitle() + TString("_sub"));
    c_temp->Draw();
    c_global[ic] = new TCanvas(
        c_temp->GetName() + TString("_final"), c_temp->GetTitle(), c_temp->GetWindowWidth(), c_temp->GetWindowHeight());
    c_global[ic]->Draw();
    TPad *p_up = new TPad(TString("legend_") + c_temp->GetName(),
                          "",
                          0.,
                          0.9,
                          1.,
                          1.,  // relative position
                          -1,
                          0,
                          0),  // display options
        *p_down = new TPad(TString("main_") + c_temp->GetName(), "", 0., 0., 1., 0.9, -1, 0, 0);
    // in the lower part, draw the plots
    p_down->Draw();
    p_down->cd();
    c_temp->DrawClonePad();
    c_global[ic]->cd();
    // in the upper part, pimp the canvas :p
    p_up->Draw();
    p_up->cd();
    if (ic == 0)  // tracker
    {
      TLegend *global_legend = MakeLegend(.05, .1, .7, .8, NB_SUBLEVELS);  //, "brNDC");
      global_legend->Draw();
      TPaveText *pt_geom = new TPaveText(.75, .1, .95, .8, "NB");
      pt_geom->SetFillColor(0);
      pt_geom->SetTextSize(0.25);
      pt_geom->AddText(TString("x: ") + _reference_name);
      pt_geom->AddText(TString("y: ") + _alignment_name + TString(" - ") + _reference_name);
      pt_geom->Draw();
    } else if (ic == 7)  // pixel
    {
      TLegend *global_legend = MakeLegend(.05, .1, .7, .8, 2);  //, "brNDC");
      global_legend->Draw();
      TPaveText *pt_geom = new TPaveText(.75, .1, .95, .8, "NB");
      pt_geom->SetFillColor(0);
      pt_geom->SetTextSize(0.25);
      pt_geom->AddText(TString("x: ") + _reference_name);
      pt_geom->AddText(TString("y: ") + _alignment_name + TString(" - ") + _reference_name);
      pt_geom->Draw();
    } else  // sublevels
    {
      TPaveText *pt = new TPaveText(.05, .1, .7, .8, "NB");
      pt->SetFillColor(0);
      pt->AddText(_sublevel_names[ic - 1]);
      pt->Draw();
      TPaveText *pt_geom = new TPaveText(.6, .1, .95, .8, "NB");
      pt_geom->SetFillColor(0);
      pt_geom->SetTextSize(0.3);
      pt_geom->AddText(TString("x: ") + _reference_name);
      pt_geom->AddText(TString("y: ") + _alignment_name + TString(" - ") + _reference_name);
      pt_geom->Draw();
    }
    // printing
    if (_print)
      c_global[ic]->Print(_output_directory + c_global[ic]->GetName() + ExtensionFromPrintOption(_print_option),
                          _print_option);
    if (_write)
      c_global[ic]->Write();
  }

  // printing global canvases
  if (_write)
    output->Close();

  // Now produce the profile plots if the option is chosen
  // Use seperate loops since no seperate plots are produced for different module qualities
  if (_make_profile_plots) {
    // Fill Content of 2D-hists into 1D-hists for the profile plots
    // Loop over all y-bins for a certain x-bin, calculate mean and RMS as entries of the 1D-hists
    bool entries = false;
    for (unsigned int ix = 0; ix < x.size(); ix++) {
      for (unsigned int iy = 0; iy < y.size(); iy++) {
        TString min_branch = TString::Format("%s_min", y[iy].Data());
        TString max_branch = TString::Format("%s_max", y[iy].Data());
        for (unsigned int igraph = 0; igraph < NB_SUBLEVELS * NB_Z_SLICES; igraph++) {
          // Declare hists which will be plotted for the profile plots
          histos[ix][iy][igraph] =
              new TH1F("1Dhist" + x[ix] + y[iy] + _sublevel_names[igraph % NB_SUBLEVELS] +
                           TString(igraph % (NB_SUBLEVELS * NB_Z_SLICES) >= NB_SUBLEVELS ? "n" : "p") +
                           std::to_string(_canvas_index),
                       "",
                       histos2D[ix][iy][igraph]->GetXaxis()->GetNbins(),
                       _min[x[ix]],
                       _max[x[ix]]);
          histos[ix][iy][igraph]->SetMarkerColor(COLOR_CODE(igraph));
          histos[ix][iy][igraph]->SetLineColor(COLOR_CODE(igraph));
          histos[ix][iy][igraph]->StatOverflows(kTRUE);

          // Loop over x bins
          for (int binx = 0; binx <= histos2D[ix][iy][igraph]->GetXaxis()->GetNbins(); binx++) {
            entries = false;
            // Declare y-histogram for each x bin
            histosYValues[ix][iy][igraph] =
                new TH1F("1Dhist_Y-Values" + x[ix] + y[iy] + _sublevel_names[igraph % NB_SUBLEVELS] +
                             TString(igraph % (NB_SUBLEVELS * NB_Z_SLICES) >= NB_SUBLEVELS ? "n" : "p") +
                             std::to_string(_canvas_index) + std::to_string(binx),
                         "",
                         histos2D[ix][iy][igraph]->GetYaxis()->GetNbins(),
                         _min[y[iy]],
                         _max[y[iy]] + 1.);
            histosYValues[ix][iy][igraph]->StatOverflows(kTRUE);
            // Loop over y-bins for each x-bin of the 2D histogram and put it into the 1-d y histograms
            // Take overflow bin into account
            for (int biny = 0; biny <= histos2D[ix][iy][igraph]->GetYaxis()->GetNbins() + 1; biny++) {
              if (histos2D[ix][iy][igraph]->GetBinContent(binx, biny) > 1.) {
                histosYValues[ix][iy][igraph]->SetBinContent(biny, histos2D[ix][iy][igraph]->GetBinContent(binx, biny));
                entries = true;
              }
            }
            if (entries) {
              histos[ix][iy][igraph]->SetBinContent(binx, histosYValues[ix][iy][igraph]->GetMean());
              histos[ix][iy][igraph]->SetBinError(binx, histosYValues[ix][iy][igraph]->GetRMS());
            } else
              histos[ix][iy][igraph]->SetBinContent(binx, -999999.);
          }
        }

        // Customize and print the histograms

        /// TRACKER
        // fixing ranges and draw profile plot histos

        c_hist[ix][iy][0] =
            new TCanvas(TString::Format("c_hist_%s_vs_%s_tracker_%d", x[ix].Data(), y[iy].Data(), _canvas_index),
                        TString::Format("Profile plot %s vs. %s at tracker level", x[ix].Data(), y[iy].Data()),
                        _window_width,
                        _window_height);
        c_hist[ix][iy][0]->SetGrid(_grid_x, _grid_y);  // grid
        // Draw the frame that will contain the histograms
        // One needs to specify the binning and title
        c_hist[ix][iy][0]->GetPad(0)->DrawFrame(
            _min[x[ix]],
            CFG.count(min_branch.Data()) > 0 ? CFG.get<float>(min_branch.Data()) : _min[y[iy]],
            _max[x[ix]],
            CFG.count(max_branch.Data()) > 0 ? CFG.get<float>(max_branch.Data()) : _max[y[iy]],
            TString(";") + LateXstyle(x[ix]) + " /" + _units[x[ix]] + TString(";") + LateXstyle(y[iy]) + " /" +
                _units[y[iy]]);
        if (_legend)
          legend->Draw("same");

        for (unsigned short int jgraph = 0; jgraph < NB_SUBLEVELS * NB_Z_SLICES; jgraph++) {
          unsigned short int igraph =
              NB_SUBLEVELS * NB_Z_SLICES - jgraph -
              1;  // reverse counting for humane readability (one of the sublevel takes much more place than the others)

          // clone to prevent any injure on the graph
          histosTracker[ix][iy][igraph] = (TH1F *)histos[ix][iy][igraph]->Clone();
          // color
          histosTracker[ix][iy][igraph]->SetMarkerColor(COLOR_CODE(igraph % NB_SUBLEVELS));
          histosTracker[ix][iy][igraph]->SetLineColor(COLOR_CODE(igraph % NB_SUBLEVELS));
          histosTracker[ix][iy][igraph]->SetMarkerStyle(6);
          histosTracker[ix][iy][igraph]->Draw("same pe0");
        }

        if (_print && !_print_only_global)
          c_hist[ix][iy][0]->Print(
              _output_directory +
                  TString::Format("Profile_plot_%s_vs_%s_tracker_%d", x[ix].Data(), y[iy].Data(), _canvas_index) +
                  ExtensionFromPrintOption(_print_option),
              _print_option);

        //Draw into profile hists global tracker canvas
        c_global_hist[0]->cd(INDEX_IN_GLOBAL_CANVAS(ix, iy));
        c_global_hist[0]->GetPad(INDEX_IN_GLOBAL_CANVAS(ix, iy))->SetFillStyle(4000);  //  make the pad transparent
        c_global_hist[0]->GetPad(INDEX_IN_GLOBAL_CANVAS(ix, iy))->SetGrid(_grid_x, _grid_y);  // grid
        c_global_hist[0]
            ->GetPad(INDEX_IN_GLOBAL_CANVAS(ix, iy))
            ->DrawFrame(_min[x[ix]],
                        CFG.count(min_branch.Data()) > 0 ? CFG.get<float>(min_branch.Data()) : _min[y[iy]],
                        _max[x[ix]],
                        CFG.count(max_branch.Data()) > 0 ? CFG.get<float>(max_branch.Data()) : _max[y[iy]],
                        TString(";") + LateXstyle(x[ix]) + " /" + _units[x[ix]] + TString(";") + LateXstyle(y[iy]) +
                            " /" + _units[y[iy]]);

        for (unsigned short int jgraph = 0; jgraph < NB_SUBLEVELS * NB_Z_SLICES; jgraph++) {
          unsigned short int igraph =
              NB_SUBLEVELS * NB_Z_SLICES - jgraph -
              1;  // reverse counting for humane readability (one of the sublevel takes much more place than the others)
          histosTracker[ix][iy][igraph]->Draw("same pe0");
        }

        /// PIXEL
        // fixing ranges and draw profile plot histos

        c_hist[ix][iy][7] =
            new TCanvas(TString::Format("c_hist_%s_vs_%s_pixel_%d", x[ix].Data(), y[iy].Data(), _canvas_index),
                        TString::Format("Profile plot %s vs. %s at pixel level", x[ix].Data(), y[iy].Data()),
                        _window_width,
                        _window_height);
        c_hist[ix][iy][7]->SetGrid(_grid_x, _grid_y);  // grid
        // Draw the frame that will contain the histograms
        // One needs to specify the binning and title
        c_hist[ix][iy][7]->GetPad(0)->DrawFrame(
            _min[x[ix]],
            CFG.count(min_branch.Data()) > 0 ? CFG.get<float>(min_branch.Data()) : _min[y[iy]],
            _max[x[ix]],
            CFG.count(max_branch.Data()) > 0 ? CFG.get<float>(max_branch.Data()) : _max[y[iy]],
            TString(";") + LateXstyle(x[ix]) + " /" + _units[x[ix]] + TString(";") + LateXstyle(y[iy]) + " /" +
                _units[y[iy]]);
        if (_legend)
          legend->Draw("same");

        for (unsigned short int jgraph = 0; jgraph < NB_SUBLEVELS * NB_Z_SLICES; jgraph++) {
          unsigned short int igraph =
              NB_SUBLEVELS * NB_Z_SLICES - jgraph -
              1;  // reverse counting for humane readability (one of the sublevel takes much more place than the others)

          if (igraph % NB_SUBLEVELS == 0 || igraph % NB_SUBLEVELS == 1)  //Only BPIX and FPIX
          {
            // clone to prevent any injure on the graph
            histosTracker[ix][iy][igraph] = (TH1F *)histos[ix][iy][igraph]->Clone();
            // color
            histosTracker[ix][iy][igraph]->SetMarkerColor(COLOR_CODE(igraph % NB_SUBLEVELS));
            histosTracker[ix][iy][igraph]->SetLineColor(COLOR_CODE(igraph % NB_SUBLEVELS));
            histosTracker[ix][iy][igraph]->SetMarkerStyle(6);
            histosTracker[ix][iy][igraph]->Draw("same pe0");
          }
        }

        if (_print && !_print_only_global)
          c_hist[ix][iy][7]->Print(
              _output_directory +
                  TString::Format("Profile_plot_%s_vs_%s_pixel_%d", x[ix].Data(), y[iy].Data(), _canvas_index) +
                  ExtensionFromPrintOption(_print_option),
              _print_option);

        //Draw into profile hists global tracker canvas
        c_global_hist[7]->cd(INDEX_IN_GLOBAL_CANVAS(ix, iy));
        c_global_hist[7]->GetPad(INDEX_IN_GLOBAL_CANVAS(ix, iy))->SetFillStyle(4000);  //  make the pad transparent
        c_global_hist[7]->GetPad(INDEX_IN_GLOBAL_CANVAS(ix, iy))->SetGrid(_grid_x, _grid_y);  // grid
        c_global_hist[7]
            ->GetPad(INDEX_IN_GLOBAL_CANVAS(ix, iy))
            ->DrawFrame(_min[x[ix]],
                        CFG.count(min_branch.Data()) > 0 ? CFG.get<float>(min_branch.Data()) : _min[y[iy]],
                        _max[x[ix]],
                        CFG.count(max_branch.Data()) > 0 ? CFG.get<float>(max_branch.Data()) : _max[y[iy]],
                        TString(";") + LateXstyle(x[ix]) + " /" + _units[x[ix]] + TString(";") + LateXstyle(y[iy]) +
                            " /" + _units[y[iy]]);

        for (unsigned short int jgraph = 0; jgraph < NB_SUBLEVELS * NB_Z_SLICES; jgraph++) {
          unsigned short int igraph =
              NB_SUBLEVELS * NB_Z_SLICES - jgraph -
              1;  // reverse counting for humane readability (one of the sublevel takes much more place than the others)
          histosTracker[ix][iy][igraph]->Draw("same pe0");
        }
        // printing will be performed after customisation (e.g. legend or title) just after the loops on ix and iy
        /// SUBLEVELS (1..6)
        for (unsigned int isublevel = 1; isublevel <= NB_SUBLEVELS; isublevel++) {
          // Draw and print profile histograms
          c_hist[ix][iy][isublevel] =
              new TCanvas(TString::Format("c_hist_%s_vs_%s_%s_%d",
                                          x[ix].Data(),
                                          y[iy].Data(),
                                          isublevel == 0 ? "tracker" : _sublevel_names[isublevel - 1].Data(),
                                          _canvas_index),
                          TString::Format("Profile plot %s vs. %s at %s level",
                                          x[ix].Data(),
                                          y[iy].Data(),
                                          isublevel == 0 ? "tracker" : _sublevel_names[isublevel - 1].Data()),
                          _window_width,
                          _window_height);
          c_hist[ix][iy][isublevel]->SetGrid(_grid_x, _grid_y);  // grid
          c_hist[ix][iy][isublevel]->GetPad(0)->DrawFrame(
              _min[x[ix]],
              CFG.count(min_branch.Data()) > 0 ? CFG.get<float>(min_branch.Data()) : _min[y[iy]],
              _max[x[ix]],
              CFG.count(max_branch.Data()) > 0 ? CFG.get<float>(max_branch.Data()) : _max[y[iy]],
              TString(";") + LateXstyle(x[ix]) + " /" + _units[x[ix]] + TString(";") + LateXstyle(y[iy]) + " /" +
                  _units[y[iy]]);

          histos[ix][iy][isublevel - 1]->SetMarkerColor(kBlack);
          histos[ix][iy][isublevel - 1]->SetLineColor(kBlack);
          histos[ix][iy][NB_SUBLEVELS + isublevel - 1]->SetMarkerColor(kRed);
          histos[ix][iy][NB_SUBLEVELS + isublevel - 1]->SetLineColor(kRed);

          histos[ix][iy][isublevel - 1]->Draw("same pe0");
          histos[ix][iy][NB_SUBLEVELS + isublevel - 1]->Draw("same pe0");

          if (_print && !_print_only_global)
            c_hist[ix][iy][isublevel]->Print(_output_directory +
                                                 TString::Format("Profile_plot_%s_vs_%s_%s_%d",
                                                                 x[ix].Data(),
                                                                 y[iy].Data(),
                                                                 _sublevel_names[isublevel - 1].Data(),
                                                                 _canvas_index) +
                                                 ExtensionFromPrintOption(_print_option),
                                             _print_option);

          // draw into global canvas
          // printing will be performed after customisation (e.g. legend or title) just after the loops on ix and iy
          c_global_hist[isublevel]->cd(INDEX_IN_GLOBAL_CANVAS(ix, iy));
          c_global_hist[isublevel]
              ->GetPad(INDEX_IN_GLOBAL_CANVAS(ix, iy))
              ->SetFillStyle(4000);  //  make the pad transparent
          c_global_hist[isublevel]->GetPad(INDEX_IN_GLOBAL_CANVAS(ix, iy))->SetGrid(_grid_x, _grid_y);  // grid
          c_global_hist[isublevel]
              ->GetPad(INDEX_IN_GLOBAL_CANVAS(ix, iy))
              ->DrawFrame(_min[x[ix]],
                          CFG.count(min_branch.Data()) > 0 ? CFG.get<float>(min_branch.Data()) : _min[y[iy]],
                          _max[x[ix]],
                          CFG.count(max_branch.Data()) > 0 ? CFG.get<float>(max_branch.Data()) : _max[y[iy]],
                          TString(";") + LateXstyle(x[ix]) + " /" + _units[x[ix]] + TString(";") + LateXstyle(y[iy]) +
                              " /" + _units[y[iy]]);

          histos[ix][iy][isublevel - 1]->Draw("same pe0");
          histos[ix][iy][NB_SUBLEVELS + isublevel - 1]->Draw("same pe0");
        }

      }  // end of loop on y
    }    // end of loop on x

    // CUSTOMISATION
    gStyle->SetOptTitle(0);  // otherwise, the title is repeated in every pad of the global canvases
                             // -> instead, we will write it in the upper part in a TPaveText or in a TLegend
    for (unsigned int ic = 0; ic <= NB_SUBLEVELS; ic++) {
      // setting legend to tracker canvases
      if (!_legend)
        break;

      // setting legend to tracker canvases
      if (!_legend)
        break;
      TCanvas *c_temp_hist = (TCanvas *)c_global_hist[ic]->Clone(c_global_hist[ic]->GetTitle() + TString("_sub"));
      c_temp_hist->Draw();
      c_global_hist[ic] = new TCanvas(c_temp_hist->GetName() + TString("_final"),
                                      c_temp_hist->GetTitle(),
                                      c_temp_hist->GetWindowWidth(),
                                      c_temp_hist->GetWindowHeight());
      c_global_hist[ic]->Draw();
      TPad *p_up = new TPad(TString("legend_") + c_temp_hist->GetName(),
                            "",
                            0.,
                            0.9,
                            1.,
                            1.,  // relative position
                            -1,
                            0,
                            0),  // display options
          *p_down = new TPad(TString("main_") + c_temp_hist->GetName(), "", 0., 0., 1., 0.9, -1, 0, 0);
      // in the lower part, draw the plots
      p_down->Draw();
      p_down->cd();
      c_temp_hist->DrawClonePad();
      c_global_hist[ic]->cd();
      // in the upper part, pimp the canvas :p
      p_up->Draw();
      p_up->cd();
      if (ic == 0)  // tracker
      {
        TLegend *global_legend = MakeLegend(.05, .1, .7, .8, NB_SUBLEVELS);  //, "brNDC");
        global_legend->Draw();
        TPaveText *pt_geom = new TPaveText(.75, .1, .95, .8, "NB");
        pt_geom->SetFillColor(0);
        pt_geom->SetTextSize(0.25);
        pt_geom->AddText(TString("x: ") + _reference_name);
        pt_geom->AddText(TString("y: ") + _alignment_name + TString(" - ") + _reference_name);
        pt_geom->Draw();
      } else if (ic == 7)  // pixel
      {
        TLegend *global_legend = MakeLegend(.05, .1, .7, .8, 2);  //, "brNDC");
        global_legend->Draw();
        TPaveText *pt_geom = new TPaveText(.75, .1, .95, .8, "NB");
        pt_geom->SetFillColor(0);
        pt_geom->SetTextSize(0.25);
        pt_geom->AddText(TString("x: ") + _reference_name);
        pt_geom->AddText(TString("y: ") + _alignment_name + TString(" - ") + _reference_name);
        pt_geom->Draw();
      } else  // sublevels
      {
        TPaveText *pt = new TPaveText(.05, .1, .7, .8, "NB");
        pt->SetFillColor(0);
        pt->AddText(_sublevel_names[ic - 1]);
        pt->Draw();
        TPaveText *pt_geom = new TPaveText(.6, .1, .95, .8, "NB");
        pt_geom->SetFillColor(0);
        pt_geom->SetTextSize(0.3);
        pt_geom->AddText(TString("x: ") + _reference_name);
        pt_geom->AddText(TString("y: ") + _alignment_name + TString(" - ") + _reference_name);
        pt_geom->Draw();
      }
      // printing
      if (_print)
        c_global_hist[ic]->Print(
            _output_directory + c_global_hist[ic]->GetName() + ExtensionFromPrintOption(_print_option), _print_option);
    }
  }

#ifdef TALKATIVE
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: End of MakePlots method" << std::endl;
#endif
}

// Make additional table for the mean/RMS values of differences
void GeometryComparisonPlotter::MakeTables(
    std::vector<TString> x,  // axes to combine to plot
    std::vector<TString> y,  // only requires the differences (y values in the plots) and ranges
    pt::ptree CFG)           // property tree storing the min and max values under <var>_min and <var>_max
{
  /// -1) check that only existing branches are called
  // (we use a macro to avoid copy/paste)
#define CHECK_BRANCHES(branchname_vector)                                                       \
  for (unsigned int i = 0; i < branchname_vector.size(); i++) {                                 \
    if (branch_f.find(branchname_vector[i]) == branch_f.end()) {                                \
      std::cout << __FILE__ << ":" << __LINE__ << ":Error: The branch " << branchname_vector[i] \
                << " is not recognised." << std::endl;                                          \
      return;                                                                                   \
    }                                                                                           \
  }
  CHECK_BRANCHES(x);
  CHECK_BRANCHES(y);

  const unsigned int nentries = data->GetEntries();

#ifdef TALKATIVE
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: ";
  INSIDE_VECTOR(x);
  std::cout << std::endl;
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: ";
  INSIDE_VECTOR(y);
  std::cout << std::endl;
#endif

  /// 0) min and max values
  // the max and min of the graphs are computed from the tree if they have not been manually input yet
  // (we use a macro to avoid copy/paste)
#define LIMITS(axes_vector)                                                          \
  for (unsigned int i = 0; i < axes_vector.size(); i++) {                            \
    if (_SF.find(axes_vector[i]) == _SF.end())                                       \
      _SF[axes_vector[i]] = 1.;                                                      \
    if (_min.find(axes_vector[i]) == _min.end())                                     \
      _min[axes_vector[i]] = _SF[axes_vector[i]] * data->GetMinimum(axes_vector[i]); \
    if (_max.find(axes_vector[i]) == _max.end())                                     \
      _max[axes_vector[i]] = _SF[axes_vector[i]] * data->GetMaximum(axes_vector[i]); \
  }
  LIMITS(x);
  LIMITS(y);

#ifdef TALKATIVE
  CHECK_MAP_CONTENT(_min, float);
  CHECK_MAP_CONTENT(_max, float);
  CHECK_MAP_CONTENT(_SF, float);
#endif

  /// 1) declare histograms
  // the idea is to produce tables of the differences and the absolute positions containing mean and RMS values
  // for the different subdetectors - 0..5=different sublevels.
  // Values for each endcap detector are to be split in +/-z, for the barrel detectors in +/- x (half barrels)
  // Since it is easier to handle in the loops, all subdetectors will be split in
  // 4 parts at first: (+/-x)X(+/-z)
  // This means that 2*2*6 histograms will be filled during the loop on the TTree
  // Pairs of histograms need to be combined afterwards again
  // Histograms 0-5 are at +x and +z, 6-11 at +x and -z, 12-17 at -x and +z, and 18-23 at -x and -z
  //
  // Two version of the table containing the differences are produced. Once using Gaussian fits (more stable
  // vs single outliers but perform poorly if the distributions are non-Gaussian) and once using
  // the mean and RMS of the histograms (more stable but outliers have a strong impact on the RMS).
  // For the absolute positions, only mean+RMS are used since the detector layout is not Gaussian
  // (structures due to layers/rings etc)
#ifndef NB_SUBLEVELS
#define NB_SUBLEVELS 6
#endif
#define NB_Z_SLICES 2
#define NB_X_SLICES 2

  TH1F *histosx[x.size()][NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES];
  float meanValuex[x.size()][NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES];
  float RMSx[x.size()][NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES];

  TH1F *histos[y.size()][NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES];
  TF1 *gausFit[y.size()][NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES];
  float meanValue[y.size()][NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES];
  float meanValueGaussian[y.size()][NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES];
  float RMS[y.size()][NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES];
  float RMSGaussian[y.size()][NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES];

  for (unsigned int iy = 0; iy < y.size(); iy++) {
    for (unsigned int ihist = 0; ihist < NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES; ihist++) {
      // Create and correctly name a histogram for each subdetector*Z_Slice*X_Slice
      histos[iy][ihist] = new TH1F("hist" + y[iy] + _sublevel_names[ihist % NB_SUBLEVELS] +
                                       TString(ihist % (NB_SUBLEVELS * NB_Z_SLICES) >= NB_SUBLEVELS ? "zn" : "zp") +
                                       TString(ihist >= NB_SUBLEVELS * NB_Z_SLICES ? "xn" : "xp"),
                                   "",
                                   1000,
                                   _min[y[iy]],
                                   _max[y[iy]] + 1.);
      histos[iy][ihist]->StatOverflows(kTRUE);
    }
  }

  for (unsigned int ix = 0; ix < x.size(); ix++) {
    for (unsigned int ihist = 0; ihist < NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES; ihist++) {
      // Create and correctly name a histogram for each subdetector*Z_Slice*ModuleType
      histosx[ix][ihist] = new TH1F("histx" + x[ix] + _sublevel_names[ihist % NB_SUBLEVELS] +
                                        TString(ihist % (NB_SUBLEVELS * NB_Z_SLICES) >= NB_SUBLEVELS ? "zn" : "zp") +
                                        TString(ihist >= NB_SUBLEVELS * NB_Z_SLICES ? "xn" : "xp"),
                                    "",
                                    1000,
                                    _min[x[ix]],
                                    _max[x[ix]] + 1.);
      histosx[ix][ihist]->StatOverflows(kTRUE);
    }
  }

#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: Creation of the TH1F[" << y.size() << "]["
            << NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES << "] ended." << std::endl;
#endif

  /// 2) loop on the TTree data
#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: Looping on the TTree" << std::endl;
#endif
#ifdef TALKATIVE
  unsigned int progress = 0;
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: 0%" << std::endl;
#endif
  for (unsigned int ientry = 0; ientry < nentries; ientry++) {
#ifdef TALKATIVE
    if (10 * ientry / nentries != progress) {
      progress = 10 * ientry / nentries;
      std::cout << __FILE__ << ":" << __LINE__ << ":Info: " << 10 * progress << "%" << std::endl;
    }
#endif
    // load current tree entry
    data->GetEntry(ientry);

    // CUTS on entry
    if (branch_i["level"] != _levelCut)
      continue;
    if (!_1dModule && branch_i["detDim"] == 1)
      continue;
    if (!_2dModule && branch_i["detDim"] == 2)
      continue;

    for (unsigned int iy = 0; iy < y.size(); iy++) {
      if (branch_i["sublevel"] < 1 || branch_i["sublevel"] > NB_SUBLEVELS)
        continue;
      if (_SF[y[iy]] * branch_f[y[iy]] > _max[y[iy]] || _SF[y[iy]] * branch_f[y[iy]] < _min[y[iy]]) {
        //#ifdef DEBUG
        //                    std::cout << "branch_f[y[iy]]=" << branch_f[y[iy]] << std::endl;
        //#endif
        continue;
      }

      // FILLING HISTOGRAMS

      // histogram for all modules
      const short int ihisto = (branch_i["sublevel"] - 1) + (branch_f["z"] >= 0 ? 0 : NB_SUBLEVELS) +
                               (branch_f["x"] >= 0 ? 0 : NB_SUBLEVELS * NB_Z_SLICES);

      if (_module_plot_option == "all")
        histos[iy][ihisto]->Fill(_SF[y[iy]] * branch_f[y[iy]]);

      // Only good modules
      else if (_module_plot_option == "good" && branch_i["badModuleQuality"] == 0)
        histos[iy][ihisto]->Fill(_SF[y[iy]] * branch_f[y[iy]]);

      // Only good modules and those in the list
      else if (_module_plot_option == "list" && (branch_i["inModuleList"] == 1 || branch_i["badModuleQuality"] == 0))
        histos[iy][ihisto]->Fill(_SF[y[iy]] * branch_f[y[iy]]);
    }

    for (unsigned int ix = 0; ix < x.size(); ix++) {
      if (branch_i["sublevel"] < 1 || branch_i["sublevel"] > NB_SUBLEVELS)
        continue;
      if (_SF[x[ix]] * branch_f[x[ix]] > _max[x[ix]] || _SF[x[ix]] * branch_f[x[ix]] < _min[x[ix]]) {
        //#ifdef DEBUG
        //                    std::cout << "branch_f[y[iy]]=" << branch_f[y[iy]] << std::endl;
        //#endif
        continue;
      }

      // FILLING HISTOGRAMS

      // histogram for all modules
      const short int ihistosx = (branch_i["sublevel"] - 1) + (branch_f["z"] >= 0 ? 0 : NB_SUBLEVELS) +
                                 (branch_f["x"] >= 0 ? 0 : NB_SUBLEVELS * NB_Z_SLICES);

      if (_module_plot_option == "all")
        histosx[ix][ihistosx]->Fill(_SF[x[ix]] * branch_f[x[ix]]);

      // Only good modules
      else if (_module_plot_option == "good" && branch_i["badModuleQuality"] == 0)
        histosx[ix][ihistosx]->Fill(_SF[x[ix]] * branch_f[x[ix]]);

      // Only good modules and those in the list
      else if (_module_plot_option == "list" && (branch_i["inModuleList"] == 1 || branch_i["badModuleQuality"] == 0))
        histosx[ix][ihistosx]->Fill(_SF[x[ix]] * branch_f[x[ix]]);
    }
  }
#ifdef TALKATIVE
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: 100%\tLoop ended" << std::endl;
#endif

  //~ TString rangeLabel = "";
  // Calculate mean and standard deviation for each histogram
  for (unsigned int iy = 0; iy < y.size(); iy++) {
    for (unsigned int ihist = 0; ihist < NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES; ihist++) {
      // combine +/-z histograms for barrel detectors
      if (ihist % (NB_SUBLEVELS * NB_Z_SLICES) == 0 || ihist % (NB_SUBLEVELS * NB_Z_SLICES) == 2 ||
          ihist % (NB_SUBLEVELS * NB_Z_SLICES) == 4) {
        histos[iy][ihist]->Add(histos[iy][ihist + NB_SUBLEVELS]);
      }
      // combine +/-x histograms for endcap detectors (only used for half shells in barrel)
      if (ihist < NB_SUBLEVELS * NB_Z_SLICES &&
          (ihist % NB_SUBLEVELS == 1 || ihist % NB_SUBLEVELS == 3 || ihist % NB_SUBLEVELS == 5)) {
        histos[iy][ihist]->Add(histos[iy][ihist + NB_SUBLEVELS * NB_Z_SLICES]);
      }
      meanValue[iy][ihist] = histos[iy][ihist]->GetMean();
      RMS[iy][ihist] = histos[iy][ihist]->GetRMS();

      histos[iy][ihist]->Fit("gaus");
      gausFit[iy][ihist] = histos[iy][ihist]->GetFunction("gaus");
      meanValueGaussian[iy][ihist] = gausFit[iy][ihist]->GetParameter(1);
      RMSGaussian[iy][ihist] = gausFit[iy][ihist]->GetParameter(2);
    }
  }

  for (unsigned int ix = 0; ix < x.size(); ix++) {
    for (unsigned int ihist = 0; ihist < NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES; ihist++) {
      // combine +/-z histograms for barrel detectors
      if (ihist % (NB_SUBLEVELS * NB_Z_SLICES) == 0 || ihist % (NB_SUBLEVELS * NB_Z_SLICES) == 2 ||
          ihist % (NB_SUBLEVELS * NB_Z_SLICES) == 4) {
        histosx[ix][ihist]->Add(histosx[ix][ihist + NB_SUBLEVELS]);
      }
      // combine +/-x histograms for endcap detectors (only used for half shells in barrel)
      if (ihist < NB_SUBLEVELS * NB_Z_SLICES &&
          (ihist % NB_SUBLEVELS == 1 || ihist % NB_SUBLEVELS == 3 || ihist % NB_SUBLEVELS == 5)) {
        histosx[ix][ihist]->Add(histosx[ix][ihist + NB_SUBLEVELS * NB_Z_SLICES]);
      }
      meanValuex[ix][ihist] = histosx[ix][ihist]->GetMean();
      RMSx[ix][ihist] = histosx[ix][ihist]->GetRMS();
    }
  }

  TString tableFileName, tableCaption, tableAlign, tableHeadline;
  TString PXBpLine, PXBmLine, PXFpLine, PXFmLine, TIBpLine, TIBmLine, TOBpLine, TOBmLine, TIDpLine, TIDmLine, TECpLine,
      TECmLine;

  // table using mean and RMS, round to integers in µm etc.
  tableFileName = "table_differences.tex";
  if (_module_plot_option == "all")
    tableCaption = "Means and standard deviations of " + _alignment_name + " - " + _reference_name +
                   " for each subdetector, all modules used.";
  else if (_module_plot_option == "good")
    tableCaption = "Means and standard deviations of " + _alignment_name + " - " + _reference_name +
                   " for each subdetector, only good modules used.";
  else if (_module_plot_option == "list")
    tableCaption = "Means and standard deviations of " + _alignment_name + " - " + _reference_name +
                   " for each subdetector, good modules and those in given list used.";

  WriteTable(y, NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES, meanValue, RMS, "0", tableCaption, tableFileName);

  //~ // table using  Gaussian fit, round to integers in µm etc.
  tableFileName = "table_differences_Gaussian.tex";
  if (_module_plot_option == "all")
    tableCaption = "Means and standard deviations for Gaussian fit of " + _alignment_name + " - " + _reference_name +
                   " for each subdetector, all modules used.";
  else if (_module_plot_option == "good")
    tableCaption = "Means and standard deviations for Gaussian fit of " + _alignment_name + " - " + _reference_name +
                   " for each subdetector, only good modules used.";
  else if (_module_plot_option == "list")
    tableCaption = "Means and standard deviations for Gaussian fit of " + _alignment_name + " - " + _reference_name +
                   " for each subdetector, good modules and those in given list used.";

  WriteTable(
      y, NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES, meanValueGaussian, RMSGaussian, "0", tableCaption, tableFileName);

  // Table for the mean positions on the x-axis, round to 3 digits in cm etc.
  tableFileName = "table_meanPos.tex";

  if (_module_plot_option == "all")
    tableCaption = "Mean positions and standard deviations in " + _reference_name +
                   " geometry for each subdetector, all modules used.";
  else if (_module_plot_option == "good")
    tableCaption = "Mean positions and standard deviations in " + _reference_name +
                   " geometry for each subdetector, only good modules used.";
  else if (_module_plot_option == "list")
    tableCaption = "Mean positions and standard deviations in " + _reference_name +
                   " geometry for each subdetector, good modules and those in given list used.";

  WriteTable(x, NB_SUBLEVELS * NB_Z_SLICES * NB_X_SLICES, meanValuex, RMSx, "3", tableCaption, tableFileName);

#ifdef TALKATIVE
  std::cout << __FILE__ << ":" << __LINE__ << ":Info: End of MakeLegends method" << std::endl;
#endif
}

// OPTION METHODS
void GeometryComparisonPlotter::SetPrint(const bool kPrint) { _print = kPrint; }
void GeometryComparisonPlotter::SetLegend(const bool kLegend) { _legend = kLegend; }
void GeometryComparisonPlotter::SetWrite(const bool kWrite) { _write = kWrite; }
void GeometryComparisonPlotter::Set1dModule(const bool k1dModule) { _1dModule = k1dModule; }
void GeometryComparisonPlotter::Set2dModule(const bool k2dModule) { _2dModule = k2dModule; }
void GeometryComparisonPlotter::SetLevelCut(const int kLevelCut) { _levelCut = kLevelCut; }
void GeometryComparisonPlotter::SetBatchMode(const bool kBatchMode) { _batchMode = kBatchMode; }
void GeometryComparisonPlotter::SetGrid(const int kGridX, const int kGridY) {
  _grid_x = kGridX;
  _grid_y = kGridY;
}
void GeometryComparisonPlotter::SetBranchMax(const TString branchname, const float max) { _max[branchname] = max; }
void GeometryComparisonPlotter::SetBranchMin(const TString branchname, const float min) { _min[branchname] = min; }
void GeometryComparisonPlotter::SetBranchSF(const TString branchname, const float SF) { _SF[branchname] = SF; }
void GeometryComparisonPlotter::SetBranchUnits(const TString branchname, const TString units) {
  _units[branchname] = units;
}
void GeometryComparisonPlotter::SetPrintOption(const Option_t *print_option) { _print_option = print_option; }
void GeometryComparisonPlotter::SetCanvasSize(const int window_width, const int window_height) {
  _window_width = window_width;
  _window_height = window_height;
}
void GeometryComparisonPlotter::SetOutputFileName(const TString name) { _output_filename = name; }
void GeometryComparisonPlotter::SetOutputDirectoryName(const TString name) {
  _output_directory = name + TString(name.EndsWith("/") ? "" : "/");
}

// PRIVATE METHODS
TString GeometryComparisonPlotter::LateXstyle(TString word) {
  word.ToLower();
  if (word.BeginsWith("d"))
    word.ReplaceAll("d", "#Delta");
  if (word == TString("rdphi"))
    word = "r#Delta#phi";  // TO DO: find something less ad hoc...
  else if (word.EndsWith("phi"))
    word.ReplaceAll("phi", "#phi");
  else if (word.EndsWith("alpha"))
    word.ReplaceAll("alpha", "#alpha");
  else if (word.EndsWith("beta"))
    word.ReplaceAll("beta", "#beta");
  else if (word.EndsWith("gamma"))
    word.ReplaceAll("gamma", "#gamma");
  else if (word.EndsWith("eta"))
    word.ReplaceAll("eta", "#eta");
  return word;
}

TString GeometryComparisonPlotter::LateXstyleTable(TString word) {
  word.ToLower();
  if (word.BeginsWith("d"))
    word.ReplaceAll("d", "$\\Delta$");
  if (word == TString("rdphi"))
    word = "r$\\Delta\\phi$";  // TO DO: find something less ad hoc...
  else if (word.EndsWith("phi"))
    word.ReplaceAll("phi", "$\\phi$");
  else if (word.EndsWith("alpha"))
    word.ReplaceAll("alpha", "$\\alpha$");
  else if (word.EndsWith("beta"))
    word.ReplaceAll("beta", "$\\beta$");
  else if (word.EndsWith("gamma"))
    word.ReplaceAll("gamma", "#$\\gamma$");
  else if (word.EndsWith("eta"))
    word.ReplaceAll("eta", "$\\eta$");
  return word;
}

TString GeometryComparisonPlotter::ExtensionFromPrintOption(TString print_option) {
  if (print_option.Contains("pdf"))
    return TString(".pdf");
  else if (print_option.Contains("eps"))
    return TString(".eps");
  else if (print_option.Contains("ps"))
    return TString(".ps");
  else if (print_option.Contains("svg"))
    return TString(".svg");
  else if (print_option.Contains("tex"))
    return TString(".tex");
  else if (print_option.Contains("gif"))
    return TString(".gif");
  else if (print_option.Contains("xpm"))
    return TString(".xpm");
  else if (print_option.Contains("png"))
    return TString(".png");
  else if (print_option.Contains("jpg"))
    return TString(".jpg");
  else if (print_option.Contains("tiff"))
    return TString(".tiff");
  else if (print_option.Contains("cxx"))
    return TString(".cxx");
  else if (print_option.Contains("xml"))
    return TString(".xml");
  else if (print_option.Contains("root"))
    return TString(".root");
  else {
    std::cout << __FILE__ << ":" << __LINE__ << ":Warning: unknown format. Returning .pdf, but possibly wrong..."
              << std::endl;
    return TString(".pdf");
  }
}

TLegend *GeometryComparisonPlotter::MakeLegend(
    double x1, double y1, double x2, double y2, int nPlottedSublevels, const TString title) {
  TLegend *legend = new TLegend(x1, y1, x2, y2, title.Data(), "NBNDC");
  legend->SetNColumns(nPlottedSublevels);
  legend->SetFillColor(0);
  legend->SetLineColor(0);  // redundant with option
  legend->SetLineWidth(0);  // redundant with option
  for (int isublevel = 0; isublevel < nPlottedSublevels;
       isublevel++)  // nPlottedSublevels is either NB_SUBLEVELS for the tracker or 2 for the pixel
  {
    TGraph *g = new TGraph(0);
    g->SetMarkerColor(COLOR_CODE(isublevel));
    g->SetFillColor(COLOR_CODE(isublevel));
    g->SetMarkerStyle(kFullSquare);
    g->SetMarkerSize(10);
    legend->AddEntry(g, _sublevel_names[isublevel], "p");
  }
  return legend;
}

void GeometryComparisonPlotter::WriteTable(std::vector<TString> x,
                                           unsigned int nLevelsTimesSlices,
                                           float meanValue[10][24],
                                           float RMS[10][24],
                                           const TString nDigits,
                                           const TString tableCaption,
                                           const TString tableFileName) {
  std::ofstream output(_output_directory + tableFileName);

  TString tableAlign, tableHeadline;
  TString PXBpLine, PXBmLine, PXFpLine, PXFmLine, TIBpLine, TIBmLine, TOBpLine, TOBmLine, TIDpLine, TIDmLine, TECpLine,
      TECmLine;
  char meanChar[x.size()][nLevelsTimesSlices][10];
  char RMSChar[x.size()][nLevelsTimesSlices][10];

  tableAlign = "l";
  tableHeadline = "";
  PXBpLine = "PXB x$+$";
  PXBmLine = "PXB x$-$";
  PXFpLine = "PXF z$+$";
  PXFmLine = "PXF z$-$";
  TIBpLine = "TIB x$+$";
  TIBmLine = "TIB x$-$";
  TIDpLine = "TID z$+$";
  TIDmLine = "TID z$-$";
  TOBpLine = "TOB x$+$";
  TOBmLine = "TOB x$-$";
  TECpLine = "TEC z$+$";
  TECmLine = "TEC z$-$";

  for (unsigned int ix = 0; ix < x.size(); ix++) {
    for (unsigned int isubDet = 0; isubDet < nLevelsTimesSlices; isubDet++) {
      sprintf(meanChar[ix][isubDet], "%." + nDigits + "f", meanValue[ix][isubDet]);
      sprintf(RMSChar[ix][isubDet], "%." + nDigits + "f", RMS[ix][isubDet]);
    }
    tableAlign += "|c";
    tableHeadline += " & " + LateXstyleTable(x[ix]) + " / " + _units[x[ix]].ReplaceAll("#mum", "$\\mu$m");

    PXBpLine += " & $";
    PXBpLine += meanChar[ix][0];
    PXBpLine += "\\pm";
    PXBpLine += RMSChar[ix][0];
    PXBpLine += " $";
    PXBmLine += " & $";
    PXBmLine += meanChar[ix][12];
    PXBmLine += "\\pm";
    PXBmLine += RMSChar[ix][12];
    PXBmLine += " $";
    PXFpLine += " & $";
    PXFpLine += meanChar[ix][1];
    PXFpLine += "\\pm";
    PXFpLine += RMSChar[ix][1];
    PXFpLine += " $";
    PXFmLine += " & $";
    PXFmLine += meanChar[ix][7];
    PXFmLine += "\\pm";
    PXFmLine += RMSChar[ix][7];
    PXFmLine += " $";
    TIBpLine += " & $";
    TIBpLine += meanChar[ix][2];
    TIBpLine += "\\pm";
    TIBpLine += RMSChar[ix][2];
    TIBpLine += " $";
    TIBmLine += " & $";
    TIBmLine += meanChar[ix][14];
    TIBmLine += "\\pm";
    TIBmLine += RMSChar[ix][14];
    TIBmLine += " $";
    TIDpLine += " & $";
    TIDpLine += meanChar[ix][3];
    TIDpLine += "\\pm";
    TIDpLine += RMSChar[ix][3];
    TIDpLine += " $";
    TIDmLine += " & $";
    TIDmLine += meanChar[ix][9];
    TIDmLine += "\\pm";
    TIDmLine += RMSChar[ix][9];
    TIDmLine += " $";
    TOBpLine += " & $";
    TOBpLine += meanChar[ix][4];
    TOBpLine += "\\pm";
    TOBpLine += RMSChar[ix][4];
    TOBpLine += " $";
    TOBmLine += " & $";
    TOBmLine += meanChar[ix][16];
    TOBmLine += "\\pm";
    TOBmLine += RMSChar[ix][16];
    TOBmLine += " $";
    TECpLine += " & $";
    TECpLine += meanChar[ix][5];
    TECpLine += "\\pm";
    TECpLine += RMSChar[ix][5];
    TECpLine += " $";
    TECmLine += " & $";
    TECmLine += meanChar[ix][11];
    TECmLine += "\\pm";
    TECmLine += RMSChar[ix][11];
    TECmLine += " $";
  }

  // Write the table to the tex file
  output << "\\begin{table}" << std::endl;
  output << "\\caption{" << tableCaption << "}" << std::endl;
  output << "\\begin{tabular}{" << tableAlign << "}" << std::endl;
  output << "\\hline" << std::endl;
  output << tableHeadline << " \\\\" << std::endl;
  output << "\\hline" << std::endl;
  output << PXBpLine << " \\\\" << std::endl;
  output << PXBmLine << " \\\\" << std::endl;
  output << PXFpLine << " \\\\" << std::endl;
  output << PXFmLine << " \\\\" << std::endl;
  output << TIBpLine << " \\\\" << std::endl;
  output << TIBmLine << " \\\\" << std::endl;
  output << TIDpLine << " \\\\" << std::endl;
  output << TIDmLine << " \\\\" << std::endl;
  output << TOBpLine << " \\\\" << std::endl;
  output << TOBmLine << " \\\\" << std::endl;
  output << TECpLine << " \\\\" << std::endl;
  output << TECmLine << " \\\\" << std::endl;
  output << "\\hline" << std::endl;
  output << "\\end{tabular}" << std::endl;
  output << "\\end{table}" << std::endl;
}
