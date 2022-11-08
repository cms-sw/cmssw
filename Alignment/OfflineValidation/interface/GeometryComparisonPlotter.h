#ifndef Alignment_OfflineValidation_GeometryComparisonPlotter_h
#define Alignment_OfflineValidation_GeometryComparisonPlotter_h

#include <TROOT.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "TString.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include "TMultiGraph.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TTree.h"
#include "TDirectory.h"
#include "TMath.h"
#include "TPaveText.h"
#include "TLatex.h"
#include "TList.h"

#include "boost/property_tree/ptree.hpp"
namespace pt = boost::property_tree;

class GeometryComparisonPlotter {
  // internal variables
#ifndef NB_SUBLEVELS
#define NB_SUBLEVELS 6
#endif
  TString _sublevel_names[NB_SUBLEVELS], _output_directory, _output_filename, _print_option, _module_plot_option,
      _alignment_name, _reference_name;
  bool _print, _legend, _write, _print_only_global, _make_profile_plots, _batchMode, _1dModule, _2dModule;
  int _levelCut, _grid_x, _grid_y, _window_width, _window_height,
      _canvas_index;  // to append to the name of the canvases in case of duplication

  // branches
  std::map<TString, int> branch_i;
  std::map<TString, float> branch_f, _max, _min, _SF;
  std::map<TString, TString> _units;

  // variables of external objects
  TFile *tree_file;
  TFile *output;
  TTree *data;

  // methods
  TString LateXstyle(TString);
  TString LateXstyleTable(TString);
  TString ExtensionFromPrintOption(TString);
  TLegend *MakeLegend(double x1, double y1, double x2, double y2, int nPlottedSublevels, const TString title = "");

public:
  static int canvas_profile_index;  // to append to the name of the canvases in case of duplication

  // constructor and destructor
  GeometryComparisonPlotter(TString tree_file_name,
                            TString outputDirname = "output/",
                            TString modulesToPlot = "all",
                            TString referenceName = "Ideal",
                            TString alignmentName = "Alignment",
                            bool plotOnlyGlobal = false,
                            bool makeProfilePlots = false,
                            int canvas_idx = 0);
  ~GeometryComparisonPlotter();

  // main methods
  void MakePlots(const std::vector<TString>, const std::vector<TString>, pt::ptree CFG);

  void MakeTables(const std::vector<TString>, const std::vector<TString>, pt::ptree CFG);

  void WriteTable(const std::vector<TString> x,
                  unsigned int nLevelsTimesSlices,
                  float meanValue[10][24],
                  float RMS[10][24],
                  const TString nDigits,
                  const TString tableCaption,
                  const TString tableFileName);

  // option methods
  void SetPrint(const bool);     // activates the printing of the individual and global pdf
  void SetLegend(const bool);    // activates the legends
  void SetWrite(const bool);     // activates the writing into a Root file
  void Set1dModule(const bool);  // cut to include 1D modules
  void Set2dModule(const bool);  // cut to include 2D modules
#define DEFAULT_LEVEL 1
  void SetLevelCut(const int);    // module level: level=1 (default)
  void SetBatchMode(const bool);  // activates the display of the canvases
  void SetGrid(const int,         // activates the display of the grids
               const int);
  void SetBranchMax(const TString,    // sets a max value for the variable
                    const float);     // by giving the name and the value
  void SetBranchMin(const TString,    // sets a min value for the variable
                    const float);     // by giving the name and the value
  void SetBranchSF(const TString,     // sets a rescaling factor for the variable
                   const float);      // by giving the name and the value
  void SetBranchUnits(const TString,  // writes de units next on the axis
                      const TString);
  void SetOutputDirectoryName(const TString);  // sets the output name of the directory
  void SetOutputFileName(const TString);       // sets the name of the root file (if applicable)
  void SetPrintOption(const Option_t *);       // litteraly the print option of the TPad::Print()
#define DEFAULT_WINDOW_WIDTH 3508
#define DEFAULT_WINDOW_HEIGHT 2480
  void SetCanvasSize(const int window_width = DEFAULT_WINDOW_WIDTH, const int window_height = DEFAULT_WINDOW_HEIGHT);
};
#endif
