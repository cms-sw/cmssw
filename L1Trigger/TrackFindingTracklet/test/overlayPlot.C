#include "TROOT.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TMath.h"

#include <iostream>
#include <string>
#include <vector>

using namespace std;

void SetPlotStyle();
void mySmallText(Double_t x, Double_t y ,Color_t color, TString text); 


// ----------------------------------------------------------------------------------------------------------------
// OVERLAY PLOT: A function that takes in plot output files with a common prefix ("output_TTbar_PU200_", etc.)
//   and an output graph to overlay and overlays the output graphs on top of one another, using the part of the
//   files that isn't common as the legend.
//
//   This function was originally created to compare properties of tracks that come from different seeding 
//   configurations. To produce these files, go to interface/Constants.h and set "d0L1L2" (etc) flags to restrict
//   allowed seeding layers for the tracklet algorithm. Then, use L1TrackNtuplePlot.C to produce plot output files
//   from the results of these restrictions.
//
//   Parameters:
//   	- plot_name: name of the plot to be compared across plot output files. ("eff_eta_H", "tp_pt", etc.)
//   	- common_prefix: the name that the output files share. Can also begin with a directory.
//   	- suffixes: a list of the suffixes of the plot output files you are overlaying. This list
//   	  will be used as the legend.
//      - group_name: an optional parameter that allows easy separation of output files. For example, I
//        created this parameter to deal with comparing track properties for different tracklet seeding
//        configurations for electrons and muons separately.
//
//   "Sort of parameters" but actually just variables set in the code:
//   	- "displayText": This can be commendted out if no text needs to be displayed.
//   	- All of the plotting stuff. (Legend position, draw options, etc.)
//   	- output location: this function outputs a single graph to the location specified in "save output as pdf".
//
//   Run this file with:
//   	root <options> 'overlayPlot(
//   		<name of plot in output to overlay>,
//   	        <common prefix of output files>, 
//   	        <array of suffixes of output files>,
//   	        <optional group name>
//   	      )' 
void overlayPlot(TString plot_name, TString common_prefix, std::vector<TString> suffixes, TString group_name="") {

  SetPlotStyle();
  
  // load trees
  const int NUM_SUFFIXES = suffixes.size();
  TH1F* h_output[NUM_SUFFIXES];
  for (int suffix = 0; suffix < NUM_SUFFIXES; suffix++) {
    TFile* file = new TFile(common_prefix + suffixes[suffix] + ".root");
    h_output[suffix] = (TH1F*)file->Get(plot_name);
  }

  // set colors; draw the histograms
  TCanvas c;
  for (int suffix = 0; suffix < NUM_SUFFIXES; suffix++) {
    h_output[suffix]->SetLineColor(suffix+1); //here and below, uses plot index in files to choose color
    h_output[suffix]->SetMarkerColor(suffix+1);
    TString* options;
    if (suffix == 0) {
      h_output[suffix]->Draw("hist");
    } else {
      h_output[suffix]->Draw("hist, same");
    }
  }

  // add legend or other text depending on plot_name and group_name
  TLegend* l = new TLegend();
  l->SetFillColor(0);
  l->SetLineColor(1);
  l->SetTextSize(0.04);
  l->SetTextFont(42);
  l->SetHeader("Legend:");

  // add specific identifier for each plot output file to legend
  for (int suffix = 0; suffix < NUM_SUFFIXES; suffix++) {
    l->AddEntry(h_output[suffix], suffixes[suffix], "lep");
  }
  l->Draw();	

  // set grid lines  
  gPad->SetGridy();

  // Append group name to front with underscore if there is one
  TString group_name_print;
  if (group_name == "") {
    group_name_print = "";
  } else {
    group_name_print = group_name + "_";
  }

  // save as pdf
  TString rel_save_dir = "TrkPlots/OverlayPlots/";
  c.SaveAs(rel_save_dir + group_name_print + plot_name + ".pdf");

}


void SetPlotStyle() {

  // Below from ATLAS plot style macro

  // use plain black on white colors
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameFillColor(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetStatColor(0);
  gStyle->SetHistLineColor(1);

  gStyle->SetPalette(1);

  // set the paper & margin sizes
  gStyle->SetPaperSize(20,26);
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.16);

  // set title offsets (for axis label)
  gStyle->SetTitleXOffset(1.4);
  gStyle->SetTitleYOffset(1.4);

  // use large fonts
  gStyle->SetTextFont(42);
  gStyle->SetTextSize(0.05);
  gStyle->SetLabelFont(42,"x");
  gStyle->SetTitleFont(42,"x");
  gStyle->SetLabelFont(42,"y");
  gStyle->SetTitleFont(42,"y");
  gStyle->SetLabelFont(42,"z");
  gStyle->SetTitleFont(42,"z");
  gStyle->SetLabelSize(0.05,"x");
  gStyle->SetTitleSize(0.05,"x");
  gStyle->SetLabelSize(0.05,"y");
  gStyle->SetTitleSize(0.05,"y");
  gStyle->SetLabelSize(0.05,"z");
  gStyle->SetTitleSize(0.05,"z");

  // use bold lines and markers
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(1.2);
  gStyle->SetHistLineWidth(2.);
  gStyle->SetLineStyleString(2,"[12 12]");

  // get rid of error bar caps
  gStyle->SetEndErrorSize(0.);

  // do not display any of the standard histogram decorations
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  // put tick marks on top and RHS of plots
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

}


void mySmallText(Double_t x, Double_t y, Color_t color, TString text) {
  Double_t tsize=0.044;
  TLatex l;
  l.SetTextSize(tsize); 
  l.SetNDC();
  l.SetTextColor(color);
  l.DrawLatex(x,y,text);
}


