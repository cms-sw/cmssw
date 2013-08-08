
/** \executable showROCcurvesTauIdMVA
 *
 * Show curves of backgreound rejection vs. signal efficiency
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: showROCcurvesTauIdMVA.cc,v 1.1 2012/03/06 17:34:42 veelken Exp $
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "DataFormats/FWLite/interface/InputSource.h"
#include "DataFormats/FWLite/interface/OutputFiles.h"

#include "RecoTauTag/TauAnalysisTools/bin/tauIdMVATrainingAuxFunctions.h"

#include <TSystem.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveText.h>
#include <TBenchmark.h>
#include <TROOT.h>

#include <iostream>
#include <string>
#include <vector>
#include <assert.h>

typedef std::vector<std::string> vstring;

namespace
{
  struct graphEntryType
  {
    graphEntryType(const edm::ParameterSet& cfg)
      : graph_(0)
    {
      graphName_   = cfg.getParameter<std::string>("graphName");
      legendEntry_ = cfg.getParameter<std::string>("legendEntry");
      markerSize_  = ( cfg.exists("markerSize")  ) ? cfg.getParameter<int>("markerSize")  : 1;
      markerStyle_ = ( cfg.exists("markerStyle") ) ? cfg.getParameter<int>("markerStyle") : 1;
      markerColor_ = cfg.getParameter<int>("color");
      lineWidth_   = ( cfg.exists("lineWidth")   ) ? cfg.getParameter<int>("lineWidth")   : 1;
      lineStyle_   = ( cfg.exists("lineStyle")   ) ? cfg.getParameter<int>("lineStyle")   : 1;
      lineColor_   = cfg.getParameter<int>("color");
    }
    ~graphEntryType() {}
    std::string graphName_;
    TGraph* graph_;
    std::string legendEntry_;
    int markerSize_;
    int markerStyle_;
    int markerColor_;
    int lineWidth_;
    int lineStyle_;
    int lineColor_;
  };

  void showGraphs(const TString& title, double canvasSizeX, double canvasSizeY,
		  const std::vector<graphEntryType*>& graphEntries,
		  double xMin, double xMax, const std::string& xAxisTitle, double xAxisOffset,
		  bool useLogScale, double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset,
		  double legendX0, double legendX1, double legendY0, double legendY1, 
		  const std::string& outputFileName)
  {
    TCanvas* canvas = new TCanvas("canvas", "canvas", canvasSizeX, canvasSizeY);
    canvas->SetFillColor(10);
    canvas->SetBorderSize(2);
    canvas->SetLeftMargin(0.12);
    canvas->SetBottomMargin(0.12);
    canvas->SetLogy(useLogScale);
    canvas->SetGridx();
    canvas->SetGridy();

    TH1* dummyHistogram = new TH1D("dummyHistogram", "dummyHistogram", 10, xMin, xMax);
    dummyHistogram->SetTitle("");
    dummyHistogram->SetStats(false);
    dummyHistogram->SetMinimum(yMin);
    dummyHistogram->SetMaximum(yMax);
  
    TAxis* xAxis = dummyHistogram->GetXaxis();
    xAxis->SetTitle(xAxisTitle.data());
    xAxis->SetTitleOffset(xAxisOffset);

    TAxis* yAxis = dummyHistogram->GetYaxis();
    yAxis->SetTitle(yAxisTitle.data());
    yAxis->SetTitleOffset(yAxisOffset);

    dummyHistogram->Draw();
    
    TLegend* legend = new TLegend(legendX0, legendY0, legendX1, legendY1, "", "brNDC"); 
    legend->SetBorderSize(0);
    legend->SetFillColor(0);
    
    for ( std::vector<graphEntryType*>::const_iterator graphEntry = graphEntries.begin();
	  graphEntry != graphEntries.end(); ++graphEntry ) {
      assert((*graphEntry)->graph_);
      if ( (*graphEntry)->graph_->GetN() < 10 ) {
	(*graphEntry)->graph_->Draw("P");
	if ( (*graphEntry)->legendEntry_ != "" ) legend->AddEntry((*graphEntry)->graph_, (*graphEntry)->legendEntry_.data(), "p");
      } else {
	(*graphEntry)->graph_->Draw("L");
	if ( (*graphEntry)->legendEntry_ != "" ) legend->AddEntry((*graphEntry)->graph_, (*graphEntry)->legendEntry_.data(), "l");
      }
    }

    legend->Draw();

    TPaveText* label = 0;
    if ( title.Length() > 0 ) {
      label = new TPaveText(0.175, 0.925, 0.48, 0.98, "NDC");
      label->AddText(title.Data());
      label->SetTextAlign(13);
      label->SetTextSize(0.045);
      label->SetFillStyle(0);
      label->SetBorderSize(0);
      label->Draw();
    }

    canvas->Update();
    size_t idx = outputFileName.find_last_of('.');
    std::string outputFileName_plot = std::string(outputFileName, 0, idx);
    if ( useLogScale ) outputFileName_plot.append("_log");
    else outputFileName_plot.append("_linear");
    if ( idx != std::string::npos ) canvas->Print(std::string(outputFileName_plot).append(std::string(outputFileName, idx)).data());
    canvas->Print(std::string(outputFileName_plot).append(".png").data());
    //canvas->Print(std::string(outputFileName_plot).append(".pdf").data());
    canvas->Print(std::string(outputFileName_plot).append(".root").data());
  
    delete legend;
    delete label;
    delete dummyHistogram;
    delete canvas;
  }
}

int main(int argc, char* argv[]) 
{
//--- parse command-line arguments
  if ( argc < 2 ) {
    std::cout << "Usage: " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  std::cout << "<showROCcurvesTauIdMVA>:" << std::endl;

//--- keep track of time it takes the macro to execute
  TBenchmark clock;
  clock.Start("showROCcurvesTauIdMVA");

//--- CV: disable automatic association of histograms to files
//       (default behaviour ROOT object ownership, 
//        cf. http://root.cern.ch/download/doc/Users_Guide_5_26.pdf section 8)
  TH1::AddDirectory(false);

//--- read python configuration parameters
  if ( !edm::readPSetsFrom(argv[1])->existsAs<edm::ParameterSet>("process") ) 
    throw cms::Exception("showROCcurvesTauIdMVA") 
      << "No ParameterSet 'process' found in configuration file = " << argv[1] << " !!\n";

  edm::ParameterSet cfg = edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process");

  edm::ParameterSet cfgShowROCcurvesTauIdMVA = cfg.getParameter<edm::ParameterSet>("showROCcurvesTauIdMVA");

  std::vector<graphEntryType*> graphEntries;
  
  typedef std::vector<edm::ParameterSet> vParameterSet;
  vParameterSet cfgGraphs = cfgShowROCcurvesTauIdMVA.getParameter<vParameterSet>("graphs");
  for ( vParameterSet::const_iterator cfgGraph = cfgGraphs.begin();
	cfgGraph != cfgGraphs.end(); ++cfgGraph ) {
    graphEntries.push_back(new graphEntryType(*cfgGraph));
  }

  double xMin = cfgShowROCcurvesTauIdMVA.getParameter<double>("xMin");
  double xMax = cfgShowROCcurvesTauIdMVA.getParameter<double>("xMax");
  double yMin = cfgShowROCcurvesTauIdMVA.getParameter<double>("yMin");
  double yMax = cfgShowROCcurvesTauIdMVA.getParameter<double>("yMax");

  std::string outputFileName = cfgShowROCcurvesTauIdMVA.getParameter<std::string>("outputFileName");
  std::cout << " outputFileName = " << outputFileName << std::endl;

  fwlite::InputSource inputFiles(cfg);  

  typedef std::pair<double, double> pdouble;
  std::vector<pdouble> tauPtBins;
  tauPtBins.push_back(pdouble(  -1.,   50.));
  tauPtBins.push_back(pdouble(  50.,  100.));
  tauPtBins.push_back(pdouble( 100.,  200.));
  tauPtBins.push_back(pdouble( 200.,  400.));
  tauPtBins.push_back(pdouble( 400.,  600.));
  tauPtBins.push_back(pdouble( 600.,  900.));
  tauPtBins.push_back(pdouble( 900., 1200.));
  tauPtBins.push_back(pdouble(1200.,   -1.));
  tauPtBins.push_back(pdouble(  -1.,   -1.));
  
  for ( std::vector<pdouble>::const_iterator tauPtBin = tauPtBins.begin();
	tauPtBin != tauPtBins.end(); ++tauPtBin ) {
    for ( vstring::const_iterator inputFileName = inputFiles.files().begin();
	  inputFileName != inputFiles.files().end(); ++inputFileName ) {
      TFile* inputFile = new TFile(inputFileName->data());
      if ( !inputFile ) {
	throw cms::Exception("showROCcurvesTauIdMVA") 
	  << "Failed to open input file = " << (*inputFileName) << " !!\n";
      }
      for ( std::vector<graphEntryType*>::iterator graphEntry = graphEntries.begin();
	    graphEntry != graphEntries.end(); ++graphEntry ) {
	std::string graphName_full = (*graphEntry)->graphName_;
	graphName_full.append(getTauPtLabel(tauPtBin->first, tauPtBin->second));
	TGraph* graph = dynamic_cast<TGraph*>(inputFile->Get(graphName_full.data()));
	if ( graph ) {
	  if ( (*graphEntry)->graph_ ) {
	    throw cms::Exception("showROCcurvesTauIdMVA") 
	      << "Graphs of name = " << graphName_full << " found in multiple input files !!\n";
	  }
	  (*graphEntry)->graph_ = (TGraph*)graph->Clone(Form("%s_cloned", graph->GetName()));
	} 
      }
      delete inputFile;
    }
    
    for ( std::vector<graphEntryType*>::iterator graphEntry = graphEntries.begin();
	  graphEntry != graphEntries.end(); ++graphEntry ) {
      if ( (*graphEntry)->graph_ ) { 
	std::cout << "graph = " << (*graphEntry)->graph_->GetName() << ": #points = " << (*graphEntry)->graph_->GetN() << std::endl;
	for ( int iPoint = 0; iPoint < (*graphEntry)->graph_->GetN(); ++iPoint ) {
	  double x, y;
	  (*graphEntry)->graph_->GetPoint(iPoint, x, y);
	  std::cout << " point #" << iPoint << ": x = " << x << ", y = " << y << std::endl;
	}
	(*graphEntry)->graph_->SetMarkerSize((*graphEntry)->markerSize_);
	(*graphEntry)->graph_->SetMarkerStyle((*graphEntry)->markerStyle_);
	(*graphEntry)->graph_->SetMarkerColor((*graphEntry)->markerColor_);
	(*graphEntry)->graph_->SetLineWidth((*graphEntry)->lineWidth_);
	(*graphEntry)->graph_->SetLineStyle((*graphEntry)->lineStyle_);
	(*graphEntry)->graph_->SetLineColor((*graphEntry)->lineColor_);
	if ( (*graphEntry)->graph_->GetN() < 10 ) { // CV: remove "trivial" points (signal and background events either all pass or all fail)
	  int numPoints = (*graphEntry)->graph_->GetN();
	  int iPoint = 0;
	  while ( iPoint < numPoints ) {
	    double x, y;
	    (*graphEntry)->graph_->GetPoint(iPoint, x, y);
	    const double epsilon = 1.e-3;
	    if ( (fabs(x) < epsilon || fabs(1. - x) < epsilon) &&
		 (fabs(y) < epsilon || fabs(1. - y) < epsilon) ) {
	      std::cout << "removing point #" << iPoint << " @ (x = " << x << ", y = " << y << ")" << std::endl;
	      (*graphEntry)->graph_->RemovePoint(iPoint);
	      iPoint = 0;
	      numPoints = (*graphEntry)->graph_->GetN();
	      std::cout << " #points remaining = " << numPoints << std::endl;
	    } else {
	      ++iPoint;
	    }
	  }
	}
	std::cout << "#points to be drawn = " << (*graphEntry)->graph_->GetN() << ":" << std::endl;
	for ( int iPoint = 0; iPoint < (*graphEntry)->graph_->GetN(); ++iPoint ) {
	  double x, y;
	  (*graphEntry)->graph_->GetPoint(iPoint, x, y);
	  std::cout << " point #" << iPoint << ": x = " << x << ", y = " << y << std::endl;
	}
      } else {
	throw cms::Exception("showROCcurvesTauIdMVA") 
	  << "Graph of name = " << (*graphEntry)->graphName_ << " not found in any input file !!\n";
      }
    }

    int numLegendEntries = 0;
    for ( std::vector<graphEntryType*>::const_iterator graphEntry = graphEntries.begin();
	  graphEntry != graphEntries.end(); ++graphEntry ) {
      if ( (*graphEntry)->graph_ && (*graphEntry)->legendEntry_ != "" ) ++numLegendEntries;
    }

    size_t idx = outputFileName.find_last_of('.');
    std::string outputFileName_full = std::string(outputFileName, 0, idx);
    outputFileName_full.append(getTauPtLabel(tauPtBin->first, tauPtBin->second));
    
    showGraphs("#tau_{had} Efficiency vs. Fake-rate", 800, 600,
	       graphEntries,
	       xMin, xMax, "Signal Efficiency", 1.2,
	       false, yMin, yMax, "Background Rejection", 1.2,
	       0.145, 0.145 + 0.440, 0.145, 0.145 + 0.050*numLegendEntries,
	       outputFileName_full);
    for ( std::vector<graphEntryType*>::iterator graphEntry = graphEntries.begin();
	  graphEntry != graphEntries.end(); ++graphEntry ) {
      for ( int iPoint = 0; iPoint < (*graphEntry)->graph_->GetN(); ++iPoint ) {
	double x, y;
	(*graphEntry)->graph_->GetPoint(iPoint, x, y);
	(*graphEntry)->graph_->SetPoint(iPoint, x, 1. - y);
      }
    }
    showGraphs("#tau_{had} Efficiency vs. Fake-rate", 800, 600,
	       graphEntries,
	       xMin, xMax, "Efficiency", 1.2,
	       true, yMin, yMax, "Fake-rate", 1.2,
	       0.145, 0.145 + 0.440, 0.875 - 0.050*numLegendEntries, 0.875,
	       outputFileName_full);

    for ( std::vector<graphEntryType*>::iterator graphEntry = graphEntries.begin();
	  graphEntry != graphEntries.end(); ++graphEntry ) {
      delete (*graphEntry)->graph_;
      (*graphEntry)->graph_ = 0;
    }
  }
  
  for ( std::vector<graphEntryType*>::iterator it = graphEntries.begin();
	it != graphEntries.end(); ++it ) {
    delete (*it);
  }
  
  clock.Show("showROCcurvesTauIdMVA");

  return 0;
}
