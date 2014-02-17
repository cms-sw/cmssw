// @(#)root/hist:$Id: ExclusionBandPlot.cc,v 1.4 2009/05/15 09:55:59 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"
#include "math.h"

#if (defined (STANDALONE) or defined (__CINT__) )
   #include "ExclusionBandPlot.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/ExclusionBandPlot.h"
#endif
#include "TGraphAsymmErrors.h"
#include "TStyle.h"
#include "TAxis.h"


//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassImp(ExclusionBandPlot)
#endif
/*----------------------------------------------------------------------------*/
ExclusionBandPlot::ExclusionBandPlot(const char* name,
                                     const char* title,
                                     const int n_points,
                                     double* x_vals,
                                     double* y_vals,
                                     double* y_up_points1,
                                     double* y_down_points1,
                                     double* y_up_points2,
                                     double* y_down_points2):
                    StatisticalPlot(name,title,false){

    // Prepare errorbars
    double* y_down_bars2 = new double[n_points];
    double* y_down_bars1 = new double[n_points];
    double* y_up_bars1 = new double[n_points];
    double* y_up_bars2 = new double[n_points];

    for (int i=0;i<n_points;++i){
        y_down_bars2[i]=y_vals[i]-y_down_points2[i];
        y_down_bars1[i]=y_vals[i]-y_down_points1[i];
        y_up_bars2[i]=y_up_points2[i]-y_vals[i];
        y_up_bars1[i]=y_up_points1[i]-y_vals[i];
        }

    // bline
    m_y_line_graph = new TGraph(n_points, x_vals, y_vals);
    m_y_line_graph->SetLineWidth(2);
    m_y_line_graph->SetLineStyle(2);
    m_y_line_graph->SetFillColor(kWhite);



    // y band 1 sigma
    m_y_band_graph_1sigma = new TGraphAsymmErrors(n_points,
                                                  x_vals,
                                                  y_vals,
                                                  0,
                                                  0,
                                                  y_down_bars1,
                                                  y_up_bars1);
    m_y_band_graph_1sigma->SetFillColor(kGreen);
    m_y_band_graph_1sigma->SetLineColor(kGreen);
    m_y_band_graph_1sigma->SetMarkerColor(kGreen);

    // y band 2 sigma
    m_y_band_graph_2sigma = new TGraphAsymmErrors(n_points,
                                                  x_vals,
                                                  y_vals,
                                                  0,
                                                  0,
                                                  y_down_bars2,
                                                  y_up_bars2);
    m_y_band_graph_2sigma->SetFillColor(kYellow);
    m_y_band_graph_2sigma->SetFillColor(kYellow);
    m_y_band_graph_2sigma->SetLineColor(kYellow);
    m_y_band_graph_2sigma->SetMarkerColor(kYellow);
    m_y_band_graph_2sigma->GetYaxis()->SetTitle("#sigma/#sigma_{SM}");

    // Line for 1
    m_one_line = new TLine(m_y_line_graph->GetXaxis()->GetXmin(),1,
                           m_y_line_graph->GetXaxis()->GetXmax(),1);

    // The legend 

    m_legend = new TLegend(0.60,0.78,0.98,0.98);
    m_legend->SetName("SM exclusion");
    m_legend->AddEntry(m_y_band_graph_1sigma,"#pm 1#sigma");
    m_legend->AddEntry(m_y_band_graph_2sigma,"#pm 2#sigma");
    m_legend->AddEntry(m_y_line_graph,title);

    m_legend->SetFillColor(0);

    delete[] y_down_bars2;
    delete[] y_down_bars1;
    delete[] y_up_bars2;
    delete[] y_up_bars1;

    }

/*----------------------------------------------------------------------------*/

ExclusionBandPlot::~ExclusionBandPlot(){

    delete m_y_line_graph;

    delete m_y_band_graph_1sigma;
    delete m_y_band_graph_2sigma;

    delete m_one_line;

    delete m_legend;

    }

/*----------------------------------------------------------------------------*/

/**
The title of the x axis is here set only for the plot of the 2sigma band 
plot. Indeed its axes will be the only one to be drawn.
**/
void ExclusionBandPlot::setXaxisTitle(const char* title){
        m_y_band_graph_2sigma->GetXaxis()->SetTitle(title);
        }

/*----------------------------------------------------------------------------*/

/**
The title of the y axis is here set only for the plot of the 2sigma band 
plot. Indeed its axes will be the only one to be drawn.
**/
void ExclusionBandPlot::setYaxisTitle(const char* title){
        m_y_band_graph_2sigma->GetYaxis()->SetTitle(title);
        }

/*----------------------------------------------------------------------------*/

/**
The title is here set only for the plot of the 2sigma band 
plot. Indeed its axes will be the only one to be drawn.
**/
void ExclusionBandPlot::setTitle(const char* title){
    m_y_band_graph_2sigma->SetTitle(title);
    }

/*----------------------------------------------------------------------------*/

void ExclusionBandPlot::draw (const char* options){

    setCanvas(new TCanvas(GetName(),GetTitle()));
    getCanvas()->cd();

    getCanvas()->SetGridx();
    getCanvas()->SetGridy();

    TString opt(options);
    // Bands
    if (opt.Contains("4")==0){
        m_y_band_graph_2sigma->Draw("A3");
        m_y_band_graph_1sigma->Draw("3");
        }
    else{
        m_y_band_graph_2sigma->Draw("A4");
        m_y_band_graph_1sigma->Draw("4");
        }

    // Lines
    if (opt.Contains("4")==0){
        m_y_line_graph->Draw("L");
    }
    else{
        m_y_line_graph->Draw("C");
    }

    m_one_line->Draw("Same");

    // Legend
    m_legend->Draw("Same");

    }

/*----------------------------------------------------------------------------*/

void ExclusionBandPlot::dumpToFile (const char* RootFileName, const char* options){

    TFile ofile(RootFileName,options);
    ofile.cd();

    // Bands
    m_y_band_graph_2sigma->Write("band_2sigma");
    m_y_band_graph_1sigma->Draw("band_1sigma");

    // Lines
    m_y_line_graph->Draw("line");

    // Legend
    m_legend->Draw("IamTheLegend");

    ofile.Close();

    }

/*----------------------------------------------------------------------------*/

void ExclusionBandPlot::print (const char* options){
    std::cout << "\nExclusionBandPlot object " << GetName() << ":\n";
    }

/*----------------------------------------------------------------------------*/
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
