// @(#)root/hist:$Id: LEPBandPlot.cc,v 1.4 2009/05/15 09:55:59 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"
#include "math.h"

#if (defined (STANDALONE) or defined (__CINT__) )
   #include "LEPBandPlot.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/LEPBandPlot.h"
#endif
#include "TGraphAsymmErrors.h"
#include "TStyle.h"
#include "TAxis.h"


//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassImp(LEPBandPlot)
#endif
/*----------------------------------------------------------------------------*/

/// Constructor
LEPBandPlot::LEPBandPlot(const char* name,
                         const char* title,
                         const int n_points,
                         double* x_vals,
                         double* sb_vals,
                         double* b_vals,
                         double* b_rms,
                         double* exp_vals):
    StatisticalPlot(name,title,false){

    // bkg hypothesis line
    m_b_line_graph = new TGraph(n_points, x_vals, b_vals);
    m_b_line_graph->SetLineWidth(2);
    m_b_line_graph->SetLineStyle(2);
    m_b_line_graph->SetFillColor(kWhite);


    // bkg hypothesis band 1 sigma
    m_b_band_graph_1sigma = new TGraphErrors(n_points, x_vals, b_vals ,0, b_rms);
    m_b_band_graph_1sigma->SetFillColor(kGreen);
    m_b_band_graph_1sigma->SetLineColor(kGreen);
    m_b_band_graph_1sigma->SetMarkerColor(kGreen);

    // Make the band 2 times wider:
    double* b_2rms = new double[n_points];
    for (int i=0;i<n_points;++i)
        b_2rms[i]=2*b_rms[i];

    // bkg hypothesis band 2 sigma
    m_b_band_graph_2sigma = new TGraphErrors(n_points, x_vals, b_vals ,0, b_2rms);
    m_b_band_graph_2sigma->SetFillColor(kYellow);
    m_b_band_graph_2sigma->SetFillColor(kYellow);
    m_b_band_graph_2sigma->SetLineColor(kYellow);
    m_b_band_graph_2sigma->SetMarkerColor(kYellow);
    m_b_band_graph_2sigma->GetYaxis()->SetTitle("-2lnQ");

    // sig+bkg hypothesis line
    m_sb_line_graph = new TGraph(n_points, x_vals, sb_vals);
    m_sb_line_graph->SetLineWidth(2);
    m_sb_line_graph->SetLineStyle(4);
    m_sb_line_graph->SetLineColor(kRed);
    m_sb_line_graph->SetFillColor(kWhite);

    // The points of the data
    if (exp_vals!=0){
        m_data_line_graph = new TGraph(n_points, x_vals, exp_vals);
        m_data_line_graph->SetLineWidth(2);
        m_data_line_graph->SetFillColor(kWhite);
        }
    else
        m_data_line_graph =NULL;


    // Line for 0
    m_zero_line = new TLine(m_b_line_graph->GetXaxis()->GetXmin(),0,
                            m_b_line_graph->GetXaxis()->GetXmax(),0);

    // The legend 

    m_legend = new TLegend(0.75,0.78,0.98,0.98);
    m_legend->SetName("Confidence Levels");
    m_legend->AddEntry(m_b_band_graph_1sigma,"-2lnQ #pm 1#sigma");
    m_legend->AddEntry(m_b_band_graph_2sigma,"-2lnQ #pm 2#sigma");
    m_legend->AddEntry(m_b_line_graph,"-2lnQ_{B}");
    m_legend->AddEntry(m_sb_line_graph,"-2lnQ_{SB}");
    if (m_data_line_graph!=NULL)
        m_legend->AddEntry(m_data_line_graph,"-2lnQ_{Obs}");

    m_legend->SetFillColor(0);

    delete[] b_2rms;
    }

/*----------------------------------------------------------------------------*/

LEPBandPlot::LEPBandPlot(const char* name,
                         const char* title,
                         const int n_points,
                         double* x_vals,
                         double* sb_vals,
                         double* b_vals,
                         double* b_up_bars1,
                         double* b_down_bars1,
                         double* b_up_bars2,
                         double* b_down_bars2,
                         double* exp_vals):
    StatisticalPlot(name,title,false){

    // bkg hypothesis line
    m_b_line_graph = new TGraph(n_points, x_vals, b_vals);
    m_b_line_graph->SetLineWidth(2);
    m_b_line_graph->SetLineStyle(2);
    m_b_line_graph->SetFillColor(kWhite);



    // bkg hypothesis band 1 sigma
    m_b_band_graph_1sigma = new TGraphAsymmErrors(n_points,
                                                  x_vals,
                                                  b_vals,
                                                  0,
                                                  0,
                                                  b_down_bars1,
                                                  b_up_bars1);
    m_b_band_graph_1sigma->SetFillColor(kGreen);
    m_b_band_graph_1sigma->SetLineColor(kGreen);
    m_b_band_graph_1sigma->SetMarkerColor(kGreen);

    // bkg hypothesis band 2 sigma
    m_b_band_graph_2sigma = new TGraphAsymmErrors(n_points,
                                                  x_vals,
                                                  b_vals,
                                                  0,
                                                  0,
                                                  b_down_bars2,
                                                  b_up_bars2);
    m_b_band_graph_2sigma->SetFillColor(kYellow);
    m_b_band_graph_2sigma->SetFillColor(kYellow);
    m_b_band_graph_2sigma->SetLineColor(kYellow);
    m_b_band_graph_2sigma->SetMarkerColor(kYellow);
    m_b_band_graph_2sigma->GetYaxis()->SetTitle("-2lnQ");

    // sig+bkg hypothesis line
    m_sb_line_graph = new TGraph(n_points, x_vals, sb_vals);
    m_sb_line_graph->SetLineWidth(2);
    m_sb_line_graph->SetLineStyle(4);
    m_sb_line_graph->SetLineColor(kRed);
    m_sb_line_graph->SetFillColor(kWhite);

    // The points of the data
    if (exp_vals!=0){
        m_data_line_graph = new TGraph(n_points, x_vals, exp_vals);
        m_data_line_graph->SetLineWidth(2);
        m_data_line_graph->SetFillColor(kWhite);
        }
    else
        m_data_line_graph =0;

    // Line for 0
    m_zero_line = new TLine(m_b_line_graph->GetXaxis()->GetXmin(),0,
                            m_b_line_graph->GetXaxis()->GetXmax(),0);

    // The legend 

    m_legend = new TLegend(0.75,0.78,0.98,0.98);
    m_legend->SetName("Confidence Levels");
    m_legend->AddEntry(m_b_band_graph_1sigma,"-2lnQ #pm 1#sigma");
    m_legend->AddEntry(m_b_band_graph_2sigma,"-2lnQ #pm 2#sigma");
    m_legend->AddEntry(m_b_line_graph,"-2lnQ_{B}");
    m_legend->AddEntry(m_sb_line_graph,"-2lnQ_{SB}");
    if (m_data_line_graph!=NULL)
        m_legend->AddEntry(m_data_line_graph,"-2lnQ_{Obs}");

    m_legend->SetFillColor(0);
    }

/*----------------------------------------------------------------------------*/

LEPBandPlot::~LEPBandPlot(){

    delete m_b_line_graph;
    delete m_sb_line_graph;

    delete m_b_band_graph_1sigma;
    delete m_b_band_graph_2sigma;

    if (m_data_line_graph!=NULL)
        delete m_data_line_graph;

    delete m_zero_line;

    delete m_legend;

    }

/*----------------------------------------------------------------------------*/

/**
The title of the x axis is here set only for the plot of the 2sigma band 
plot. Indeed its axes will be the only one to be drawn.
**/
void LEPBandPlot::setXaxisTitle(const char* title){
        m_b_band_graph_2sigma->GetXaxis()->SetTitle(title);
        }

/*----------------------------------------------------------------------------*/

/**
The title is here set only for the plot of the 2sigma band 
plot. Indeed its axes will be the only one to be drawn.
**/
void LEPBandPlot::setTitle(const char* title){
    m_b_band_graph_2sigma->SetTitle(title);
    }

/*----------------------------------------------------------------------------*/

void LEPBandPlot::draw (const char* options){

    setCanvas(new TCanvas(GetName(),GetTitle()));
    getCanvas()->cd();

    getCanvas()->SetGridx();
    getCanvas()->SetGridy();

    TString opt(options);
    // Bands
    if (opt.Contains("4")==0){
        m_b_band_graph_2sigma->Draw("A3");
        m_b_band_graph_1sigma->Draw("3");
        }
    else{
        m_b_band_graph_2sigma->Draw("A4");
        m_b_band_graph_1sigma->Draw("4");
        }

    // Lines
    if (opt.Contains("4")==0){
        m_b_line_graph->Draw("L");
        m_sb_line_graph->Draw("L");
    }
    else{
        m_b_line_graph->Draw("C");
        m_sb_line_graph->Draw("C");
    }

    if (m_data_line_graph!=NULL)
        m_data_line_graph->Draw("L");

    m_zero_line->Draw("Same");

    // Legend
    m_legend->Draw("Same");

    }

/*----------------------------------------------------------------------------*/

void LEPBandPlot::dumpToFile (const char* RootFileName, const char* options){

    TFile ofile(RootFileName,options);
    ofile.cd();

    // Bands
    m_b_band_graph_2sigma->Write("bkg_band_2sigma");
    m_b_band_graph_1sigma->Draw("bkg_band_1sigma");

    // Lines
    m_b_line_graph->Draw("bkg_line");
    m_sb_line_graph->Draw("sigbkg_line");
    if (m_data_line_graph)
        m_data_line_graph->Draw("observed_line");

    // Legend
    m_legend->Draw("IamTheLegend");

    ofile.Close();

    }

/*----------------------------------------------------------------------------*/

void LEPBandPlot::print (const char* options){
    std::cout << "\nLEPBandPlot object " << GetName() << ":\n";
    }

/*----------------------------------------------------------------------------*/
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
