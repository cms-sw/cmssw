// @(#)root/hist:$Id: RatioFinderPlot.cc,v 1.1 2009/01/06 12:22:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"
#include <iostream>

#include "TString.h"
#include "TFile.h"

#include "PhysicsTools/RooStatsCms/interface/RatioFinderPlot.h"

/*----------------------------------------------------------------------------*/

RatioFinderPlot::RatioFinderPlot(const char* name,
                                 const char* title,
                                 double CL_level,
                                 std::map<double,double> points):
    StatisticalPlot(name,title,true){

    std::cout << "Constructing ratiofinderplot!\n";
    // Allocate and fill the Tgraph
    m_graph = new TGraph(points.size());
    int index=0;

    for (std::map<double,double>::iterator iter=points.begin();
        iter!=points.end();
        ++iter){
        m_graph->SetPoint(index,iter->first,iter->second);
        ++index;
        }
    TString s_name(GetName());
    m_graph->SetName((s_name+"_graph").Data());
    m_graph->SetTitle(title);
    m_graph->SetMarkerStyle(8);
    m_graph->SetMarkerSize(0.7);
    m_graph->SetLineColor(kBlue);
    m_graph->SetLineWidth(2);
    m_graph->GetYaxis()->SetTitle("CL_{s}");
    m_graph->GetXaxis()->SetTitle("Ratio");

    // The orizzontal line
    double x_low=m_graph->GetXaxis()->GetXmin();
    double x_high=m_graph->GetXaxis()->GetXmax();
    m_CL_line = new TLine(x_low,CL_level,x_high,CL_level);
    m_CL_line->SetLineWidth(2);

    }
/*----------------------------------------------------------------------------*/

RatioFinderPlot::~RatioFinderPlot(){
    delete m_CL_line;
    delete m_graph;
    }

/*----------------------------------------------------------------------------*/

void RatioFinderPlot::print(const char* options){
    std::cout << "\nRatioFinderPlot object " << GetName() << ":\n";
    }

/*----------------------------------------------------------------------------*/

void RatioFinderPlot::draw(const char* options){

    setCanvas(new TCanvas(GetName(),GetTitle()));
    getCanvas()->cd();
    getCanvas()->Draw(options);

    m_graph->Draw("APL");
    m_CL_line->Draw("same");
    }

/*----------------------------------------------------------------------------*/

void RatioFinderPlot::dumpToFile(const char* RootFileName, const char* options){

    TFile ofile(RootFileName,options);
    ofile.cd();

    m_CL_line->Write("CLs_line");
    m_graph->Write("Scan_points");

    ofile.Close();
    }

/*----------------------------------------------------------------------------*/

/// To build the cint dictionaries
//ClassImp(RatioFinderPlot)
