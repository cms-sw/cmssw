// @(#)root/hist:$Id: LimitPlot.cc,v 1.1 2009/01/06 12:22:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"
#include <math.h>

#include "PhysicsTools/RooStatsCms/interface/LimitPlot.h"
#include "TStyle.h"

/// To build the cint dictionaries
//ClassImp(LimitPlot)

/*----------------------------------------------------------------------------*/

LimitPlot::LimitPlot(const char* name,
                     const  char* title,
                     std::vector<float> sb_vals,
                     std::vector<float> b_vals,
                     float m2lnQ_data,
                     int n_bins,
                     bool verbosity):
    m_b_histo_shaded(NULL),
    m_sb_histo_shaded(NULL),
    StatisticalPlot(name,title,verbosity){

    // Get the max and the min of the plots
    int n_toys=sb_vals.size();

    assert (n_toys >0);

    double max=-1e40;
    double min=1e40;

    // Extremes of the plot
    for (int i=0;i<n_toys;++i){
        if (sb_vals[i]>max)
            max=sb_vals[i];
        if (b_vals[i]>max)
            max=b_vals[i];
        if (sb_vals[i]<min)
            min=sb_vals[i];
        if (b_vals[i]<min)
            min=b_vals[i];
        }

    if (m2lnQ_data<min)
        min=m2lnQ_data;

    if (m2lnQ_data>max)
        max=m2lnQ_data;

    min*=1.1;
    max*=1.1;

    // Build the histos
    //int n_bins=100;

    m_sb_histo = new TH1F ("SB_model",title,n_bins,min,max);
    m_sb_histo->SetTitle(m_sb_histo->GetTitle());
    m_sb_histo->SetLineColor(kBlue);
    m_sb_histo->GetXaxis()->SetTitle("-2lnQ");
    //m_sb_histo->GetYaxis()->SetTitle("Entries");
    m_sb_histo->SetLineWidth(2);

    m_b_histo = new TH1F ("B_model",title,n_bins,min,max);
    m_b_histo->SetTitle(m_b_histo->GetTitle());
    m_b_histo->SetLineColor(kRed);
    m_b_histo->GetXaxis()->SetTitle("-2lnQ");
    //m_b_histo->GetYaxis()->SetTitle("Entries");
    m_b_histo->SetLineWidth(2);


    for (int i=0;i<n_toys;++i){
        m_sb_histo->Fill(sb_vals[i]);
        m_b_histo->Fill(b_vals[i]);
        }

    double histos_max_y=m_sb_histo->GetMaximum();
    if (histos_max_y<m_b_histo->GetMaximum())
        histos_max_y=m_b_histo->GetMaximum();

    double line_hight=histos_max_y/n_toys;

    // Build the line of the measured -2lnQ
    m_data_m2lnQ_line = new TLine(m2lnQ_data,0,m2lnQ_data,line_hight);
    m_data_m2lnQ_line->SetLineWidth(3);
    m_data_m2lnQ_line->SetLineColor(kBlack);

    // The legend
    double golden_section=(sqrt(5)-1)/2;

    m_legend = new TLegend(0.75,0.95-0.2*golden_section,0.95,0.95);
    TString title_leg="-2lnQ distributions ";
    title_leg+=sb_vals.size();
    title_leg+=" toys";
    m_legend->SetName(title_leg.Data());
    m_legend->AddEntry(m_sb_histo,"SB toy datasets");
    m_legend->AddEntry(m_b_histo,"B toy datasets");
    m_legend->AddEntry((TLine*)m_data_m2lnQ_line,"-2lnQ on Data","L");
    m_legend->SetFillColor(0);

    }

/*----------------------------------------------------------------------------*/

LimitPlot::~LimitPlot(){

    if (m_sb_histo)
        delete m_sb_histo;

    if (m_b_histo)
        delete m_b_histo;

    if (m_data_m2lnQ_line)
        delete m_data_m2lnQ_line;

   if (m_legend)
        delete m_legend;
    }


/*----------------------------------------------------------------------------*/

void LimitPlot::draw(const char* options){

    setCanvas(new TCanvas(GetName(),GetTitle()));
    getCanvas()->cd();
    getCanvas()->Draw(options);

    // We don't want the statistics of teh histos
    gStyle->SetOptStat(0);

    // The histos

    if (m_sb_histo->GetMaximum()>m_b_histo->GetMaximum()){
        m_sb_histo->DrawNormalized();
        m_b_histo->DrawNormalized("same");
        }
    else{
        m_b_histo->DrawNormalized();
        m_sb_histo->DrawNormalized("same");
        }

    // Shaded
    m_b_histo_shaded = (TH1F*)m_b_histo->Clone("b_shaded");
    m_b_histo_shaded->SetFillStyle(3005);
    m_b_histo_shaded->SetFillColor(kRed);

    m_sb_histo_shaded = (TH1F*)m_sb_histo->Clone("sb_shaded");
    m_sb_histo_shaded->SetFillStyle(3004);
    m_sb_histo_shaded->SetFillColor(kBlue);


    // Empty the bins according to the data -2lnQ
    double data_m2lnq= m_data_m2lnQ_line->GetX1();
    for (int i=0;i<m_sb_histo->GetNbinsX();++i){
        if (m_sb_histo->GetBinCenter(i)<data_m2lnq){
            m_sb_histo_shaded->SetBinContent(i,0);
            m_b_histo_shaded->SetBinContent(i,m_b_histo->GetBinContent(i)/m_b_histo->GetEntries());
            }
        else{
            m_b_histo_shaded->SetBinContent(i,0);
            m_sb_histo_shaded->SetBinContent(i,m_sb_histo->GetBinContent(i)/m_sb_histo->GetEntries());
            }
        }

    // Draw the shaded histos
    m_b_histo_shaded->Draw("same");
    m_sb_histo_shaded->Draw("same");

    // The line 
    m_data_m2lnQ_line->Draw("same");

    // The legend
    m_legend->Draw("same");


    }

/*----------------------------------------------------------------------------*/

void LimitPlot::print(const char* options){
    std::cout << "\nLimitPlot " << GetName() << std::endl;
    }

/*----------------------------------------------------------------------------*/

void LimitPlot::dumpToFile (const char* RootFileName, const char* options){

    TFile ofile(RootFileName,options);
    ofile.cd();

    // The histos
    m_sb_histo->Write();
    m_b_histo->Write();

    // The shaded histos
    if (m_b_histo_shaded!=NULL and m_sb_histo_shaded!=NULL){
        m_b_histo_shaded->Write();
        m_sb_histo_shaded->Write();
        }

    // The line 
    m_data_m2lnQ_line->Write("Measured -2lnQ line tag");

    // The legend
    m_legend->Write();

    ofile.Close();

    }

/*----------------------------------------------------------------------------*/
