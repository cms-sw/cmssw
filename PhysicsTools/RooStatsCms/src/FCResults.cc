// @(#)root/hist:$Id: FCResults.cc,v 1.1.1.1 2009/04/15 08:40:01 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"
#include <iostream>

#include "TChain.h"
#include "TAxis.h"

#include "PhysicsTools/RooStatsCms/interface/FCResults.h"


/*----------------------------------------------------------------------------*/

/**
Fill 2 histograms per point in the grid: one with R values calculated using 
the minimum value and the other using the measured value.
All the rootfiles are put in a chain.
**/

FCResults::FCResults(const char* name,
                     const char* title,
                     const char* rootfiles_wildcard):
    StatisticalMethod(name,title,true){

    // Build the Chain to collect data
    TChain FCdata("FCdata");
    FCdata.Add(rootfiles_wildcard);

    double measured_value;
    double studied_value;
    double minimum_value;
    double nll_measured_value;
    double nll_studied_value;
    double nll_minimum_value;

    FCdata.SetBranchAddress("measured_value",(&measured_value));
    FCdata.SetBranchAddress("studied_value",(&studied_value));
    FCdata.SetBranchAddress("minimum_value",(&minimum_value));
    FCdata.SetBranchAddress("nll_measured_value",(&nll_measured_value));
    FCdata.SetBranchAddress("nll_studied_value",(&nll_studied_value));
    FCdata.SetBranchAddress("nll_minimum_value",(&nll_minimum_value));

    // Set the addresses

    long int n_entries = FCdata.GetEntries();

    std::cout << "Results " << title << ": " << n_entries <<std::endl;

    // Loop over the MC data

    for (long int i=0;i<n_entries;++i){
        FCdata.GetEntry(i);

        /*
        If the studied point is not in the x grid, add it.
        Moreover allocate the two histograms
        */

        if (m_x_RminHisto_map.find(studied_value) == m_x_RminHisto_map.end()){
            std::cout << "Studied value: " << studied_value << std::endl;
            m_x_grid.push_back(studied_value);

            TString Rmin_histo_title=GetName();
            Rmin_histo_title+="_Rmin_point_";
            Rmin_histo_title+=studied_value;
            Rmin_histo_title.ReplaceAll(" ","");

            TH1F* Rmin_histo = new TH1F(Rmin_histo_title.Data(),
                                        Rmin_histo_title.Data(),
                                        500,
                                        0,
                                        10);

            Rmin_histo->GetXaxis()->SetTitle("Rmin");

            m_x_RminHisto_map[studied_value]=Rmin_histo;

            TString Rmeas_histo_title=GetName();
            Rmeas_histo_title+="_Rmeas_point_";
            Rmeas_histo_title+=studied_value;
            Rmeas_histo_title.ReplaceAll(" ","");

            TH1F* Rmeas_histo = new TH1F(Rmeas_histo_title.Data(),
                                         Rmeas_histo_title.Data(),
                                         1000,
                                         -20,
                                         20);

            Rmeas_histo->GetXaxis()->SetTitle("Rmeas");

            m_x_RmeasHisto_map[studied_value]=Rmeas_histo;

            } // end of the operations for a new point in the scan

        // Fill the histograms with the R values
        double min_val=nll_studied_value-nll_minimum_value;
        //std::cout << "Rmin Val = " << min_val << std::endl;
        m_x_RminHisto_map[studied_value]
                                    ->Fill(min_val);
        m_x_RmeasHisto_map[studied_value]
                                   ->Fill(nll_studied_value-nll_measured_value);
        } // End of the loop on the MC data

    }

/*----------------------------------------------------------------------------*/

FCResults::~FCResults(){

    for (unsigned int i=0;i<m_x_grid.size();++i){
        //delete m_Rmin_histos[i];
        //delete m_Rmeas_histos[i];
        }
    }

/*----------------------------------------------------------------------------*/

int FCResults::contains(std::vector<double>& vec, double val){
    //std::cout << "Value: " << val << std::endl;
    for (unsigned int i =0; i< vec.size();++i)
        if (fabs(val-vec[i]) < 0.0000001)
            return i;
    return -1;
    }

/*----------------------------------------------------------------------------*/

void FCResults::print(const char* options){
    std::cout << "Hello I am FC Results..\n";
    }

/*----------------------------------------------------------------------------*/
/**
Each point is calculated using the integral of the histogram.
The error is the square root of the entries.
**/
TGraphErrors* FCResults::getCLpoints(double CL){

    assert (CL<1);

    TString CLpoints_name=GetName();
    CLpoints_name+="_CL";
    CLpoints_name+=CL;
    CLpoints_name.ReplaceAll(" ","");

    TGraphErrors* CLpoints=new TGraphErrors(m_x_grid.size());
    CLpoints->SetName(CLpoints_name.Data());
    CLpoints->SetTitle(CLpoints_name.Data());

    CLpoints->SetMarkerStyle(4);
    CLpoints->SetMarkerSize(0.7);

    // Min points
    int i=0;
    std::map<double,TH1F*>::iterator iter;
    for( iter = m_x_RminHisto_map.begin();
         iter != m_x_RminHisto_map.end();
         iter++ ) {
        // Go through the integral to find the right percentage
        double* h_integral=iter->second->GetIntegral();
        double n_entries=iter->second->GetEntries();

        int bin_n=0;

        while (h_integral[bin_n]<CL)
            ++bin_n;

        double x = iter->first;
        double y = iter->second->GetBinCenter(bin_n);
        double ey = sqrt(h_integral[bin_n]*(1-h_integral[bin_n])/n_entries);
        //double ey = y*rel_stat_err;


        std::cout << "Filling --  (x,y) = (" << x << "," << y<< ")"<< std::endl;
        std::cout << "Filling --  (ex,ey) = (" << 0 << "," << ey<< ")"<< std::endl;
        CLpoints->SetPoint(i,x,y);
        CLpoints->SetPointError(i,0,ey);
        ++i;// index for graph points
        }

    return CLpoints;

    }


/*----------------------------------------------------------------------------*/

/**
Each point is calculated using the integral of the histogram.
The error is the square root of the entries.
**/
TGraphErrors* FCResults::getCLpointsMeasured(double CL){

    assert (CL<1);

    TString CLpoints_name=GetName();
    CLpoints_name+="_CL";
    CLpoints_name+=CL;
    CLpoints_name.ReplaceAll(" ","");

    TGraphErrors* CLpoints=new TGraphErrors(m_x_grid.size());
    CLpoints->SetName(CLpoints_name.Data());
    CLpoints->SetTitle(CLpoints_name.Data());

    CLpoints->SetMarkerStyle(26);
    CLpoints->SetMarkerSize(0.7);
    //CLpoints->SetMarkerColor(kBlue);

    // Min points
    int i=0;
    std::map<double,TH1F*>::iterator iter;
    for( iter = m_x_RmeasHisto_map.begin();
         iter != m_x_RmeasHisto_map.end();
         iter++ ) {
        // Go through the integral to find the right percentage
        double* h_integral=iter->second->GetIntegral();
        double n_entries=iter->second->GetEntries();

        int bin_n=0;

        while (h_integral[bin_n]<CL)
            ++bin_n;

        double x = iter->first;
        double y = iter->second->GetBinCenter(bin_n);
        double rel_stat_err = 1/sqrt(h_integral[bin_n]*n_entries);
        double ey = y*rel_stat_err;

        CLpoints->SetPoint(i,x,y);
        CLpoints->SetPointError(i,0,ey);
        ++i;// index for graph points
        }

    return CLpoints;

    }

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
