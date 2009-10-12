// HIC SUNT LEONES

#include <iostream>
#include <map>

#include "math.h"

#include "PhysicsTools/RooStatsCms/interface/Rsc.h"

#include "TF1.h"
#include "TAxis.h"
#include "TCanvas.h"

namespace Rsc
    {
    TRandom3 random_generator;

    /**
    Perform 2 times a gaussian fit to fetch the center of the histo.
    To get the second fit range get an interval that tries to keep into account 
    the skewness of the distribution.
    **/
    double getHistoCenter(TH1* histo_orig, double n_rms, bool display_result){

        TCanvas* c = new TCanvas();
        c->cd();

        TH1F* histo = (TH1F*)histo_orig->Clone();

        // get the histo x extremes
        double x_min = histo->GetXaxis()->GetXmin(); 
        double x_max = histo->GetXaxis()->GetXmax();

        // First fit!

        TF1* gaus = new TF1("mygaus", "gaus", x_min, x_max);

        gaus->SetParameter("Constant",histo->GetEntries());
        gaus->SetParameter("Mean",histo->GetMean());
        gaus->SetParameter("Sigma",histo->GetRMS());

        histo->Fit(gaus);

        // Second fit!
        double sigma = gaus->GetParameter("Sigma");
        double mean = gaus->GetParameter("Mean");

        delete gaus;

        std::cout << "Center is 1st pass = " << mean << std::endl;

        double skewness = histo->GetSkewness();

        x_min = mean - n_rms*sigma - sigma*skewness/2;
        x_max = mean + n_rms*sigma - sigma*skewness/2;;

        TF1* gaus2 = new TF1("mygaus2", "gaus", x_min, x_max);
        gaus2->SetParameter("Mean",mean);

        histo->Fit(gaus2,"L","", x_min, x_max);

        histo->Draw();
        gaus2->Draw("same");

        double center = gaus2->GetParameter("Mean");


        delete gaus2;
        delete histo;
        if (not display_result)
            delete c;

        return center;


    }


    /**
    We let an orizzontal bar go down and we stop when we have the integral 
    equal to the desired one.
    **/

    double* getHistoPvals (TH1F* histo_orig, double percentage){

        //TH1F* histo = new TH1F(*histo_orig);
        TH1F* histo=histo_orig;
        if (percentage>1){
            std::cerr << "Percentage greater or equal to 1!\n";
            return NULL;
            }

        double integral=histo->GetSumOfWeights();
        int initial_maxbin=histo->GetMaximumBin();
        double remaining_integral=integral;
        while (remaining_integral/integral > 1-percentage ){
            int maxbin = histo->GetMaximumBin();
            histo->SetBinContent(maxbin,0);
            remaining_integral=histo->GetSumOfWeights();
            }
        std::cout << "Remaining histogram of histo " << histo->GetName() 
                  << " " << remaining_integral/integral << " %\n";

        double* d = new double[2];
        d[0]=1;
        d[1]=1;

        // go through the histo to find the first non 0 bins:
        int current_bin=initial_maxbin;
        double bincontent=0;
        // right
        while(bincontent==0){
            current_bin++;
            bincontent=histo->GetBinContent(current_bin);
            }
        d[1]=histo->GetBinLowEdge(current_bin--);
        bincontent=0;
        // left
        while(bincontent==0){
            if (current_bin==0)
                break;
            current_bin--;
            bincontent=histo->GetBinContent(current_bin);
            }
        d[0]=histo->GetBinLowEdge(current_bin+1);

       //delete histo;

        return d;
        }

//----------------------------------------------------------------------------//
/**
Get the median of an histogram.
**/
    double getMedian(TH1* histo){

        Double_t* integral = histo->GetIntegral();
        int median_i = 0;
        for (int j=0;j<histo->GetNbinsX(); j++) 
            if (integral[j]<0.5) 
                median_i = j;
    
        double median_x = 
            histo->GetBinCenter(median_i)+
            (histo->GetBinCenter(median_i+1)-
            histo->GetBinCenter(median_i))*
            (0.5-integral[median_i])/(integral[median_i+1]-integral[median_i]);
    
        return median_x;
        }

    } // end of the Rsc Namespace
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
