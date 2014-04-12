// ROOT script to read the output produced by mps_parse_pedechi2hist.pl
// Author : Joerg Behr


// Usage:

// Start ROOT and compile the script:
// root [0] .L createChi2ndfplot.C+

// Call the plotting function and provide the name of inputfile produced by mps_parse_pedechi2hist.pl:
// createChi2ndfplot("chi2pedehis.txt");
//
// Or on the command line:
// root -l -x -b -q 'createChi2ndfplot.C+(\"chi2pedehis.txt\")'

#include <fstream>
#include <vector>
#include <utility>
#include <iostream>
#include <TROOT.h>
#include <TFile.h>
#include <TDirectory.h>
#include <TError.h>
#include <TH1.h>
#include <TH1D.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TPaveText.h>
#include <map>
#include <list>
#include <string>
#include <TStyle.h>
#include <TGaxis.h>
#include <TLegend.h>

class histdata
{
public:
  histdata(
           const std::string name2,
           const int number2,
           const double value2
           ) : name(name2),
               number(number2),
               value(value2)
  {
  }
  //needed for sort.
  bool operator < (const histdata& rhs)
  {
    return number < rhs.number;
  }
  
  const std::string name; //name of configuration to which the Mille binary belongs
  const int number; //The binary number (milleBinary334.dat)
  const double value; // <chi^2/ndf>
};

//Some configurations for the pad in which the plot is drawn.
void pad_cfg(
             TH1D *h
             )            
{
  const Int_t maxd = 2;
  const Short_t borderMode = 0;
  const Double_t leftMargin = 0.16;
  const Double_t rightMargin = 0.05;
  //const Double_t bottomMargin = 0.12;
  gPad->SetFrameBorderMode(0);

  gStyle->SetPalette(1,0);
  gPad->SetTickx(1);
  gPad->SetTicky(1);
  gPad->SetBorderMode(borderMode);
  gPad->SetLeftMargin(leftMargin);
  gPad->SetRightMargin(rightMargin);
  gPad->SetBottomMargin(0.15);
  gPad->SetTopMargin(0.07);
  TGaxis::SetMaxDigits(maxd);
  gStyle->SetTitleX(.0);
  gStyle->SetTitleW(.95);
  gStyle->SetTitleFontSize(.06);
  gStyle->SetTitleStyle(0);
  gStyle->SetTitleBorderSize(0);
  h->GetXaxis()->SetNoExponent();  
  h->GetYaxis()->SetNoExponent();
  h->GetXaxis()->SetLabelSize(0.05);
  h->GetXaxis()->SetTitleSize(0.07); //1.0
  h->GetXaxis()->SetTitleOffset(0.83); //1.0
  h->GetYaxis()->SetLabelSize(0.05);
  h->GetYaxis()->SetTitleOffset(0.74); //1.2 //0.7
  h->GetYaxis()->SetTitleSize(0.07);
  h->SetStats(kFALSE);
  h->GetYaxis()->SetNdivisions(505);
  h->GetXaxis()->SetNdivisions(505);

  //the margins
  gPad->SetLeftMargin(0.1162393);
  gPad->SetRightMargin(0.006837607);
  gPad->SetTopMargin(0.05347594);
  gPad->SetBottomMargin(0.1354724);



}

//This methods checks whether a histogram is existing, books a new histogram if nessesary, and fills the histogram.
void fillhisto(std::map<std::string,TH1D*> &histomap, const std::string name, const int value, const double wtx, const int nbins, const double lower, const double upper)
{
  std::map<std::string,TH1D*>::iterator it = histomap.find(name); //histogram existing?
  if(it != histomap.end())
    it->second->Fill(static_cast<double>(value),wtx);
  else
    {
      histomap[name] = new TH1D(TString(name),TString(name),nbins,lower, upper); //book new histogram
      histomap[name]->Fill(static_cast<double>(value),wtx);
    }
}

//the main (and only) plotting routine.
void createChi2ndfplot(const char *txtFile)
{
  std::ifstream theStream(txtFile, ios::in);
  if (!theStream.is_open()) {
    ::Error("createChi2ndfplot", "file %s could not be opened", txtFile);
  } else {
    //the collection of histograms.
    std::map<std::string,TH1D*> histomap; 
    
    //The data read from file txtFile
    std::list<histdata> h;

    //Variables needed for reading
    std::string name("");
    int number = 0;
    double value = 0.0;
  
    //Read the data
    while(
          theStream >> name
          >> number
          >> value
          )
      {
        h.push_back(histdata(name,number,value));
      }
    theStream.close();
    
    //If data was read
    if(h.size() > 0)
      {
        //Sort the list of data, because the output of mps_parse_pedechi2hist.pl can be arbitrary ordered.
        h.sort();

        double min = 10000.0, max = 0.0;
        int bincounter = 0; //The bin number (If binaries are missing in the pede job, then this quantity differs from the pure binary number)
        for(std::list<histdata>::const_iterator it = h.begin(); it != h.end(); it++)
          {
            bincounter++;
            
            //fill and book histogram
            fillhisto(histomap, it->name, bincounter, it->value, static_cast<int>(h.size()), 1.0, static_cast<double>(h.size()+1)); 
            
            //find minimal and maximal value of <chi2/ndf> used for axis limits.
            it->value > max ? max = it->value : 0;
            it->value < min ? min = it->value : 0;
          }
        
        TCanvas* c = new TCanvas("chi2ndfperbinary","chi2ndfperbinary", 200, 10, 900, 1200);
    
        gStyle->SetPadBorderMode(0);
        gStyle->SetOptStat(0);
        c->SetFillColor(10);
        c->Divide(1,2);
        c->cd(1);
    

        //haxis is only used for the axis.
        TH1D *haxis = new TH1D("haxis","; binary number; <#chi^{2}/ndf>", 1, 0.0, static_cast<double>(h.size()+1));
    
        //setup the pad.
        pad_cfg(haxis);
        

        haxis->SetMaximum(max*1.2);
        haxis->SetMinimum(min*0.8);
        
        haxis->DrawCopy("axis");
       
        //The legend which will be drawn in the second pad.
        TLegend *leg = new TLegend(0.01826713,0.003922913,0.9976767,0.9979814);
        leg->SetBorderSize(0);
        leg->SetFillColor(19);
        leg->SetFillStyle(0);

        //some usable colors
        const int ncolors = 11;
        const int kcolors[ncolors] = {kRed, kBlue, kBlack, kOrange, kGreen, kMagenta, kBlue+1, kBlue-1, kBlue+2, kBlue-2, kMagenta-2};
        
        int histocounter = 0;
        for( std::map<std::string,TH1D*>::iterator it = histomap.begin(); it != histomap.end(); it++)
          {
            int color = 1;
            //assign a color.
            if(histocounter < ncolors)
              {
                color = kcolors[histocounter];
                histocounter++;
              }
            else
              {
                color = histocounter;
                histocounter++;
              }

            //optical appearance
            it->second->SetMarkerColor(color);
            it->second->SetMarkerSize(1);
            it->second->SetMarkerStyle(20+histocounter);
            it->second->DrawCopy("p same");
            leg->AddEntry(it->second, TString(it->first), "p");
          }    

        
        
        //go the the second pad where the legend is depicted.
        c->cd(2);
        pad_cfg(haxis);
        leg->Draw();
        
        
        c->Print("chi2ndfperbinary.eps");
        c->Print("chi2ndfperbinary.C");
    

        //clean up
        delete c;
        delete leg;
        delete haxis;
        for( std::map<std::string,TH1D*>::iterator it = histomap.begin(); it != histomap.end(); it++)
          {
            delete it->second;
          }    
      }
  }
}
