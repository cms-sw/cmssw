#include "TFile.h"
#include "TH2D.h"
#include "TH1D.h"
#include <iostream>
#include <sstream>

void ProbabilitySlice()
{
  TFile * file = new TFile("limitedSigma2.root", "READ");
  TH2D * hist;
  file->GetObject("GLh", hist);
  hist->SetTitle("J/Psi shape");

  TFile * outputFile = new TFile("Projections2.root", "RECREATE");
  outputFile->cd();
  int binsX = hist->GetNbinsX();
  for( int i = 1; i <= binsX; ++i ) {
    std::cout << "projecting bin " << i << std::endl;
    std::stringstream ss;
    ss << i;
    TH1D * projX = hist->ProjectionX((ss.str()+"_"+hist->GetName()).c_str(), i, i);
    projX->Write();
  }
  outputFile->Write();
  outputFile->Close();
}
