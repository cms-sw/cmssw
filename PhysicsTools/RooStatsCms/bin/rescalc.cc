#include "PhysicsTools/RooStatsCms/interface/ResonanceCalculators.hh"

#include "TROOT.h"
#include "TFile.h"
#include "TH1D.h"

int main(int argc, char* argv[])
{
  // grab the histogram
  TFile *inputfile=TFile::Open("~/BumpDataInDiEle.root");
  inputfile->cd();
  TH1* hist=dynamic_cast<TH1*>(gROOT->FindObject("hHeepMassEBEBandEBEEdata"));

  // run the calculator
  int numPEs=100;
  int seed=1;
  FactoryResCalc rc("RooGaussian::signal(obs, signalmass, signalwidth)",
                    "EXPR::background('pow(1.0-obs/7000.0,p1)/pow(obs/7000.0,p2+p3*log(obs/7000.0))', p1[10,-40,40], p2[7,-20,20], p3[0.1,-20,20], obs)",
                    "prod::signalwidth(0.02, signalmass)");
  rc.setBinnedData(hist,21,200);
  rc.setNumBinsToDraw(hist->GetNbinsX());
  rc.setMinMaxSignalMass(180., 500.);
  rc.setNumPseudoExperiments(numPEs);
  rc.setRandomSeed(seed);
  runResCalc(rc, "output");

  return 0;
}
