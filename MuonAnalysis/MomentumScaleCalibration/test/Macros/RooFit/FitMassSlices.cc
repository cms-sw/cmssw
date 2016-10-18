#ifndef FitMassSlices_cc
#define FitMassSlices_cc

#include "FitSlices.cc"
#include "TFile.h"
#include "TH1F.h"
#include "TROOT.h"

/**
 * This class can be used to fit the X slices of a TH1 histogram using RooFit.
 * It uses the FitXslices class to do the fitting.
 */
class FitMassSlices : public FitSlices
{
public:

  FitMassSlices(
    double xMean_ = 3.1,
    double xMin_ = 3.,
    double xMax_ = 3.2,
    double sigma_ = 0.03,
    double sigmaMin_ = 0.,
    double sigmaMax_ = 0.1,
    TString signalType_ = "gaussian",
    TString backgroundType_ = "exponential"
    ) :
    FitSlices(xMean_, xMin_, xMax_, sigma_, sigmaMin_, sigmaMax_, signalType_, backgroundType_)
  {}

  void fit(
    const TString & inputFileName = "0_MuScleFit.root", const TString & outputFileName = "BiasCheck_0.root",
    
    // change 0 if you want to rebin phi distributions
    const int rebinXphi = 4, const int rebinXetadiff = 2, const int rebinXeta = 2, const int rebinXpt = 8,//(ORIGINAL 64 bin in eta, 64 bin in phi, 32 bin in deltaEta, 200 bin in pt)

    TDirectory * externalDir = 0,
    const TString & histoBaseName = "hRecBestResVSMu", const TString & histoBaseTitle = "MassVs"
    ){
    gROOT->SetBatch(kTRUE);

    TFile* inputFile = TFile::Open(inputFileName, "READ");
    TFile* outputFile = 0;
    TDirectory * dir = externalDir;
    if (dir == 0) {
      outputFile = new TFile(outputFileName, "RECREATE");
      dir = outputFile->GetDirectory("");
    }

    fitSlice(histoBaseName+"_MassVSPt", histoBaseTitle+"Pt",
      inputFile, dir);

    //     fitSlice(histoBaseName+"_MassVSEta", histoBaseTitle+"Eta",
    // 	     inputFile, dir);

    if (rebinXeta != 0) rebinX = rebinXeta;
    fitSlice(histoBaseName+"_MassVSEtaPlus", histoBaseTitle+"EtaPlus",
      inputFile, dir);

    fitSlice(histoBaseName+"_MassVSEtaMinus", histoBaseTitle+"EtaMinus",
      inputFile, dir);


    fitSlice(histoBaseName+"_MassVSEtaPhiPlus", histoBaseTitle+"EtaPhiPlus",
      inputFile, dir);

    fitSlice(histoBaseName+"_MassVSEtaPhiMinus", histoBaseTitle+"EtaPhiMinus",
      inputFile, dir);

    //    //    New entries...        
    fitSlice(histoBaseName+"_MassVSCosThetaCS", histoBaseTitle+"CosThetaCS",
      inputFile, dir);

    fitSlice(histoBaseName+"_MassVSPhiCS", histoBaseTitle+"PhiCS",
      inputFile, dir);

    if (rebinXphi != 0) rebinX = rebinXphi;
    fitSlice(histoBaseName+"_MassVSPhiPlus", histoBaseTitle+"PhiPlus",
      inputFile, dir);

    fitSlice(histoBaseName+"_MassVSPhiMinus", histoBaseTitle+"PhiMinus",
      inputFile, dir);


    //     fitSlice(histoBaseName+"_MassVSPhiPlusMinusDiff", histoBaseTitle+"PhiPlusMinusDiff",
    //              inputFile, dir);


    //     fitSlice(histoBaseName+"_MassVSPhiPlusPhiMinus", histoBaseTitle+"PhiPlusPhiMinus",
    //              inputFile, dir);

    //     fitSlice(histoBaseName+"_MassVSEtaPlusEtaMinus", histoBaseTitle+"EtaPlusEtaMinus",
    //              inputFile, dir);

    if (rebinXetadiff != 0) rebinX = rebinXetadiff;
    fitSlice(histoBaseName+"_MassVSEtaPlusMinusDiff", histoBaseTitle+"EtaPlusMinusDiff",
      inputFile, dir);

    // Close the output file
    if (outputFile != 0) {
      outputFile->Write();
      outputFile->Close();
    }
  }
};

#endif
