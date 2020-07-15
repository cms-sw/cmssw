//################## EcnaHistosExample2.cc ####################
// B. Fabbro      04/07/2011
//
//   Drawing Histos with and without TEcnaRead and calls to ViewHisto(...)

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHistos.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRead.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"

#include "Riostream.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TGClient.h"
#include "TRint.h"
#include <cstdlib>

//extern void InitGui();
//VoidFuncPtr_t initfuncs[] = { InitGui, 0 };
//TROOT root("GUI","GUI test environnement", initfuncs);

#include "TObject.h"
#include "TSystem.h"
#include <cstdlib>
#include <string>

using namespace std;

int main(int argc, char** argv) {
  //--------------------------------------------------------------------
  //                      Init
  //--------------------------------------------------------------------

  std::cout << "*EcalCorrelatedNoiseExampleHistos> Starting ROOT session" << std::endl;
  TRint theApp("App", &argc, argv);

  TString fTTBELL = "\007";

  //.............. Default values
  TString fKeyAnaType = "StdPeg12";  // Analysis name
  Int_t fKeyNbOfSamples = 10;        // Number of required samples
  Int_t fKeyRunNumber = 136098;      // Run number
  Int_t fKeyFirstEvt = 1;            // First required event number
  Int_t fKeyLastEvt = 0;             // Last required event number
  Int_t fKeyNbOfEvts = 150;          // Required number of events
  Int_t fKeySuMoNumber = 11;         // Super-module number (EB)
  Int_t fKeyDeeNumber = 1;           // Dee number (EE)

  Int_t xAlreadyRead = 1;

  //========================================================================

  TEcnaObject* myTEcnaManagerEB = new TEcnaObject();
  TEcnaParPaths* fEcnaParPathsEB = new TEcnaParPaths(myTEcnaManagerEB);

  TEcnaObject* myTEcnaManagerEE = new TEcnaObject();
  TEcnaParPaths* fEcnaParPathsEE = new TEcnaParPaths(myTEcnaManagerEE);

  if (fEcnaParPathsEB->GetPaths() == kTRUE && fEcnaParPathsEE->GetPaths() == kTRUE) {
    //=====================================================================
    //
    //                TEST TEcnaRead and TEcnaHistos for EB
    //
    //=====================================================================
    Int_t SMtower = 1;
    Int_t TowEcha = 0;
    Int_t n1Sample = 4;

    //  Int_t DeeSC   = 1;
    //  Int_t SCEcha  = 0;

    fKeyAnaType = "StdPeg12";
    fKeyNbOfSamples = 10;
    fKeyRunNumber = 136098;
    fKeyFirstEvt = 1;
    fKeyLastEvt = 0;
    fKeyNbOfEvts = 150;
    fKeySuMoNumber = 32;

    TEcnaParEcal* fEcalParEB = new TEcnaParEcal(myTEcnaManagerEB, "EB");
    TEcnaRead* fMyRootFileEB = new TEcnaRead(myTEcnaManagerEB, "EB");

    TEcnaHistos* MyHistosEB = new TEcnaHistos(myTEcnaManagerEB, "EB");

#define HGLO
#ifdef HGLO
    //==================================================== Test Plot1DHisto SM
    fMyRootFileEB->PrintNoComment();
    fMyRootFileEB->FileParameters(fKeyAnaType,
                                  fKeyNbOfSamples,
                                  fKeyRunNumber,
                                  fKeyFirstEvt,
                                  fKeyLastEvt,
                                  fKeyNbOfEvts,
                                  fKeySuMoNumber,
                                  fEcnaParPathsEB->ResultsRootFilePath().Data());

    if (fMyRootFileEB->LookAtRootFile() == kTRUE) {
      MyHistosEB->FileParameters(fMyRootFileEB);
      MyHistosEB->GeneralTitle("EcnaHistosExample2");
      TVectorD read_h_tno(fEcalParEB->MaxCrysInSM());
      TVectorD read_h_lfn(fEcalParEB->MaxCrysInSM());
      TVectorD read_h_hfn(fEcalParEB->MaxCrysInSM());

      read_h_tno = fMyRootFileEB->Read1DHisto(fEcalParEB->MaxCrysInSM(), "TotalNoise", "SuperModule");
      read_h_lfn = fMyRootFileEB->Read1DHisto(fEcalParEB->MaxCrysInSM(), "lfn", "SM");
      read_h_hfn = fMyRootFileEB->Read1DHisto(fEcalParEB->MaxCrysInSM(), "high frequency noise", "SM");

      MyHistosEB->FileParameters(fMyRootFileEB);

      MyHistosEB->SetHistoMax(3.5);
      MyHistosEB->SetHistoScaleY("LOG");
      MyHistosEB->NewCanvas("SAME n");
      MyHistosEB->Plot1DHisto(read_h_tno, "TNo", "NbOfXtals", "SM", "SAME n");
      MyHistosEB->Plot1DHisto(read_h_lfn, "LFN", "NOX", "SM", "SAME n");
      MyHistosEB->Plot1DHisto(read_h_hfn, "HFN", "NOX", "SM", "SAME n");
    } else {
      std::cout << "!EcnaHistosExample2> *ERROR* =====> "
                << " ROOT file not found" << fTTBELL << std::endl;
    }

#endif  //  HGLO

    MyHistosEB->SetHistoScaleY("LIN");

#define HBAE
#ifdef HBAE
    fKeyAnaType = "AdcPeg12";
    fMyRootFileEB->FileParameters(fKeyAnaType,
                                  fKeyNbOfSamples,
                                  fKeyRunNumber,
                                  fKeyFirstEvt,
                                  fKeyLastEvt,
                                  fKeyNbOfEvts,
                                  fKeySuMoNumber,
                                  fEcnaParPathsEB->ResultsRootFilePath().Data());

    if (fMyRootFileEB->LookAtRootFile() == kTRUE) {
      TVectorD read_histo(fKeyNbOfEvts);
      read_histo = fMyRootFileEB->Read1DHisto(fKeyNbOfEvts, "AdcValue", SMtower, TowEcha, n1Sample);
      MyHistosEB->FileParameters(fMyRootFileEB);
      MyHistosEB->SetHistoMin(125.);
      MyHistosEB->Plot1DHisto(read_histo, "Event", "ADC", SMtower, TowEcha, n1Sample);
      MyHistosEB->Plot1DHisto(read_histo, "Adc", "NbOfEvts", SMtower, TowEcha, n1Sample);
    } else {
      std::cout << "!EcnaHistosExample2> *ERROR* =====> "
                << " ROOT file not found" << fTTBELL << std::endl;
    }
#endif  // HBAE

#define PMCC
#ifdef PMCC
    fKeyAnaType = "SccPeg12";
    fKeyRunNumber = 136098;
    fKeySuMoNumber = 1;
    fMyRootFileEB->FileParameters(fKeyAnaType,
                                  fKeyNbOfSamples,
                                  fKeyRunNumber,
                                  fKeyFirstEvt,
                                  fKeyLastEvt,
                                  fKeyNbOfEvts,
                                  fKeySuMoNumber,
                                  fEcnaParPathsEB->ResultsRootFilePath().Data());

    if (fMyRootFileEB->LookAtRootFile() == kTRUE) {
      TMatrixD read_matrix(fEcalParEB->MaxTowInSM(), fEcalParEB->MaxTowInSM());
      read_matrix = fMyRootFileEB->ReadMatrix(fEcalParEB->MaxTowInSM(), "Cor", "HFBetweenTowers");
      MyHistosEB->FileParameters(fMyRootFileEB);
      MyHistosEB->PlotMatrix(read_matrix, "Cor", "HFBetweenTowers");
    } else {
      std::cout << "!EcnaHistosExample2> *ERROR* =====> "
                << " ROOT file not found" << fTTBELL << std::endl;
    }
#endif  // PMCC

    fKeyAnaType = "StdPeg12";

#define HBAS
#ifdef HBAS

    fKeyRunNumber = 136098;
    //==================================================== Test Plot1DHisto samples
    fMyRootFileEB->FileParameters(fKeyAnaType,
                                  fKeyNbOfSamples,
                                  fKeyRunNumber,
                                  fKeyFirstEvt,
                                  fKeyLastEvt,
                                  fKeyNbOfEvts,
                                  fKeySuMoNumber,
                                  fEcnaParPathsEB->ResultsRootFilePath().Data());

    if (fMyRootFileEB->LookAtRootFile() == kTRUE) {
      TVectorD read_histo_2(fEcalParEB->MaxCrysInTow() * fEcalParEB->MaxSampADC());
      TVectorD read_histo_samps(fEcalParEB->MaxSampADC());

      MyHistosEB->FileParameters(fMyRootFileEB);

      std::cout << "EcnahistosExample2> *************** Plot1DHisto 1 crystal ONLYONE  ******************" << std::endl;
      TowEcha = 12;
      read_histo_2 =
          fMyRootFileEB->Read1DHisto(fEcalParEB->MaxCrysInTow() * fEcalParEB->MaxSampADC(), "SigmaOfSamples", SMtower);

      std::cout << "*EcnaHistosExample2> channel " << std::setw(2) << TowEcha << ": ";
      for (Int_t i0_samp = 0; i0_samp < fEcalParEB->MaxSampADC(); i0_samp++) {
        read_histo_samps(i0_samp) = read_histo_2(TowEcha * fEcalParEB->MaxSampADC() + i0_samp);
        std::cout << std::setprecision(4) << std::setw(8) << read_histo_samps(i0_samp) << ", ";
      }
      std::cout << std::endl;

      MyHistosEB->Plot1DHisto(read_histo_samps, "Sample", "SSp", SMtower, TowEcha);

      std::cout << "EcnahistosExample2> ***************** Plot1DHisto 1 crystal ONLYONE  ****************" << std::endl;
      MyHistosEB->Plot1DHisto("Sample", "SSp", SMtower, TowEcha);

      std::cout << "EcnahistosExample2> ******************** All Xtals Plot1DHisto **********************" << std::endl;
      SMtower = 59;
      read_histo_2 =
          fMyRootFileEB->Read1DHisto(fEcalParEB->MaxCrysInTow() * fEcalParEB->MaxSampADC(), "SampleMean", SMtower);

      // => BUG SCRAM? (pas vu a la compilation si methode Plot1DHisto absente avec ces arguments; plantage a l'execution
      MyHistosEB->SetHistoMin(180.);
      MyHistosEB->SetHistoMax(220.);
      MyHistosEB->Plot1DHisto(read_histo_2, "Sample#", "SampleMean", SMtower);

      std::cout << "EcnahistosExample2> ****************** Plot1DHisto 1 crystal SAME *******************" << std::endl;

      TowEcha = 13;
      std::cout << "*EcnaHistosExample2> channel " << std::setw(2) << TowEcha << ": ";
      for (Int_t i0_samp = 0; i0_samp < fEcalParEB->MaxSampADC(); i0_samp++) {
        read_histo_samps(i0_samp) = read_histo_2(TowEcha * fEcalParEB->MaxSampADC() + i0_samp);
        std::cout << std::setprecision(4) << std::setw(8) << read_histo_samps(i0_samp) << ", ";
      }
      std::cout << std::endl;
      MyHistosEB->SetHistoMin(180.);
      MyHistosEB->SetHistoMax(220.);
      MyHistosEB->Plot1DHisto(read_histo_samps, "Sample", "SampleMean", SMtower, TowEcha, "SAME");

      TowEcha = 14;
      std::cout << "*EcnaHistosExample2> channel " << std::setw(2) << TowEcha << ": ";
      for (Int_t i0_samp = 0; i0_samp < fEcalParEB->MaxSampADC(); i0_samp++) {
        read_histo_samps(i0_samp) = read_histo_2(TowEcha * fEcalParEB->MaxSampADC() + i0_samp);
        std::cout << std::setprecision(4) << std::setw(8) << read_histo_samps(i0_samp) << ", ";
      }
      std::cout << std::endl;
      MyHistosEB->Plot1DHisto(read_histo_samps, "SampleNumber", "SampleMean", SMtower, TowEcha, "SAME");

      TowEcha = 15;
      std::cout << "*EcnaHistosExample2> channel " << std::setw(2) << TowEcha << ": ";
      for (Int_t i0_samp = 0; i0_samp < fEcalParEB->MaxSampADC(); i0_samp++) {
        read_histo_samps(i0_samp) = read_histo_2(TowEcha * fEcalParEB->MaxSampADC() + i0_samp);
        std::cout << std::setprecision(4) << std::setw(8) << read_histo_samps(i0_samp) << ", ";
      }
      std::cout << std::endl;
      MyHistosEB->Plot1DHisto(read_histo_samps, "Sample#", "SampleMean", SMtower, TowEcha, "SAME");

      std::cout << "EcnahistosExample2> ******************** Plot1DHisto 1 crystal ONLYONE **************" << std::endl;
      MyHistosEB->SetHistoMin(180.);
      MyHistosEB->SetHistoMax(220.);
      MyHistosEB->Plot1DHisto(read_histo_samps, "Sample", "SampMean", SMtower, TowEcha);
    } else {
      std::cout << "!EcnaHistosExample2> *ERROR* =====> "
                << " ROOT file not found" << fTTBELL << std::endl;
    }

#endif  // NODR

#define VMAT
#ifdef VMAT
    //==================================================== Test PlotMatrix, ViewMatrix

    SMtower = 22;
    TowEcha = 15;

    xAlreadyRead = 1;
    if (xAlreadyRead == 1) {
      fMyRootFileEB->FileParameters(fKeyAnaType,
                                    fKeyNbOfSamples,
                                    fKeyRunNumber,
                                    fKeyFirstEvt,
                                    fKeyLastEvt,
                                    fKeyNbOfEvts,
                                    fKeySuMoNumber,
                                    fEcnaParPathsEB->ResultsRootFilePath().Data());

      if (fMyRootFileEB->LookAtRootFile() == kTRUE) {
        TMatrixD read_matrix_sample(fEcalParEB->MaxSampADC(), fEcalParEB->MaxSampADC());
        read_matrix_sample = fMyRootFileEB->ReadMatrix(fEcalParEB->MaxSampADC(), "cor", "samples", SMtower, TowEcha);

        MyHistosEB->FileParameters(fMyRootFileEB);

        MyHistosEB->PlotMatrix(read_matrix_sample, "Cor", "Mss", SMtower, TowEcha);
        MyHistosEB->PlotMatrix(read_matrix_sample, "Cor", "Mss", SMtower, TowEcha, "SURF1Z");
      } else {
        std::cout << "!EcnaHistosExample2> *ERROR* =====> "
                  << " ROOT file not found" << fTTBELL << std::endl;
      }
    }

    SMtower = 51;
    TowEcha = 9;

    xAlreadyRead = 0;
    if (xAlreadyRead == 0) {
      MyHistosEB->FileParameters(
          fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
      MyHistosEB->PlotMatrix("Cor", "Mss", SMtower, TowEcha, "SURF4");
      MyHistosEB->PlotMatrix("Cor", "Mss", SMtower, TowEcha, "LEGO2Z");
    }
#endif  // VMAT

#define VSTX
#ifdef VSTX
    //==================================================== Test PlotDetector, ViewStex

    xAlreadyRead = 1;
    if (xAlreadyRead == 1) {
      fKeySuMoNumber = 18;
      fMyRootFileEB->FileParameters(fKeyAnaType,
                                    fKeyNbOfSamples,
                                    fKeyRunNumber,
                                    fKeyFirstEvt,
                                    fKeyLastEvt,
                                    fKeyNbOfEvts,
                                    fKeySuMoNumber,
                                    fEcnaParPathsEB->ResultsRootFilePath().Data());

      if (fMyRootFileEB->LookAtRootFile() == kTRUE) {
        MyHistosEB->FileParameters(fMyRootFileEB);
        MyHistosEB->PlotDetector(fMyRootFileEB->ReadLowFrequencyNoise(fEcalParEB->MaxCrysInSM()), "LFN", "SM");
      } else {
        std::cout << "!EcnaHistosExample2> *ERROR* =====> "
                  << " ROOT file not found" << fTTBELL << std::endl;
      }
    }

    xAlreadyRead = 0;
    if (xAlreadyRead == 0) {
      fKeySuMoNumber = 18;
      MyHistosEB->FileParameters(
          fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);

      MyHistosEB->PlotDetector("HFN", "SM");
    }
#endif  // VSTX

#define VSTS
#ifdef VSTS
    //==================================================== Test PlotDetector, ViewStas

    xAlreadyRead = 1;
    if (xAlreadyRead == 1) {
      fMyRootFileEB->FileParameters(fKeyAnaType,
                                    fKeyNbOfSamples,
                                    fKeyRunNumber,
                                    fKeyFirstEvt,
                                    fKeyLastEvt,
                                    fKeyNbOfEvts,
                                    fKeySuMoNumber,
                                    fEcnaParPathsEB->ResultsRootFilePath().Data());

      if (fMyRootFileEB->LookAtRootFile() == kTRUE) {
        TVectorD read_eb_histo(fEcalParEB->MaxTowInEB());
        MyHistosEB->FileParameters(fMyRootFileEB);

        MyHistosEB->SetHistoMax(3.5);
        MyHistosEB->SetHistoScaleY("LIN");
        MyHistosEB->NewCanvas("SAME n");
        read_eb_histo = fMyRootFileEB->Read1DHisto(fEcalParEB->MaxTowInEB(), "TNo", "EB");
        MyHistosEB->Plot1DHisto(read_eb_histo, "Tow", "TNo", "EB", "SAME n");
        read_eb_histo = fMyRootFileEB->Read1DHisto(fEcalParEB->MaxTowInEB(), "LFN", "EB");
        MyHistosEB->Plot1DHisto(read_eb_histo, "Tow", "LFN", "EB", "SAME n");
        read_eb_histo = fMyRootFileEB->Read1DHisto(fEcalParEB->MaxTowInEB(), "HFN", "EB");
        MyHistosEB->Plot1DHisto(read_eb_histo, "Tow", "HFN", "EB", "SAME n");
      } else {
        std::cout << "!EcnaHistosExample2> *ERROR* =====> "
                  << " ROOT file not found" << fTTBELL << std::endl;
      }
    }

    xAlreadyRead = 0;
    if (xAlreadyRead == 0) {
      MyHistosEB->FileParameters(
          fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);

      MyHistosEB->Plot1DHisto("Tow", "TNo", "EB");
    }
#endif  // VSTS

    //=====================================================================
    //
    //                TEST TEcnaHistos for EE
    //
    //=====================================================================

    TEcnaParEcal* fEcalParEE = new TEcnaParEcal(myTEcnaManagerEE, "EE");
    TEcnaRead* fMyRootFileEE = new TEcnaRead(myTEcnaManagerEE, "EE");

    TEcnaHistos* MyHistosEE = new TEcnaHistos(myTEcnaManagerEE, "EE");

    fKeyDeeNumber = 1;
    fKeyRunNumber = 161311;

    xAlreadyRead = 1;
    if (xAlreadyRead == 1) {
      fMyRootFileEE->FileParameters(fKeyAnaType,
                                    fKeyNbOfSamples,
                                    fKeyRunNumber,
                                    fKeyFirstEvt,
                                    fKeyLastEvt,
                                    fKeyNbOfEvts,
                                    fKeyDeeNumber,
                                    fEcnaParPathsEE->ResultsRootFilePath().Data());

      if (fMyRootFileEE->LookAtRootFile() == kTRUE) {
        TVectorD read_ee_histo(fEcalParEE->MaxSCInEE());
        MyHistosEE->FileParameters(fMyRootFileEE);

        MyHistosEE->SetHistoMax(3.5);
        MyHistosEE->SetHistoScaleY("LIN");
#define HSAA
#ifdef HSAA
        MyHistosEE->NewCanvas("SAME n");
        read_ee_histo = fMyRootFileEE->Read1DHisto(fEcalParEE->MaxSCInEE(), "TNo", "EE");
        MyHistosEE->Plot1DHisto(read_ee_histo, "SC", "TNo", "EE", "SAME n");
        read_ee_histo = fMyRootFileEE->Read1DHisto(fEcalParEE->MaxSCInEE(), "LFN", "EE");
        MyHistosEE->Plot1DHisto(read_ee_histo, "SC", "LFN", "EE", "SAME n");
        read_ee_histo = fMyRootFileEE->Read1DHisto(fEcalParEE->MaxSCInEE(), "HFN", "EE");
        MyHistosEE->Plot1DHisto(read_ee_histo, "SC", "HFN", "EE", "SAME n");
#endif  // HSAA
        read_ee_histo = fMyRootFileEE->Read1DHisto(fEcalParEE->MaxSCInEE(), "TNo", "EE");
        MyHistosEE->StartStopDate(fMyRootFileEE->GetStartDate(), fMyRootFileEE->GetStopDate());
        MyHistosEE->Plot1DHisto(read_ee_histo, "SC", "TNo", "EE");
        MyHistosEE->PlotDetector(read_ee_histo, "TotalNoise", "EE");
      } else {
        std::cout << "!EcnaHistosExample2> *ERROR* =====> "
                  << " ROOT file not found" << fTTBELL << std::endl;
      }
    }

    xAlreadyRead = 0;
    if (xAlreadyRead == 0) {
      MyHistosEE->FileParameters(
          fKeyAnaType, fKeyNbOfSamples, fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeyDeeNumber);

      MyHistosEE->SetHistoMax(3.5);
      MyHistosEE->SetHistoScaleY("LIN");

#define HSAB
#ifdef HSAB
      MyHistosEE->NewCanvas("SAME n");
      MyHistosEE->Plot1DHisto("SC", "TNo", "EE", "SAME n");
      MyHistosEE->Plot1DHisto("SC", "LFN", "EE", "SAME n");
      MyHistosEE->Plot1DHisto("SC", "HFN", "EE", "SAME n");
#endif  // HSAB

      MyHistosEE->Plot1DHisto("SC", "TNo", "EE");
      MyHistosEE->PlotDetector("TotalNoise", "EE");
    }

    //=====================================================================
    //
    //                            END OF  TEST
    //
    //=====================================================================

    std::cout << "*H4Cna(main)> End of the example. You can quit ROOT (.q)" << std::endl;

    Bool_t retVal = kTRUE;
    theApp.Run(retVal);
    std::cout << std::endl << "*EcalCorrelatedNoiseExampleHistos> Terminating ROOT session." << std::endl;
    theApp.Terminate(0);
    std::cout << "*EcalCorrelatedNoiseExampleHistos> Exiting main program." << std::endl;
    exit(0);

    delete MyHistosEB;  // always after exit(0) (because of TEcnaHistos::DoCanvasClosed())
    delete MyHistosEE;  // always after exit(0) (because of TEcnaHistos::DoCanvasClosed())
  }
}
