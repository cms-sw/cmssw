//################## EcnaHistosExample1.cc ####################
// B. Fabbro      09/08/2012
//
//   Drawing histos with TEcnaHistos
//   with and without direct calls to TEcnaRead methods

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHistos.h"

#include "Riostream.h"
#include "TROOT.h"
#include "TRint.h"

#include "TString.h"
#include <cstdlib>
#include <string>

using namespace std;

int main(int argc, char** argv) {
  TEcnaObject* myTEcnaManagerEB = new TEcnaObject();
  TEcnaObject* myTEcnaManagerEE = new TEcnaObject();
  TEcnaParPaths* fEcnaParPathsEB = new TEcnaParPaths(myTEcnaManagerEB);
  TEcnaParPaths* fEcnaParPathsEE = new TEcnaParPaths(myTEcnaManagerEE);

  if (fEcnaParPathsEB->GetPaths() == kTRUE && fEcnaParPathsEE->GetPaths() == kTRUE) {
    std::cout << "*EcnaHistosExample> Starting ROOT session" << std::endl;
    TRint theApp("App", &argc, argv);

    //--------------------------------------------------------------------
    //                      Init
    //--------------------------------------------------------------------
    Int_t xCnew = 0;
    Int_t xCdelete = 0;

    TString fTTBELL = "\007";

    //--------------------------------------------------------------------
    //                   view histos
    //--------------------------------------------------------------------
    //TEcnaParEcal* fEcalParEB = new TEcnaParEcal(myTEcnaManagerEB, "EB");
    TEcnaParEcal* fEcalParEE = new TEcnaParEcal(myTEcnaManagerEE, "EE");

    //TEcnaRead*   fMyRootFileEB = new TEcnaRead(myTEcnaManagerEB, "EB");
    TEcnaRead* fMyRootFileEE = new TEcnaRead(myTEcnaManagerEE, "EE");

    TEcnaHistos* MyHistosEB = new TEcnaHistos(myTEcnaManagerEB, "EB");
    xCnew++;
    TEcnaHistos* MyHistosEE = new TEcnaHistos(myTEcnaManagerEE, "EE");
    xCnew++;

    //.............. Declarations and default values
    TString fKeyAnaType = "StdPeg12";  // Analysis name
    Int_t fKeyNbOfSamples = 10;        // Number of required samples
    Int_t fKeyRunNumber = 136098;      // Run number
    Int_t fKeyFirstEvt = 1;            // First Event number (to be analyzed)
    Int_t fKeyLastEvt = 0;             // Last Event number (to be analyzed)
    Int_t fKeyNbOfEvts = 150;          // Number of required events (events to be analyzed)
    Int_t fKeySuMoNumber = 11;         // Super-module number (EB)
    Int_t fKeyDeeNumber = 3;           // Dee number (EE)

    Int_t SMtower = 1;
    Int_t TowEcha = 1;

    //================================== Plots

    fKeyAnaType = "StdPeg12";
    fKeyRunNumber = 136098;
    fKeyFirstEvt = 1;
    fKeyLastEvt = 0;
    fKeyNbOfEvts = 150;
    fKeySuMoNumber = 11;
    MyHistosEB->FileParameters(
        fKeyAnaType.Data(), fKeyNbOfSamples, fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);

    MyHistosEB->SetHistoColorPalette(" ");

    MyHistosEB->GeneralTitle("EcnaHistosExample1");
    MyHistosEB->SetHistoScaleY("LIN");

    //............................. Pedestals
    MyHistosEB->SetHistoMin(0.);
    MyHistosEB->SetHistoMax();
    MyHistosEB->Plot1DHisto("Crystal#", "Ped", "SM");

    //.............................
    fKeyAnaType = "StdPeg12";
    fKeyRunNumber = 136098;
    fKeyLastEvt = 0;
    fKeySuMoNumber = 11;

    //.................................... EtaPhiSuperModuleMeanCorss (EB)
    MyHistosEB->PlotDetector("MCs", "SM");

#define FPLO
#ifdef FPLO
    //............................. Correlations between samples
    SMtower = 21;
    TowEcha = 12;
    MyHistosEB->PlotMatrix("Cor", "Mss", SMtower, TowEcha, "LEGO2Z");
#endif  // FPLO

#define NOLO
#ifdef NOLO
    //--------------------------------------------------- Proj, Log
    fKeyFirstEvt = 1;
    fKeySuMoNumber = 3;
    MyHistosEB->FileParameters(
        "StdPeg12", fKeyNbOfSamples, fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
    MyHistosEB->SetHistoMax(2.5);
    MyHistosEB->SetHistoScaleY("LOG");
    //MyHistosEB->Plot1DHisto("LowFrequencyNoise", "NbOfXtals", "SM", "SAME");

    std::cout << "*EcnaHistosExample1> *** TEST OF WRONG CODE (BEGINNING)."
              << " MESSAGE: < code not found > (and lists after) ARE THERE ON PURPOSE." << std::endl
              << std::endl;

    MyHistosEB->Plot1DHisto("LowFrequencyNoise", "NbOfXtal", "SM", "SAME");
    std::cout << std::endl
              << "*EcnaHistosExample1> *** TEST OF WRONG CODE (END)."
              << " MESSAGE: < code not found > (and lists after) WERE THERE ON PURPOSE." << std::endl;

    //.................................
    fKeySuMoNumber = 4;
    MyHistosEB->FileParameters(
        "StdPeg12", fKeyNbOfSamples, fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
    MyHistosEB->Plot1DHisto("LowFrequencyNoise", "NbOfXtals", "SM", "SAME");
    //.................................
    fKeySuMoNumber = 5;
    MyHistosEB->FileParameters(
        "StdPeg12", fKeyNbOfSamples, fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
    MyHistosEB->Plot1DHisto("LowFrequencyNoise", "NbOfXtals", "SM", "SAME");

    MyHistosEB->SetHistoMax(2.5);
    MyHistosEB->Plot1DHisto("HighFrequencyNoise", "NbOfXtals", "SM");

    MyHistosEB->SetHistoScaleY("LIN");
#endif  // NOLO

#define SAMP
#ifdef SAMP
    //--------------------------------------------------- tests option SAME n

    MyHistosEB->NewCanvas("SAME n");
    MyHistosEB->SetHistoScaleY("LIN");
    MyHistosEB->SetHistoMax(2.5);
    MyHistosEB->Plot1DHisto("Xtal", "TNo", "SM", "SAME n");
    MyHistosEB->Plot1DHisto("Xtal", "LFN", "SM", "SAME n");
    MyHistosEB->Plot1DHisto("Xtal", "HFN", "SM", "SAME n");

    MyHistosEB->NewCanvas("SAME n");
    MyHistosEB->SetHistoScaleY("LOG");
    MyHistosEB->SetHistoMax(2.5);
    MyHistosEB->Plot1DHisto("TNo", "NbOfXtals", "SM", "SAME n");
    MyHistosEB->Plot1DHisto("LFN", "NbOfXtals", "SM", "SAME n");
    MyHistosEB->Plot1DHisto("HFN", "NbOfXtals", "SM", "SAME n");

    MyHistosEB->SetHistoScaleY("LIN");

#endif  // SAMP

#define SIGM
#ifdef SIGM
    //.................................. Sample sigma Distribution for 7 crystals
    fKeyAnaType = "StdPeg12";
    fKeyRunNumber = 136098;
    fKeyFirstEvt = 1;
    fKeyLastEvt = 0;
    fKeySuMoNumber = 17;
    MyHistosEB->FileParameters(
        fKeyAnaType.Data(), fKeyNbOfSamples, fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);

    MyHistosEB->SetHistoScaleY("LIN");

    SMtower = 38;

    MyHistosEB->SetHistoColorPalette("rainbow");

    TowEcha = 10;
    MyHistosEB->Plot1DHisto("SampleSigma", "NOS", SMtower, TowEcha, "SAME");
    TowEcha = 11;
    MyHistosEB->Plot1DHisto("SampleSigma", "NOS", SMtower, TowEcha, "SAME");
    TowEcha = 12;
    MyHistosEB->Plot1DHisto("SampleSigma", "NOS", SMtower, TowEcha, "SAME");
    TowEcha = 13;
    MyHistosEB->Plot1DHisto("SampleSigma", "NOS", SMtower, TowEcha, "SAME");
    TowEcha = 14;
    MyHistosEB->Plot1DHisto("SampleSigma", "NOS", SMtower, TowEcha, "SAME");
    TowEcha = 15;
    MyHistosEB->Plot1DHisto("SampleSigma", "NOS", SMtower, TowEcha, "SAME");
    TowEcha = 16;
    MyHistosEB->Plot1DHisto("SampleSigma", "NOS", SMtower, TowEcha, "SAME");
#endif  // SIGM

#define SIGC
#ifdef SIGC

    Int_t DeeSC = 245;
    //..................................... Dee & crystal numbering
    MyHistosEE->SCCrystalNumbering(fKeyDeeNumber, DeeSC);
    MyHistosEE->DeeSCNumbering(fKeyDeeNumber);

#endif  // SIGC

    //..................................... stability

#define SIGS
#ifdef SIGS
    //........................................... History Plots
    SMtower = 28;
    TowEcha = 12;
    MyHistosEB->SetHistoScaleY("LIN");
    fKeyAnaType = "StdPeg12";
    fKeyFirstEvt = 1;
    fKeyLastEvt = 0;
    fKeySuMoNumber = 17;
    MyHistosEB->SetHistoColorPalette(" ");

    MyHistosEB->FileParameters(
        fKeyAnaType.Data(), fKeyNbOfSamples, 0, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);

    TString run_par_file_name = "Ecna_132440_137033";

    MyHistosEB->SetHistoMin(0.15);
    MyHistosEB->SetHistoMax(0.8);
    MyHistosEB->PlotHistory("Time", "MCs", run_par_file_name, SMtower, TowEcha);

    MyHistosEB->SetHistoMin(-0.2);
    MyHistosEB->SetHistoMax(0.75);
    MyHistosEB->PlotHistory("MCs", "NOR", run_par_file_name, SMtower, TowEcha);

    MyHistosEB->NewCanvas("SAME n");
    MyHistosEB->SetHistoMin(0.);
    MyHistosEB->SetHistoMax(2.5);
    MyHistosEB->PlotHistory("Time", "TotalNoise", run_par_file_name, SMtower, TowEcha, "SAME n");
    MyHistosEB->PlotHistory("Time", "LFN", run_par_file_name, SMtower, TowEcha, "SAME n");
    MyHistosEB->PlotHistory("Time", "HFN", run_par_file_name, SMtower, TowEcha, "SAME n");
    MyHistosEB->PlotHistory("Time", "MeanCorss", run_par_file_name, SMtower, TowEcha, "SAME n");
    MyHistosEB->PlotHistory("Time", "SigCorss", run_par_file_name, SMtower, TowEcha, "SAME n");
#endif  // SIGS

#define SIGE
#ifdef SIGE
    //............................................ EE plots
    fKeyAnaType = "StdPeg12";
    fKeyRunNumber = 136098;
    fKeyFirstEvt = 1;
    fKeyLastEvt = 0;
    fKeyDeeNumber = 2;

    MyHistosEE->SetHistoScaleY("LIN");
    MyHistosEE->GeneralTitle("EcnaHistosExample1");

    fMyRootFileEE->PrintNoComment();
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
      read_ee_histo = fMyRootFileEE->Read1DHisto(fEcalParEE->MaxSCInEE(), "total noise", "EE");

      MyHistosEE->FileParameters(fMyRootFileEE);
      MyHistosEE->SetHistoMax(3.5);
      MyHistosEE->SetHistoScaleY("LIN");
      MyHistosEE->PlotDetector(read_ee_histo, "TotalNoise", "EE");
      MyHistosEE->Plot1DHisto(read_ee_histo, "SC", "TotalNoise", "EE");
    }

    MyHistosEE->FileParameters(
        fKeyAnaType.Data(), fKeyNbOfSamples, fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeyDeeNumber);
    MyHistosEE->SetHistoScaleY("LIN");
    MyHistosEE->SetHistoMin();
    MyHistosEE->SetHistoMax();
    MyHistosEE->PlotDetector("TNo", "Dee");

#endif  // SIGE
    //.......................................................................

    std::cout << "*EcnaHistosExample1> End of the example. You can quit ROOT (.q)" << std::endl;

    Bool_t retVal = kTRUE;
    theApp.Run(retVal);
    std::cout << std::endl << "*EcnaHistosExample> Terminating ROOT session." << std::endl;
    theApp.Terminate(0);
    std::cout << "*EcnaHistosExample> Exiting main program." << std::endl;
    exit(0);

    delete MyHistosEB;
    xCdelete++;
    delete MyHistosEE;
    xCdelete++;

    if (xCnew != xCdelete) {
      std::cout << "!EcnaHistosExample1> WRONG MANAGEMENT OF ALLOCATIONS: xCnew = " << xCnew
                << ", xCdelete = " << xCdelete << '\007' << std::endl;
    } else {
      //  std::cout << "*EcnaHistosExample1> BRAVO! GOOD MANAGEMENT OF ALLOCATIONS: xCnew = "
      //      << xCnew << ", xCdelete = " << xCdelete << std::endl;
    }
  }
}
