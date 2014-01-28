//################## EcnaHistosExample.cc ####################
// B. Fabbro       17/03/2010
//
//

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHistos.h"

#include "Riostream.h"
#include "TROOT.h"
#include "TRint.h"

//extern void InitGui();
//VoidFuncPtr_t initfuncs[] = { InitGui, 0 };
//TROOT root("GUI","GUI test environnement", initfuncs);

#include <stdlib.h>
#include <string>
#include "TString.h"

using namespace std;

int main ( int argc, char **argv )
{
  cout << "*EcnaHistosExample> Starting ROOT session" << endl;
  TRint theApp("App", &argc, argv);

  //--------------------------------------------------------------------
  //                      Init
  //--------------------------------------------------------------------
  Int_t xCnew      = 0;
  Int_t xCdelete   = 0;

  TString fTTBELL = "\007";

  //--------------------------------------------------------------------
  //                   view histos
  //--------------------------------------------------------------------

  TEcnaHistos* MyHistosEB = 0;  
  if ( MyHistosEB == 0 ){MyHistosEB = new TEcnaHistos("EB");       xCnew++;}
 
  TEcnaHistos* MyHistosEE = 0;  
  if ( MyHistosEE == 0 ){MyHistosEE = new TEcnaHistos("EE");       xCnew++;}

  //.............. Declarations and default values

  Int_t SMtower = 1;
  Int_t TowEcha = 0;

  TString    fKeyAnaType     = "StdPeg12";     // Analysis name
  Int_t      fKeyNbOfSamples =        10;      // Number of required samples
  Int_t      fKeyRunNumber   =    132440;      // Run number
  Int_t      fKeyFirstEvt    =         1;      // First Event number (to be analyzed)
  Int_t      fKeyLastEvt     =         0;      // Last Event number (to be analyzed)
  Int_t      fKeyNbOfEvts    =       150;      // Number of required events (events to be analyzed)
  Int_t      fKeySuMoNumber  =        11;      // Super-module number (EB)
  Int_t      fKeyDeeNumber   =         3;      // Dee number (EE)

  //================================== Plots

  fKeyAnaType    = "StdPeg12";
  fKeyRunNumber  = 132440;  
  fKeyFirstEvt   =      1;
  fKeyLastEvt    =      0;
  fKeyNbOfEvts   =    150;
  fKeySuMoNumber =     11;
  MyHistosEB->FileParameters(fKeyAnaType.Data(), fKeyNbOfSamples, fKeyRunNumber,
			     fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);

  MyHistosEB->SetHistoColorPalette(" ");

  MyHistosEB->GeneralTitle("TB2006, CMS ECAL EB");
  MyHistosEB->SetHistoScaleY("LIN");

  //............................. Pedestals
  MyHistosEB->SetHistoMin(0.); MyHistosEB->SetHistoMax();
  MyHistosEB->SMXtalsPedestals("ASCII");
  MyHistosEB->SMXtalsPedestals();

#define FPLO
#ifndef FPLO
  //............................. Correlations between samples
  SMtower = 21;
  TowEcha = 11;
  MyHistosEB->CorrelationsBetweenSamples(SMtower, TowEcha, "ASCII");
  MyHistosEB->SMXtalsPedestals("ASCII");
  MyHistosEB->SMXtalsMeanOfCorss("ASCII");
#endif // FPLO

  //............................. MeanOfSampleSigmasDistribution for 3 gains
  fKeyAnaType    = "StdPeg12";
  fKeyRunNumber  = 132440;
  fKeyLastEvt    =      0;
  fKeySuMoNumber =     11;

  //.................................... EtaPhiSuperModuleMeanOfCorss (EB)
  MyHistosEB->SMEtaPhiMeanOfCorss();

#define NOLO
#ifdef NOLO
  //--------------------------------------------------- Proj, Log
  fKeyFirstEvt   = 1;
  fKeySuMoNumber = 3;
  MyHistosEB->FileParameters("StdPeg12", fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts ,fKeySuMoNumber);
  MyHistosEB->SetHistoMax(2.5);  MyHistosEB->SetHistoScaleY("LOG");
  MyHistosEB->SMLowFrequencyNoiseXtals("SAME");
  //.................................
  fKeySuMoNumber = 4;
  MyHistosEB->FileParameters("StdPeg12", fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
  MyHistosEB->SMLowFrequencyNoiseXtals("SAME");
  //.................................
  fKeySuMoNumber = 5;
  MyHistosEB->FileParameters("StdPeg12", fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
  MyHistosEB->SMLowFrequencyNoiseXtals("SAME");

  MyHistosEB->SetHistoMax(2.5);
  MyHistosEB->SMHighFrequencyNoiseXtals();
#endif // NOLO 

#define SAMP
#ifdef SAMP
  //--------------------------------------------------- tests option SAME n

  MyHistosEB->NewCanvas("SAME n");
  MyHistosEB->SetHistoScaleY("LIN");
  MyHistosEB->SetHistoMax(2.5);
  MyHistosEB->SMXtalsTotalNoise("SAME n");
  MyHistosEB->SMXtalsLowFrequencyNoise("SAME n");
  MyHistosEB->SMXtalsHighFrequencyNoise("SAME n");

  MyHistosEB->NewCanvas("SAME n");
  MyHistosEB->SetHistoScaleY("LOG");
  MyHistosEB->SetHistoMax(2.5);
  MyHistosEB->SMTotalNoiseXtals("SAME n");
  MyHistosEB->SMLowFrequencyNoiseXtals("SAME n");
  MyHistosEB->SMHighFrequencyNoiseXtals("SAME n");
#endif // SAMP

  MyHistosEB->SetHistoScaleY("LIN");

#define SIGM
#ifndef SIGM
  //............................. MeanOfSampleSigmasDistribution for gain 12
  //MyHistosEB->SetHistoMax();
  //MyHistosEB->SMMeanOfSampleSigmasXtals();

  //.................................. MeanOfCorssDistribution for 7 crystals
  fKeyAnaType    = "StdPeg12";
  fKeyRunNumber  = 132440;
  fKeyFirstEvt   =      1;  
  fKeyLastEvt   =       0;
  fKeySuMoNumber =     17;
  MyHistosEB->FileParameters(fKeyAnaType.Data(),   fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts,
			     fKeySuMoNumber);

  MyHistosEB->SetHistoScaleY("LIN"); 

  SMtower = 38;

  MyHistosEB->SetHistoColorPalette("rainbow");

  TowEcha = 10;
  MyHistosEB->XtalSamplesSigma(SMtower, TowEcha, "SAME");
  TowEcha = 11;
  MyHistosEB->XtalSamplesSigma(SMtower, TowEcha, "SAME");
  TowEcha = 12;
  MyHistosEB->XtalSamplesSigma(SMtower, TowEcha, "SAME");
  TowEcha = 13;
  MyHistosEB->XtalSamplesSigma(SMtower, TowEcha, "SAME");
  TowEcha = 14;
  MyHistosEB->XtalSamplesSigma(SMtower, TowEcha, "SAME");
  TowEcha = 15;
  MyHistosEB->XtalSamplesSigma(SMtower, TowEcha, "SAME");
  TowEcha = 16;
  MyHistosEB->XtalSamplesSigma(SMtower, TowEcha, "SAME");
#endif // SIGM

  //.......................................................................
  fKeyAnaType    = "StdPeg12";
  fKeyRunNumber  = 132440;
  fKeyFirstEvt   =      1;  
  fKeyLastEvt   =       0;
  fKeyDeeNumber  =      3;
  MyHistosEE->SetHistoScaleY("LIN");

  MyHistosEE->FileParameters(fKeyAnaType.Data(),   fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts,
			     fKeyDeeNumber);
  MyHistosEE->GeneralTitle("TB2007, CMS ECAL EE");

#define SIGC
#ifdef SIGC

  Int_t DeeSC = 245;
  Int_t SCEcha = 18;

  //..................................... Dee & crystal numbering
  MyHistosEE->DeeSCNumbering(fKeyDeeNumber);
  MyHistosEE->SCCrystalNumbering(fKeyDeeNumber, DeeSC);

  //............................. Correlations between samples
  MyHistosEE->CorrelationsBetweenSamples(DeeSC);
  SCEcha = 17;
  MyHistosEE->CorrelationsBetweenSamples(DeeSC, SCEcha, "LEGO2Z");
  MyHistosEE->CorrelationsBetweenSamples(DeeSC, SCEcha, "ASCII");

  SCEcha = 18;
  MyHistosEE->CorrelationsBetweenSamples(DeeSC, SCEcha, "SURF1Z");
  //SCEcha = 19;
  //MyHistosEE->CorrelationsBetweenSamples(DeeSC, SCEcha, "SURF1Z");
#endif // SIGC 

  //..................................... stability

  //............................................................ correlations and noises (EB)
  SMtower = 28;
  TowEcha = 12;
  MyHistosEB->SetHistoScaleY("LIN");
  fKeyAnaType  = "StdPeg12";
  fKeyFirstEvt =     1;  
  fKeyLastEvt  =     0;
  fKeySuMoNumber =  17;
  MyHistosEB->SetHistoColorPalette(" ");

  MyHistosEB->FileParameters(fKeyAnaType.Data(), fKeyNbOfSamples, 0,
			     fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);

  TString run_par_file_name = "Ecna_132440_132524";

  MyHistosEB->SetHistoMin(0.15);  MyHistosEB->SetHistoMax(0.8);
  MyHistosEB->XtalTimeMeanOfCorss(run_par_file_name, SMtower, TowEcha);
  MyHistosEB->SetHistoMin(-0.2);  MyHistosEB->SetHistoMax(0.75);
  MyHistosEB->XtalMeanOfCorssRuns(run_par_file_name, SMtower, TowEcha);

#define SIGS
#ifdef SIGS
  MyHistosEB->NewCanvas("SAME n");
  MyHistosEB->SetHistoMin(0.);  MyHistosEB->SetHistoMax(2.5);
  MyHistosEB->XtalTimeTotalNoise(run_par_file_name, SMtower, TowEcha, "SAME n");
  MyHistosEB->XtalTimeLowFrequencyNoise(run_par_file_name, SMtower, TowEcha, "SAME n");
  MyHistosEB->XtalTimeHighFrequencyNoise(run_par_file_name, SMtower, TowEcha, "SAME n");
  MyHistosEB->XtalTimeMeanOfCorss(run_par_file_name, SMtower, TowEcha, "SAME n");
  MyHistosEB->XtalTimeSigmaOfCorss(run_par_file_name, SMtower, TowEcha, "SAME n");

  //.................................... IXIYDeeMeanOfCorss (EE)
  //MyHistosEE->DeeIXIYMeanOfCorss();

  //..................................... Mean of Pedestal Global and Proj (EE)
  //  MyHistosEE->SetHistoScaleY("LOG"); 
  //  MyHistosEE->DeePedestalsXtals();

  MyHistosEE->SetHistoScaleY("LIN");
#endif // SIGS

#define SIGZ
#ifndef SIGZ
  // MyHistosEE->SetHistoMin(100.);
  MyHistosEE->DeeXtalsPedestals("ASCII");

  //.................................... sample sigmas histo
  MyHistosEE->SetHistoMax();
  MyHistosEE->DeeXtalsMeanOfCorss("ASCII");
#endif // SIGZ

  //MyHistosEE->DeeXtalsMeanOfCorss();

  //.......................................................................

  delete MyHistosEB;                          xCdelete++;
  delete MyHistosEE;                          xCdelete++;

      cout << "*H4Cna(main)> End of the example. You can quit ROOT (.q)"  << endl;

  if ( xCnew != xCdelete )
    {
      cout << "!H4Cna(main)> WRONG MANAGEMENT OF ALLOCATIONS: xCnew = "
	   << xCnew << ", xCdelete = " << xCdelete << '\007' << endl;
    }
  else
    {
      //  cout << "*H4Cna(main)> BRAVO! GOOD MANAGEMENT OF ALLOCATIONS: xCnew = "
      //      << xCnew << ", xCdelete = " << xCdelete << endl;
    }

  Bool_t retVal = kTRUE;
  theApp.Run(retVal);
  cout << endl
       << "*EcnaHistosExample> Terminating ROOT session." << endl;
  theApp.Terminate(0);
  cout << "*EcnaHistosExample> Exiting main program." << endl;
  exit(0);
}
