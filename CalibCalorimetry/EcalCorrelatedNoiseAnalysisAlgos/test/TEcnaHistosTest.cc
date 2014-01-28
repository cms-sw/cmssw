//################## TEcnaHistosTest.cc ####################
// B. Fabbro       08/04/2010
//
//

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHistos.h"

#include "Riostream.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TGClient.h"
#include "TRint.h"
#include <cstdlib>

extern void InitGui();
VoidFuncPtr_t initfuncs[] = { InitGui, 0 };
TROOT root("GUI","GUI test environnement", initfuncs);

#include <stdlib.h>
#include <string>
#include "TSystem.h"
#include "TObject.h"

using namespace std;

int main ( int argc, char **argv )
{
  cout << "*EcalCorrelatedNoiseExampleHistos> Starting ROOT session" << endl;
  TRint theApp("App", &argc, argv);

  //--------------------------------------------------------------------
  //                      Init
  //--------------------------------------------------------------------
  Int_t xCnew      = 0;
  Int_t xCdelete   = 0;

  TString    fTTBELL = "\007";

  //.............. Default values


  TString    fKeyAnaType     = "StdPeg12";  // Analysis name
  Int_t      fKeyNbOfSamples =     10;      // Number of required samples
  Int_t      fKeyRunNumber   = 132440;      // Run number
  Int_t      fKeyFirstEvt    =      1;      // First required event number
  Int_t      fKeyLastEvt     =      0;      // Last required event number
  Int_t      fKeyNbOfEvts    =    150;      // Required number of events
  Int_t      fKeySuMoNumber  =     11;      // Super-module number (EB)
  Int_t      fKeyDeeNumber   =      1;      // Dee number (EE)

  Int_t SMtower = 1;
  Int_t TowEcha = 0;

  Int_t DeeSC   = 1;
  Int_t SCEcha  = 0;

  //=====================================================================
  //
  //                TEST TEcnaHistos for EB
  //
  //=====================================================================

  fKeyAnaType     = "StdPeg12"; 
  fKeyNbOfSamples =        10;  
  fKeyRunNumber   =    132440; 
  fKeyFirstEvt    =         1; 
  fKeyLastEvt     =         0;
  fKeyNbOfEvts    =       150;
  fKeySuMoNumber  =        32; 

  SMtower = 45;
  TowEcha = 11; 
  DeeSC   =  1;
  SCEcha  =  0;

  TEcnaHistos* MyHistosEB = 0;  
  if ( MyHistosEB == 0 ){MyHistosEB = new TEcnaHistos("EB");       xCnew++;}

#define TSEB
#ifdef TSEB

  //---------------------------------------------------
  
  MyHistosEB->SetHistoColorPalette(" ");

  fKeySuMoNumber = 0;
  MyHistosEB->FileParameters(fKeyAnaType,  fKeyNbOfSamples, fKeyRunNumber,
			     fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
  
  MyHistosEB->GeneralTitle("TEST TEcnaHistos, CMS ECAL EB");
  MyHistosEB->SetHistoScaleY("LIN");

  //............................. Pedestals
  MyHistosEB->SetHistoMin(0.); MyHistosEB->SetHistoMax();
  //MyHistosEB->EBXtalsAveragedPedestals();
  //MyHistosEB->EBEtaPhiAveragedPedestals();

  fKeySuMoNumber = 32;
  MyHistosEB->FileParameters(fKeyAnaType,  fKeyNbOfSamples, fKeyRunNumber,
			     fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
  //MyHistosEB->SMXtalsLowFrequencyNoise("ASCII");
  MyHistosEB->SMXtalsLowFrequencyNoise();
  //............................. Correlations between samples

  //MyHistosEB->CorrelationsBetweenSamples(SMtower, TowEcha, "ASCII");
  //MyHistosEB->SMXtalsPedestals("ASCII");
  //MyHistosEB->SMXtalsMeanOfCorss("ASCII");

  //............................. MeanOfSampleSigmasDistribution for 3 gains

  fKeyLastEvt   =   0;
  fKeySuMoNumber = 32;

  //.................................... EtaPhiSuperModuleMeanOfCorss (EB)
  //MyHistosEB->SMEtaPhiMeanOfCorss();

#endif // TSEB


#define GPLL
#ifdef GPLL

  //----------------------------------------------------- Alternate Global/Proj Lin/Log
  fKeyAnaType   =  "StdPeg12";
  MyHistosEB->FileParameters(fKeyAnaType, fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
  
  MyHistosEB->SetHistoMin(0.); MyHistosEB->SetHistoMax(2.5);
  MyHistosEB->SetHistoScaleY("LIN"); MyHistosEB->SMXtalsHighFrequencyNoise("SAME");

  MyHistosEB->SetHistoMin(0.); MyHistosEB->SetHistoMax(5.);
  MyHistosEB->SetHistoScaleY("LOG"); MyHistosEB->SMHighFrequencyNoiseXtals("SAME");

  fKeySuMoNumber++; 
  fKeyAnaType   =  "StdPeg12";
  MyHistosEB->FileParameters(fKeyAnaType, fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
  MyHistosEB->SetHistoScaleY("LIN"); MyHistosEB->SMXtalsHighFrequencyNoise("SAME");

  MyHistosEB->SetHistoScaleY("LOG"); MyHistosEB->SMHighFrequencyNoiseXtals("SAME");

  fKeySuMoNumber++; 
  fKeyAnaType   =  "StdPeg12";
  MyHistosEB->FileParameters(fKeyAnaType, fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
  MyHistosEB->SetHistoScaleY("LIN"); MyHistosEB->SMXtalsHighFrequencyNoise("SAME");
  MyHistosEB->SetHistoScaleY("LOG"); MyHistosEB->SMHighFrequencyNoiseXtals("SAME");

#endif // GPLL

  //--------------------------------------------------- Proj, Log
#define NOLO
#ifndef NOLO

  fKeySuMoNumber = 32;
  fKeyAnaType   =  "StdPeg12";
  MyHistosEB->FileParameters(fKeyAnaType, fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
  MyHistosEB->SetHistoMax(2.5);  MyHistosEB->SetHistoScaleY("LOG");
  MyHistosEB->SMHighFrequencyNoiseXtals("SAME");
  //.................................
  fKeySuMoNumber++;
  fKeyAnaType   =  "StdPeg12";
  MyHistosEB->FileParameters(fKeyAnaType, fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
  MyHistosEB->SMHighFrequencyNoiseXtals("SAME");
  //.................................
  fKeySuMoNumber++;
  fKeyAnaType   =  "StdPeg12";
  MyHistosEB->FileParameters(fKeyAnaType, fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);
  MyHistosEB->SMHighFrequencyNoiseXtals("SAME");

#endif // NOLO

  //--------------------------------------------------- tests option SAME n
  fKeySuMoNumber = 32;

#define SAMP
#ifndef SAMP

  fKeyAnaType   =  "StdPeg12";

  MyHistosEB->NewCanvas("SAME n");
  MyHistosEB->SetHistoScaleY("LIN");
  MyHistosEB->SetHistoMax(2.5);
  MyHistosEB->SMXtalsTotalNoise("SAME n");
  MyHistosEB->SMXtalsLowFrequencyNoise("SAME n");
  MyHistosEB->SMXtalsHighFrequencyNoise("SAME n");
  MyHistosEB->SMXtalsMeanOfCorss("SAME n");
  MyHistosEB->SMXtalsSigmaOfCorss("SAME n");


  MyHistosEB->NewCanvas("SAME n");
  MyHistosEB->SetHistoScaleY("LOG");
  MyHistosEB->SetHistoMax(5.);
  MyHistosEB->SMTotalNoiseXtals("SAME n");
  MyHistosEB->SMLowFrequencyNoiseXtals("SAME n");
  MyHistosEB->SMHighFrequencyNoiseXtals("SAME n");
  MyHistosEB->SMMeanOfCorssXtals("SAME n");
  MyHistosEB->SMSigmaOfCorssXtals("SAME n");

#endif // SAMP

#define SIGE
#ifdef SIGE

  MyHistosEB->SetHistoScaleY("LIN");

  TString run_par_file_name = "Ecna_132440_132524";
  MyHistosEB->SetHistoMin(0.15);
  MyHistosEB->XtalTimeMeanOfCorss(run_par_file_name, SMtower, TowEcha);

  MyHistosEB->NewCanvas("SAME n");
  MyHistosEB->SetHistoMin(0.);  MyHistosEB->SetHistoMax(2.5);
  MyHistosEB->XtalTimeTotalNoise(run_par_file_name, SMtower, TowEcha, "SAME n");
  MyHistosEB->XtalTimeLowFrequencyNoise(run_par_file_name,  SMtower, TowEcha, "SAME n");
  MyHistosEB->XtalTimeHighFrequencyNoise(run_par_file_name, SMtower, TowEcha, "SAME n");


  //............................................................

  MyHistosEB->SetHistoScaleY("LIN");

  fKeyAnaType = "StdPeg12";

  fKeyRunNumber = 0;
  MyHistosEB->FileParameters(fKeyAnaType,   fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, 
			     fKeySuMoNumber);

  //............................................................History Low Frequency Noise
  MyHistosEB->SetHistoColorPalette(" ");
  MyHistosEB->SetHistoScaleY("LIN");
  SMtower = 45;
  TowEcha = 12;


  MyHistosEB->FileParameters("StdPeg12", fKeyNbOfSamples, 0, 1, 0, 150, fKeySuMoNumber);
  run_par_file_name = "Ecna_132440_132524";

  MyHistosEB->SetHistoMin(0.);   MyHistosEB->SetHistoMax(2.5);
  MyHistosEB->XtalTimeLowFrequencyNoise(run_par_file_name, SMtower, TowEcha, "SAME");

  MyHistosEB->FileParameters("StdPeg12", fKeyNbOfSamples, 0,  1, 0, 150, fKeySuMoNumber);
  MyHistosEB->XtalTimeLowFrequencyNoise(run_par_file_name, SMtower, TowEcha+1, "SAME");

  MyHistosEB->FileParameters("StdPeg12", fKeyNbOfSamples, 0,  1, 0, 150, fKeySuMoNumber);
  MyHistosEB->XtalTimeLowFrequencyNoise(run_par_file_name, SMtower, TowEcha+2, "SAME");
#endif // SIGE

  //=====================================================================
  //
  //                TEST TEcnaHistos for EE
  //
  //=====================================================================
  
  fKeyAnaType     = "StdPeg12"; 
  fKeyNbOfSamples =        10;  
  fKeyRunNumber   =    132440; 
  fKeyFirstEvt    =         1; 
  fKeyLastEvt     =         0;
  fKeyDeeNumber   =         3;  

  TEcnaHistos* MyHistosEE = 0;  
  if ( MyHistosEE == 0 ){MyHistosEE = new TEcnaHistos("EE");       xCnew++;}

  MyHistosEE->SetHistoColorPalette("rainbow");

  //---------------------------------------------------

#define SIGM
#ifndef SIGM

  fKeyDeeNumber = 0;
  MyHistosEE->FileParameters(fKeyAnaType,   fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts,
			     fKeyDeeNumber);

  MyHistosEE->GeneralTitle("TEST TEcnaHistos, CMS ECAL EE");
  MyHistosEE->SetHistoScaleY("LIN");

  //............................. Pedestals
  //MyHistosEE->SetHistoMin(0.); MyHistosEE->SetHistoMax();
  //MyHistosEE->EEXtalsAveragedPedestals();

  MyHistosEE->EEIXIYAveragedPedestals();

  fKeyDeeNumber = 2;
  MyHistosEE->FileParameters(fKeyAnaType,   fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts,
			     fKeyDeeNumber);

  DeeSC   = 116;

  SCEcha = 10;
  MyHistosEE->XtalSamplesSigma(DeeSC, SCEcha, "SAME");
  SCEcha = 11;
  MyHistosEE->XtalSamplesSigma(DeeSC, SCEcha, "SAME");
  SCEcha = 12;
  MyHistosEE->XtalSamplesSigma(DeeSC, SCEcha, "SAME");
  SCEcha = 13;
  MyHistosEE->XtalSamplesSigma(DeeSC, SCEcha, "SAME");
  SCEcha = 14;
  MyHistosEE->XtalSamplesSigma(DeeSC, SCEcha, "SAME");
  SCEcha = 15;
  MyHistosEE->XtalSamplesSigma(DeeSC, SCEcha, "SAME");
  SCEcha = 16;
  MyHistosEE->XtalSamplesSigma(DeeSC, SCEcha, "SAME");

  MyHistosEE->CorrelationsBetweenSamples(DeeSC, SCEcha, "SURF1Z");

#endif // SIGM


#define SIGC
#ifndef SIGC

  MyHistosEE->SetHistoScaleY("LIN");

  MyHistosEE->FileParameters(fKeyAnaType,   fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts,
			     fKeyDeeNumber);

  MyHistosEE->GeneralTitle("TEST TEcnaHistos, CMS ECAL EE");

  //..................................... crystal numbering
  DeeSC  = 114;
  MyHistosEE->SCCrystalNumbering(fKeyDeeNumber, DeeSC);

  //............................. Correlations between samples
  MyHistosEE->CorrelationsBetweenSamples(DeeSC);
  SCEcha = 17;
  MyHistosEE->CorrelationsBetweenSamples(DeeSC, SCEcha, "LEGO2Z");
  MyHistosEE->CorrelationsBetweenSamples(DeeSC, SCEcha, "ASCII");

  SCEcha = 18;
  MyHistosEE->CorrelationsBetweenSamples(DeeSC, SCEcha, "SURF1Z");
  SCEcha = 19;
  MyHistosEE->CorrelationsBetweenSamples(DeeSC, SCEcha, "SURF4Z");
#endif // SIGC 


  //..................................... stability (EE)
#define SIGS
#ifndef SIGS
  //............................................................

  MyHistosEE->SetHistoScaleY("LIN");

  fKeyRunNumber = 0;
  MyHistosEE->FileParameters(fKeyAnaType,   fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts,
			     fKeyDeeNumber);

  //............................................................ Low Frequency Noise (EE)
  MyHistosEE->SetHistoColorPalette(" ");
  MyHistosEE->SetHistoScaleY("LIN");
  DeeSC = 114;
  SCEcha = 10;

  MyHistosEE->FileParameters("StdPeg12", fKeyNbOfSamples, 0, 1, 0, 150, fKeyDeeNumber);
  run_par_file_name = "Ecna_132440_132524";

  MyHistosEE->SetHistoMin(0.);   MyHistosEE->SetHistoMax(2.5);
  MyHistosEE->XtalTimeLowFrequencyNoise(run_par_file_name, DeeSC, SCEcha, "SAME");

  MyHistosEE->FileParameters("StdPeg12", fKeyNbOfSamples, 0,  1, 0, 150, fKeyDeeNumber);
  MyHistosEE->XtalTimeLowFrequencyNoise(run_par_file_name, DeeSC, SCEcha+1, "SAME");

  MyHistosEE->FileParameters("StdPeg12", fKeyNbOfSamples, 0,  1, 0, 150, fKeyDeeNumber);
  MyHistosEE->XtalTimeLowFrequencyNoise(run_par_file_name, DeeSC, SCEcha+2, "SAME");
#endif // SIGS


#define SIGZ
#ifndef SIGZ
  //.................................... IXIYDeeMeanOfCorss (EE)
  MyHistosEE->FileParameters(fKeyAnaType,   fKeyNbOfSamples,
			     fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts,
			     fKeyDeeNumber);

  MyHistosEE->SetHistoScaleY("LIN");

  //MyHistosEE->DeeIXIYMeanOfCorss();
  //MyHistosEE->DeeIXIYPedestals();
  MyHistosEE->DeeIXIYTotalNoise();

  //..................................... Mean of Pedestal Global and Proj (EE)
  //  MyHistosEE->SetHistoScaleY("LOG"); 
  //  MyHistosEE->DeePedestalsXtals();



  MyHistosEE->SetHistoMin(0.); MyHistosEE->SetHistoMax();
  MyHistosEE->DeeXtalsPedestals("ASCII");
  MyHistosEE->DeeXtalsPedestals();

  //.................................... sample sigmas histo
  //MyHistosEE->SetHistoMax();
  //MyHistosEE->DeeXtalsMeanOfCorss();
  //MyHistosEE->DeeXtalsMeanOfCorss("ASCII");
  //MyHistosEE->DeeXtalsSigmaOfCorss();
#endif // SIGZ

  //=====================================================================
  //
  //                            END OF  TEST 
  //
  //=====================================================================

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
       << "*EcalCorrelatedNoiseExampleHistos> Terminating ROOT session." << endl;
  theApp.Terminate(0);
  cout << "*EcalCorrelatedNoiseExampleHistos> Exiting main program." << endl;
  exit(0);
}
