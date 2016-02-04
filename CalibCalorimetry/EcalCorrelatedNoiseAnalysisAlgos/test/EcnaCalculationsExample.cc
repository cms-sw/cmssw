//################## EcnaCalculationsExample.cc ####################
// B. Fabbro       08/04/2010
//
//

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRun.h"

#include "Riostream.h"
#include "TROOT.h"
#include "TRint.h"

#include <stdlib.h>
#include <string>
#include "TString.h"

using namespace std;

int main ( int argc, char **argv )
{
  //--------------------------------------------------------------------
  //                      Init
  //--------------------------------------------------------------------
  Int_t xCnew      = 0;
  Int_t xCdelete   = 0;

  TString    fTTBELL = "\007";

  //--------------------------------------------------------------------
  //                   TEcnaRun
  //--------------------------------------------------------------------
  Int_t   fKeyNbOfSamples =    10;      // Number of required samples

  TEcnaRun* MyRunEB = 0;  
  if ( MyRunEB == 0 ){MyRunEB = new TEcnaRun("EB", fKeyNbOfSamples);       xCnew++;}

  //.............. Declarations and default values

  TString fKeyAnaType     = "AdcPeg12";  // Analysis name for the Adc file
  TString fKeyStdType     = "StdPeg12";  // Analysis name for the Std (calculated) file
  Int_t   fKeyRunNumber   = 132440;      // Run number
  Int_t   fKeyFirstEvt    =      1;      // First Event number (to be analyzed)
  Int_t   fKeyLastEvt     =      0;      // Last Event number  (to be analyzed)
  Int_t   fKeyNbOfEvts    =    150;      // Number of events (events to be analyzed)
  Int_t   fKeySuMoNumber  =     18;      // Super-module number (EB)
  
  MyRunEB->GetReadyToReadData(fKeyAnaType.Data(),  fKeyRunNumber,
			      fKeyFirstEvt,        fKeyLastEvt,  fKeyNbOfEvts, fKeySuMoNumber);
  Bool_t ok_read = MyRunEB->ReadEventDistributions();
  
  if( ok_read == kTRUE )
    {
      MyRunEB->GetReadyToCompute();
      MyRunEB->SampleMeans();
      MyRunEB->SampleSigmas();
      MyRunEB->CorrelationsBetweenSamples();
      
      MyRunEB->Pedestals();
      MyRunEB->TotalNoise();
      MyRunEB->MeanOfCorrelationsBetweenSamples();
      MyRunEB->LowFrequencyNoise();
      MyRunEB->HighFrequencyNoise();
      MyRunEB->SigmaOfCorrelationsBetweenSamples();

      MyRunEB->AveragedPedestals();
      MyRunEB->AveragedTotalNoise();
      MyRunEB->AveragedMeanOfCorrelationsBetweenSamples();
      MyRunEB->AveragedLowFrequencyNoise();
      MyRunEB->AveragedHighFrequencyNoise();
      MyRunEB->AveragedSigmaOfCorrelationsBetweenSamples();
            
      //------- long time, big file
      //MyRunEB->LowFrequencyCorrelationsBetweenChannels();
      //MyRunEB->HighFrequencyCorrelationsBetweenChannels();
      //------- long time
      //MyRunEB->LowFrequencyMeanCorrelationsBetweenTowers();
      //MyRunEB->HighFrequencyMeanCorrelationsBetweenTowers();

      Bool_t ok_root_file = MyRunEB->WriteNewRootFile(fKeyStdType.Data());

      if( ok_root_file == kTRUE )
	{
	  cout << "*EcnaCalculationsExample> Write ROOT file OK" << endl;
	}
      else 
	{
	  cout << "!EcnaCalculationsExample> Writing ROOT file failure."
	       << fTTBELL << endl;
	}
    }
  else
    {
      cout << "!EcnaCalculationsExample> ROOT file not found."
	   << fTTBELL << endl;
    }
  //.......................................................................

  delete MyRunEB;                          xCdelete++;

      cout << "*H4Cna(main)> End of the example."  << endl;

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

  cout << "*EcnaCalculationsExample> Exiting main program." << endl;
  exit(0);
}
