//################## EcnaCalculationsExample.cc ####################
// B. Fabbro       21/10/2010
//
//

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRun.h"

#include "Riostream.h"
#include "TROOT.h"
#include "TRint.h"

#include "TString.h"
#include <cstdlib>
#include <string>

using namespace std;

int main(int argc, char** argv) {
  //--------------------------------------------------------------------
  //                      Init
  //--------------------------------------------------------------------
  Int_t xCnew = 0;
  Int_t xCdelete = 0;

  TString fTTBELL = "\007";

  //--------------------------------------------------------------------
  //                   TEcnaRun
  //--------------------------------------------------------------------
  Int_t fKeyNbOfSamples = 10;  // Number of required samples

  TEcnaObject* myTEcnaManager = new TEcnaObject();

  std::cout << "!EcnaCalculationsExample> CONTROLE 1" << std::endl;

  TEcnaRun* MyRunEB = new TEcnaRun(myTEcnaManager, "EB", fKeyNbOfSamples);
  xCnew++;

  std::cout << "!EcnaCalculationsExample> CONTROLE 2" << std::endl;

  //.............. Declarations and default values

  TString fKeyAnaType = "AdcPeg12";  // Analysis name for the Adc file
  TString fKeyStdType = "StdPeg12";  // Analysis name for the Std (calculated) file
  Int_t fKeyRunNumber = 136098;      // Run number
  Int_t fKeyFirstEvt = 1;            // First Event number (to be analyzed)
  Int_t fKeyLastEvt = 0;             // Last Event number  (to be analyzed)
  Int_t fKeyNbOfEvts = 150;          // Number of events (events to be analyzed)
  Int_t fKeySuMoNumber = 18;         // Super-module number (EB)

  MyRunEB->GetReadyToReadData(
      fKeyAnaType.Data(), fKeyRunNumber, fKeyFirstEvt, fKeyLastEvt, fKeyNbOfEvts, fKeySuMoNumber);

  Bool_t ok_read = MyRunEB->ReadSampleAdcValues();

  if (ok_read == kTRUE) {
    MyRunEB->GetReadyToCompute();

    //------- Standard calculations
    MyRunEB->StandardCalculations();

    //------- Expert 1 calculations long time, big file
    //MyRunEB->Expert1Calculations();

    //------- Expert 2 calculations long time
    //MyRunEB->Expert2Calculations();

    Bool_t ok_root_file = MyRunEB->WriteNewRootFile(fKeyStdType.Data());

    if (ok_root_file == kTRUE) {
      std::cout << "*EcnaCalculationsExample> Write ROOT file OK" << std::endl;
    } else {
      std::cout << "!EcnaCalculationsExample> Writing ROOT file failure." << fTTBELL << std::endl;
    }
  } else {
    std::cout << "!EcnaCalculationsExample> ROOT file not found." << fTTBELL << std::endl;
  }
  //.......................................................................

  delete MyRunEB;
  xCdelete++;

  std::cout << "*H4Cna(main)> End of the example." << std::endl;

  if (xCnew != xCdelete) {
    std::cout << "!H4Cna(main)> WRONG MANAGEMENT OF ALLOCATIONS: xCnew = " << xCnew << ", xCdelete = " << xCdelete
              << '\007' << std::endl;
  } else {
    //  std::cout << "*H4Cna(main)> BRAVO! GOOD MANAGEMENT OF ALLOCATIONS: xCnew = "
    //      << xCnew << ", xCdelete = " << xCdelete << std::endl;
  }

  std::cout << "*EcnaCalculationsExample> Exiting main program." << std::endl;
  exit(0);
}
