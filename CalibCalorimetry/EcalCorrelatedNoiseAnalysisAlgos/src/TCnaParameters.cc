//---------Author's Name: B.Fabbro DSM/DAPNIA/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified: 07/06/2007 - CNA version 3.1 -
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaParameters.h"

ClassImp(TCnaParameters)
//______________________________________________________________________________
//
// TCnaParameters.
//
//    Values of different parameters for plots in the framework of TCnaViewEB
//    (see description of this class)
//
//    Examples of parameters:  ymin and ymax values for histos, 
//                             "Period of Run" description by TString
//
//-------------------------------------------------------------------------
//
//        For more details on other classes of the CNA package:
//
//                 http://www.cern.ch/cms-fabbro/cna
//
//-------------------------------------------------------------------------
//

//---------------------- TCnaParameters.cxx -----------------------------
//  
//   Creation (first version): 19 May 2005
//
//   For questions or comments, please send e-mail to Bernard Fabbro:
//             
//   fabbro@hep.saclay.cea.fr 
//
//------------------------------------------------------------------------

  TCnaParameters::~TCnaParameters()
{
//destructor

}

//===================================================================
//
//                   Constructor without arguments
//
//===================================================================
TCnaParameters::TCnaParameters()
{
  Init();
}

void  TCnaParameters::Init()
{
  fgMaxCar = (Int_t)512;
}

//===================================================================
//
//                   Methods
//
//===================================================================

//===========================================================================
//
//               SetPeriodTitles,  PeriodOfRun
//
//===========================================================================
void TCnaParameters::SetPeriodTitles()
{
//Define the titles of the periods

  fPeriod2002   = "Test beam 2002. Module M0'";

  fPeriod2003   = "Test beam during 2003: SM0 and SM1";

  fPeriod2004_1 = "29 May - 08 June 2004: first period with E0'";
  fPeriod2004_2 = "08 - 22 June 2004: second period with E0'";
  fPeriod2004_3 = "27 June - 07 July 2004: third period with E0'";
  fPeriod2004_4 = "21 July - 05 August 2004: fourth period with E0'";
  fPeriod2004_5 = "02 - 06 Sept 2004: fifth period with E0'";
  fPeriod2004_6 = "05 Oct - 16 Nov: Test beam 2004 supermodule SM10 ";

  fPeriod2005   = "Cosmic ray test in H4";
  fPeriod2006_1 = "Cosmic ray test in H4";
  fPeriod2006_2 = "Test beam CMS/ECAL";

}
//..................................................................
TString TCnaParameters::PeriodOfRun(const Int_t& run_number)
{
//Set the period of the run from the correspondance between
//run numbers and period

  TString chperiod;
  Int_t MaxCar = fgMaxCar;
  chperiod.Resize(MaxCar);
  chperiod = "(no period info)";

  //......................................................... 2002
  if ( run_number >= 50000  &&  run_number < 59999 )
    {chperiod = fPeriod2002;}
  //......................................................... 2003
  if ( run_number >= 60000 && run_number < 68050 )
    {chperiod = fPeriod2003;}

  //......................................................... 2004
  if ( run_number >= 68051 && run_number < 69055 )
    {chperiod = fPeriod2004_1;}

  if ( run_number >= 69056 && run_number < 70299 )
    {chperiod = fPeriod2004_2;}
  
  if ( run_number >= 70300 && run_number < 71332 )
    {chperiod = fPeriod2004_3;}
  
  if ( run_number >= 71333 && run_number < 72188 )
    {chperiod = fPeriod2004_4;}
  
  if ( run_number >= 72189 && run_number < 72744 )
    {chperiod = fPeriod2004_5;}

  if ( run_number >= 72745 && run_number < 9999999 )
    {chperiod = fPeriod2004_6;}

  //......................................................... 2005
  // no test beam in 2005

  //......................................................... 2006
  if ( run_number > 0 )
    {chperiod = fPeriod2006_2;}
  
 return chperiod;
}
//===========================================================================
//
//        GetCodePrintNoComment(),   
//
//===========================================================================
Int_t TCnaParameters::GetCodePrint(const TString chcode)
{
//Get the CodePrint values

  Int_t code_print = 101;  // => default value: print warnings

  // The  values must be different

  if( chcode == "NoComment"   ){code_print = 100;}
  if( chcode == "Warnings"    ){code_print = 101;}      // => default value
  if( chcode == "Comments"    ){code_print = 102;}
  if( chcode == "AllComments" ){code_print = 103;}

  return code_print;
}
