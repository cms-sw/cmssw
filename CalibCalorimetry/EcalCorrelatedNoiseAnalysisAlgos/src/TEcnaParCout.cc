//---------Author's Name: B.Fabbro DSM/IRFU/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified: 24/03/2010

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParCout.h"

//--------------------------------------
//  TEcnaParCout.cc
//  Class creation: 11 March 2008
//  Documentation: see TEcnaParCout.h
//--------------------------------------

ClassImp(TEcnaParCout)
//______________________________________________________________________________
//

  TEcnaParCout::~TEcnaParCout()
{
//destructor
 // cout << "[Info Management] CLASS: TEcnaParCout.       DESTROY OBJECT: this = " << this << endl;
}
//===================================================================
//
//                   Constructors
//
//===================================================================
TEcnaParCout::TEcnaParCout()
{
// Constructor without argument

 // cout << "[Info Management] CLASS: TEcnaParCout.       CREATE OBJECT: this = " << this << endl;

  Init();
}

TEcnaParCout::TEcnaParCout(TEcnaObject* pObjectManager)
{
// Constructor with argument

 // cout << "[Info Management] CLASS: TEcnaParCout.       CREATE OBJECT: this = " << this << endl;

  Init();
  Long_t i_this = (Long_t)this;
  pObjectManager->RegisterPointer("TEcnaParCout", i_this);
}

void  TEcnaParCout::Init()
{
  fgMaxCar = (Int_t)512;  // max number of characters in TStrings
  fTTBELL = '\007';

  //................................................... Code Print
  fCodePrintNoComment   = GetCodePrint("NoComment");
  fCodePrintWarnings    = GetCodePrint("Warnings ");      // => default
  fCodePrintComments    = GetCodePrint("Comments");
  fCodePrintAllComments = GetCodePrint("AllComments");
  
  fFlagPrint = fCodePrintWarnings;

  //................ Init CNA Command and error numbering
  fCnaCommand = 0;
  fCnaError   = 0;
}// end of Init()

//===========================================================================
//
//         GetCodePrint   
//
//===========================================================================
Int_t TEcnaParCout::GetCodePrint(const TString& chcode)
{
//Get the CodePrint values

  Int_t code_print = 101;  // => default: print warnings

  // The  values must be different
  if( chcode == "NoComment"   ){code_print = 100;}
  if( chcode == "Warnings"    ){code_print = 101;}      // => default
  if( chcode == "Comments"    ){code_print = 102;}
  if( chcode == "AllComments" ){code_print = 103;}

  return code_print;
}
