//---------Author's Name: B.Fabbro DSM/IRFU/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified: 24/03/2011
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"

//--------------------------------------
//  TEcnaObject.cc
//  Class creation: 15 October 2010
//  Documentation: see TEcnaObject.h
//--------------------------------------

ClassImp(TEcnaObject)
//______________________________________________________________________________
//

  TEcnaObject::~TEcnaObject()
{
//destructor

 // cout << "[Info Management] CLASS: TEcnaObject.      DESTROY OBJECT: this = " << this << endl;
}

//===================================================================
//
//                   Constructors
//
//===================================================================
TEcnaObject::TEcnaObject()
{
// Constructor without argument

 // cout << "[Info Management] CLASS: TEcnaObject.      CREATE OBJECT: this = " << this << endl;

  Long_t PointerValue = (Long_t)this;
  Int_t un = 1;
  NumberCreateObjectMessage("TEcnaObject", PointerValue, un);

  Init();

}

void  TEcnaObject::Init()
{
  fgMaxCar = 512;
  fTTBELL = '\007';

  //................ Init pointers to TEcna Objects
  fObjectTEcnaGui        = 0;
  fObjectTEcnaHeader     = 0;
  fObjectTEcnaHistos     = 0;
  fObjectTEcnaNArrayD    = 0;
  fObjectTEcnaNumbering  = 0;
  fObjectTEcnaParCout    = 0;
  fObjectTEcnaParEcal    = 0;
  fObjectTEcnaParHistos  = 0;
  fObjectTEcnaRead       = 0;
  fObjectTEcnaResultType = 0;
  fObjectTEcnaRootFile   = 0;
  fObjectTEcnaRun        = 0;
  fObjectTEcnaWrite      = 0;

  //................ Init counters of TEcna Object creation
  fCounterCreateTEcnaGui        = 0;
  fCounterCreateTEcnaHeader     = 0;
  fCounterCreateTEcnaHistos     = 0;
  fCounterCreateTEcnaNArrayD    = 0;
  fCounterCreateTEcnaNumbering  = 0;
  fCounterCreateTEcnaParCout    = 0;
  fCounterCreateTEcnaParEcal    = 0;
  fCounterCreateTEcnaParHistos  = 0;
  fCounterCreateTEcnaRead       = 0;
  fCounterCreateTEcnaResultType = 0;
  fCounterCreateTEcnaRootFile   = 0;
  fCounterCreateTEcnaRun        = 0;
  fCounterCreateTEcnaWrite      = 0;

  //................ Init counters of TEcna Object re-using
  fCounterReusingTEcnaGui        = 0;
  fCounterReusingTEcnaHeader     = 0;
  fCounterReusingTEcnaHistos     = 0;
  fCounterReusingTEcnaNArrayD    = 0;
  fCounterReusingTEcnaNumbering  = 0;
  fCounterReusingTEcnaParCout    = 0;
  fCounterReusingTEcnaParEcal    = 0;
  fCounterReusingTEcnaParHistos  = 0;
  fCounterReusingTEcnaRead       = 0;
  fCounterReusingTEcnaResultType = 0;
  fCounterReusingTEcnaRootFile   = 0;
  fCounterReusingTEcnaRun        = 0;
  fCounterReusingTEcnaWrite      = 0;

}// end of Init()


//=======================================================================================
//
//              P O I N T E R /  O B J E C T   M A N A G E M E N T 
//
//              TEcnaObject not in list because it is the manager
//
//=======================================================================================
Bool_t TEcnaObject::RegisterPointer(const TString& ClassName, const Long_t& PointerValue)
{
  Bool_t ClassFound = kFALSE;

  if( ClassName == "TEcnaGui"        )
    {
      ClassFound = kTRUE;
      fObjectTEcnaGui = PointerValue;
      fCounterCreateTEcnaGui++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaGui);
    }

  if( ClassName == "TEcnaHeader"     )
    {
      ClassFound = kTRUE;
      fObjectTEcnaHeader = PointerValue;
      fCounterCreateTEcnaHeader++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaHeader);
    }

  if( ClassName == "TEcnaHistos"     )
    {
      ClassFound = kTRUE;
      fObjectTEcnaHistos = PointerValue;
      fCounterCreateTEcnaHistos++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaHistos);
    }

  if( ClassName == "TEcnaNArrayD"    )
    {
      ClassFound = kTRUE;
      fObjectTEcnaNArrayD = PointerValue;
      fCounterCreateTEcnaNArrayD++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaNArrayD);
    }

  if( ClassName == "TEcnaNumbering"  )
    {
      ClassFound = kTRUE;
      fObjectTEcnaNumbering = PointerValue;
      fCounterCreateTEcnaNumbering++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaNumbering);
    }

  if( ClassName == "TEcnaParCout"    )
    {
      ClassFound = kTRUE;
      fObjectTEcnaParCout = PointerValue;
      fCounterCreateTEcnaParCout++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaParCout);
    }

  if( ClassName == "TEcnaParEcal"    )
    {
      ClassFound = kTRUE;
      fObjectTEcnaParEcal = PointerValue;
      fCounterCreateTEcnaParEcal++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaParEcal);
    }

  if( ClassName == "TEcnaParHistos"  )
    {
      ClassFound = kTRUE;
      fObjectTEcnaParHistos = PointerValue;
      fCounterCreateTEcnaParHistos++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaParHistos);
    }

  if( ClassName == "TEcnaParPaths"   )
    {
      ClassFound = kTRUE;
      fObjectTEcnaParPaths = PointerValue;
      fCounterCreateTEcnaParPaths++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaParPaths);
    }

  if( ClassName == "TEcnaRead"       )
    {
      ClassFound = kTRUE;
      fObjectTEcnaRead = PointerValue;
      fCounterCreateTEcnaRead++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaRead);
    }

  if( ClassName == "TEcnaResultType" )
    {
      ClassFound = kTRUE;
      fObjectTEcnaResultType = PointerValue;
      fCounterCreateTEcnaResultType++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaResultType);
    }

  if( ClassName == "TEcnaRootFile"   )
    {
      ClassFound = kTRUE;
      fObjectTEcnaRootFile = PointerValue;
      fCounterCreateTEcnaRootFile++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaRootFile);
    }

  if( ClassName == "TEcnaRun"        )
    {
      ClassFound = kTRUE;
      fObjectTEcnaRun = PointerValue;
      fCounterCreateTEcnaRun++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaRun);
    }

  if( ClassName == "TEcnaWrite"      )
    {
      ClassFound = kTRUE;
      fObjectTEcnaWrite = PointerValue;
      fCounterCreateTEcnaWrite++;
      NumberCreateObjectMessage(ClassName.Data(), PointerValue, fCounterCreateTEcnaWrite);
    }

  //.........................................................................................
  if( ClassFound == kFALSE )
    {
      cout << "!TEcnaObject::RegisterPointer(...)> Class " << ClassName
	   << " not found." << fTTBELL << endl;
    }

  return ClassFound;
} // end of RegisterPointer(...)


Long_t TEcnaObject::GetPointerValue(const TString& ClassName)
{
  Long_t PointerValue = 0;

  if( ClassName == "TEcnaGui" )
    {
      PointerValue = fObjectTEcnaGui;
      fCounterReusingTEcnaGui++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaGui);
    }

  if( ClassName == "TEcnaHeader" )
    {
      PointerValue = fObjectTEcnaHeader;
      fCounterReusingTEcnaHeader++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaHeader);
    }

  if( ClassName == "TEcnaHistos" )
    {
      PointerValue = fObjectTEcnaHistos;
      fCounterReusingTEcnaHistos++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaHistos);
    }

  if( ClassName == "TEcnaNArrayD" )
    {
      PointerValue = fObjectTEcnaNArrayD;
      fCounterReusingTEcnaNArrayD++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaNArrayD);
    }

  if( ClassName == "TEcnaNumbering" )
    {
      PointerValue = fObjectTEcnaNumbering;
      fCounterReusingTEcnaNumbering++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaNumbering);
    }

  if( ClassName == "TEcnaParCout" )
    {
      PointerValue = fObjectTEcnaParCout;
      fCounterReusingTEcnaParCout++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaParCout);
    }

  if( ClassName == "TEcnaParEcal" )
    {
      PointerValue = fObjectTEcnaParEcal;
      fCounterReusingTEcnaParEcal++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaParEcal);
    }

  if( ClassName == "TEcnaParHistos" )
    {
      PointerValue = fObjectTEcnaParHistos;
      fCounterReusingTEcnaParHistos++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaParHistos);
    }

  if( ClassName == "TEcnaParPaths" )
    {
      PointerValue = fObjectTEcnaParPaths;
      fCounterReusingTEcnaParPaths++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaParPaths);
    }

  if( ClassName == "TEcnaRead" )
    {
      PointerValue = fObjectTEcnaRead;
      fCounterReusingTEcnaRead++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaRead);
    }

  if( ClassName == "TEcnaResultType" )
    {
      PointerValue = fObjectTEcnaResultType;
      fCounterReusingTEcnaResultType++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaResultType);
    }

  if( ClassName == "TEcnaRootFile" )
    {
      PointerValue = fObjectTEcnaRootFile;
      fCounterReusingTEcnaRootFile++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaRootFile);
    }

  if( ClassName == "TEcnaRun" )
    {
      PointerValue = fObjectTEcnaRun;
      fCounterReusingTEcnaRun++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaRun);
    }

  if( ClassName == "TEcnaWrite" )
    {
      PointerValue = fObjectTEcnaWrite;
      fCounterReusingTEcnaWrite++;
      NumberReuseObjectMessage(ClassName.Data(), PointerValue, fCounterReusingTEcnaWrite);
    }

  return PointerValue;
}

void TEcnaObject::NumberCreateObjectMessage(const TString& ClassName, const Long_t& PointerValue, const Int_t& NbOfObjects)
{
#define NOCM
#ifndef NOCM
  cout << "*TEcnaObject::NumberCreateObjectMessage(...)> New ECNA object (pointer = "
       << PointerValue << ") from TEcnaObject " << this 
       << ". Object# = " << setw(8) << NbOfObjects
       << ", Class: " << ClassName;
  if( NbOfObjects > 1 ){cout << " (INFO: more than 1 object)";}
  cout << endl;
#endif // NOCM
}

void TEcnaObject::NumberReuseObjectMessage(const TString& ClassName, const Long_t& PointerValue, const Int_t& NbOfObjects)
{
#define NOCR
#ifndef NOCR
  if( PointerValue != 0 )
    {
      cout << "*TEcnaObject::GetPointerValue(...)> INFO: pointer " << PointerValue
	   << " used again from TEcnaObject " << this
	   << ". " << setw(8) << NbOfObjects << " times, class: " << ClassName;
    }
  cout << endl;
#endif // NOCR
}
