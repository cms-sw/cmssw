////////////////////////////////////////////////////////////////////////
//
//                       Alignment_Cocoa classes
//                       =======================
//
// see Alignment_Cocoa.hh for class defintions
//
// Author: Gervasio Gomez
//
////////////////////////////////////////////////////////////////////////

using namespace std;
#include <iostream>
#include "Alignment/CocoaDaq/interface/CocoaDaqRootEvent.h"

ClassImp(AliDaqEventHeader)
ClassImp(AliDaqPosition2D)
ClassImp(AliDaqPositionCOPS)
ClassImp(AliDaqTilt)
ClassImp(AliDaqDistance)
ClassImp(AliDaqTemperature)
ClassImp(CocoaDaqRootEvent)

//-----------------------------------------------------------------------------
CocoaDaqRootEvent::CocoaDaqRootEvent()
{
  Header = new AliDaqEventHeader();
  // define arrays of sensors
  numPosCOPS       = 0;
  numPos2D         = 0;
  numTilt        = 0;
  numDist          = 0;
  numTemp          = 0;
  Array_PositionCOPS = new TClonesArray("AliDaqPositionCOPS",50);
  Array_Position2D   = new TClonesArray("AliDaqPosition2D",50);
  Array_Tilt       = new TClonesArray("AliDaqTilt",50);
  Array_Dist         = new TClonesArray("AliDaqDistance",50);
  Array_Temp         = new TClonesArray("AliDaqTemperature",50);
}

//-----------------------------------------------------------------------------

void CocoaDaqRootEvent::DumpIt()
{
  // Dump to screen all Alignment info
  Header->DumpIt();
  for(int i=0;i<numPosCOPS;i++){
    AliDaqPositionCOPS *posCOPS = (AliDaqPositionCOPS*) Array_PositionCOPS->At(i); 
    posCOPS -> DumpIt(posCOPS->GetID());
  }
  for(int i=0;i<numPos2D;i++){
    AliDaqPosition2D *pos2D = (AliDaqPosition2D*) Array_Position2D->At(i); 
    pos2D -> DumpIt(pos2D->GetID());
  }
  for(int i=0;i<numTilt;i++){
    AliDaqTilt *tilt = (AliDaqTilt*) Array_Tilt->At(i); 
    tilt -> DumpIt(tilt->GetID());
  }
  for(int i=0;i<numDist;i++){
    AliDaqDistance *dist = (AliDaqDistance*) Array_Dist->At(i); 
    dist -> DumpIt(dist->GetID());
  }
  for(int i=0;i<numTemp;i++){
    AliDaqTemperature *temp = (AliDaqTemperature*) Array_Temp->At(i); 
    temp -> DumpIt(temp->GetID());
  }
}

//-----------------------------------------------------------------------------

void AliDaqEventHeader::DumpIt()
{
  cout<<endl;
  cout<<"------------------------------- Event Header ------------------------------"<<endl;
  cout<<"Time stamp    = "<<fTimeStamp<<endl;
  cout<<"Run number    = "<<fRunNum<<endl;
  cout<<"Event number  = "<<fEvtNum<<endl;
  cout<<endl;
}

//-----------------------------------------------------------------------------

void AliDaqPosition2D::DumpIt(TString Name)
{
  TString dashes = "------------------------------";
  TString line = dashes+Name+dashes;
  cout<<endl;
  cout<<line<<endl;
  cout<<"Time stamp = "<<fTimeStamp<<endl;
  cout<<"X          = "<<fX<<endl;
  cout<<"Y          = "<<fY<<endl;
  cout<<"X error    = "<<fX_error<<endl;
  cout<<"Y error    = "<<fY_error<<endl;
  cout<<"ID         = "<<fID<<endl;
  cout<<endl;
}

//-----------------------------------------------------------------------------

void AliDaqPositionCOPS::DumpIt(TString Name)
{
  TString dashes = "------------------------------";
  TString line = dashes+Name+dashes;
  cout<<endl;
  cout<<line<<endl;
  cout<<"Time stamp        = "<<fTimeStamp<<endl;
  cout<<"DCOPS_up          = "<<fDCOPS_up<<endl;
  cout<<"DCOPS_down        = "<<fDCOPS_down<<endl;
  cout<<"DCOPS_left        = "<<fDCOPS_left<<endl;
  cout<<"DCOPS_right       = "<<fDCOPS_right<<endl;
  cout<<"DCOPS_up_error    = "<<fDCOPS_up_error<<endl;
  cout<<"DCOPS_down_error  = "<<fDCOPS_down_error<<endl;
  cout<<"DCOPS_left_error  = "<<fDCOPS_left_error<<endl;
  cout<<"DCOPS_right_error = "<<fDCOPS_right_error<<endl;
  cout<<"ID                = "<<fID<<endl;
  cout<<endl;
}

//-----------------------------------------------------------------------------

void AliDaqTilt::DumpIt(TString Name)
{
  TString dashes = "------------------------------";
  TString line = dashes+Name+dashes;
  cout<<endl;
  cout<<line<<endl;
  cout<<"Time stamp = "<<fTimeStamp<<endl;
  cout<<"Tilt       = "<<fTilt<<endl;
  cout<<"Tilt error = "<<fTilt_error<<endl;
  cout<<"ID         = "<<fID<<endl;
  cout<<endl;
}

//-----------------------------------------------------------------------------

void AliDaqDistance::DumpIt(TString Name)
{
  TString dashes = "------------------------------";
  TString line = dashes+Name+dashes;
  cout<<endl;
  cout<<line<<endl;
  cout<<"Time stamp     = "<<fTimeStamp<<endl;
  cout<<"AliDaqDistance       = "<<fDistance<<endl;
  cout<<"AliDaqDistance error = "<<fDistance_error<<endl;
  cout<<"ID             = "<<fID<<endl;
  cout<<endl;
}

//-----------------------------------------------------------------------------

void AliDaqTemperature::DumpIt(TString Name)
{
  TString dashes = "------------------------------";
  TString line = dashes+Name+dashes;
  cout<<endl;
  cout<<line<<endl;
  cout<<"Time stamp        = "<<fTimeStamp<<endl;
  cout<<"AliDaqTemperature       = "<<fTemperature<<endl;
  cout<<"AliDaqTemperature error = "<<fTemperature_error<<endl;
  cout<<"ID                = "<<fID<<endl;
  cout<<endl;
}

//-----------------------------------------------------------------------------

