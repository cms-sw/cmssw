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
  std::cout<<std::endl;
  std::cout<<"------------------------------- Event Header ------------------------------"<<std::endl;
  std::cout<<"Time stamp    = "<<fTimeStamp<<std::endl;
  std::cout<<"Run number    = "<<fRunNum<<std::endl;
  std::cout<<"Event number  = "<<fEvtNum<<std::endl;
  std::cout<<std::endl;
}

//-----------------------------------------------------------------------------

void AliDaqPosition2D::DumpIt(TString Name)
{
  TString dashes = "------------------------------";
  TString line = dashes+Name+dashes;
  std::cout<<std::endl;
  std::cout<<line<<std::endl;
  std::cout<<"Time stamp = "<<fTimeStamp<<std::endl;
  std::cout<<"X          = "<<fX<<std::endl;
  std::cout<<"Y          = "<<fY<<std::endl;
  std::cout<<"X error    = "<<fX_error<<std::endl;
  std::cout<<"Y error    = "<<fY_error<<std::endl;
  std::cout<<"ID         = "<<fID<<std::endl;
  std::cout<<std::endl;
}

//-----------------------------------------------------------------------------

void AliDaqPositionCOPS::DumpIt(TString Name)
{
  TString dashes = "------------------------------";
  TString line = dashes+Name+dashes;
  std::cout<<std::endl;
  std::cout<<line<<std::endl;
  std::cout<<"Time stamp        = "<<fTimeStamp<<std::endl;
  std::cout<<"DCOPS_up          = "<<fDCOPS_up<<std::endl;
  std::cout<<"DCOPS_down        = "<<fDCOPS_down<<std::endl;
  std::cout<<"DCOPS_left        = "<<fDCOPS_left<<std::endl;
  std::cout<<"DCOPS_right       = "<<fDCOPS_right<<std::endl;
  std::cout<<"DCOPS_up_error    = "<<fDCOPS_up_error<<std::endl;
  std::cout<<"DCOPS_down_error  = "<<fDCOPS_down_error<<std::endl;
  std::cout<<"DCOPS_left_error  = "<<fDCOPS_left_error<<std::endl;
  std::cout<<"DCOPS_right_error = "<<fDCOPS_right_error<<std::endl;
  std::cout<<"ID                = "<<fID<<std::endl;
  std::cout<<std::endl;
}

//-----------------------------------------------------------------------------

void AliDaqTilt::DumpIt(TString Name)
{
  TString dashes = "------------------------------";
  TString line = dashes+Name+dashes;
  std::cout<<std::endl;
  std::cout<<line<<std::endl;
  std::cout<<"Time stamp = "<<fTimeStamp<<std::endl;
  std::cout<<"Tilt       = "<<fTilt<<std::endl;
  std::cout<<"Tilt error = "<<fTilt_error<<std::endl;
  std::cout<<"ID         = "<<fID<<std::endl;
  std::cout<<std::endl;
}

//-----------------------------------------------------------------------------

void AliDaqDistance::DumpIt(TString Name)
{
  TString dashes = "------------------------------";
  TString line = dashes+Name+dashes;
  std::cout<<std::endl;
  std::cout<<line<<std::endl;
  std::cout<<"Time stamp     = "<<fTimeStamp<<std::endl;
  std::cout<<"AliDaqDistance       = "<<fDistance<<std::endl;
  std::cout<<"AliDaqDistance error = "<<fDistance_error<<std::endl;
  std::cout<<"ID             = "<<fID<<std::endl;
  std::cout<<std::endl;
}

//-----------------------------------------------------------------------------

void AliDaqTemperature::DumpIt(TString Name)
{
  TString dashes = "------------------------------";
  TString line = dashes+Name+dashes;
  std::cout<<std::endl;
  std::cout<<line<<std::endl;
  std::cout<<"Time stamp        = "<<fTimeStamp<<std::endl;
  std::cout<<"AliDaqTemperature       = "<<fTemperature<<std::endl;
  std::cout<<"AliDaqTemperature error = "<<fTemperature_error<<std::endl;
  std::cout<<"ID                = "<<fID<<std::endl;
  std::cout<<std::endl;
}

//-----------------------------------------------------------------------------

