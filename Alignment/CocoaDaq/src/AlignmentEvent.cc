////////////////////////////////////////////////////////////////////////
//
//                       AlignmentEvent classes
//                       =======================
//
// see AlignmentEvent.hh for class defintions
//
// Author: Gervasio Gomez
//
////////////////////////////////////////////////////////////////////////

using namespace std;
#include <iostream>
#include "Alignment/CocoaDaq/interface/AlignmentEvent.h"

ClassImp(EventHeader)
ClassImp(Position2D)
ClassImp(Position4x1D)
ClassImp(Tilt1D)
ClassImp(Distance)
ClassImp(Temperature)
ClassImp(AlignmentEvent)

//-----------------------------------------------------------------------------

AlignmentEvent::AlignmentEvent()
{
  Header = new EventHeader();
  // define arrays of sensors
  numPos2D         = 0;
  numTilt1D        = 0;
  numDist          = 0;
  numTemp          = 0;
  Array_Position2D = new TClonesArray("Position2D",50);
  Array_Tilt1D     = new TClonesArray("Tilt1D",50);
  Array_Dist       = new TClonesArray("Distance",50);
  Array_Temp       = new TClonesArray("Temperature",50);
}

//-----------------------------------------------------------------------------

void AlignmentEvent::ReadEventFromDB()
{
  Header->ReadTimeStampFromDB();
  for(int i=0;i<numPos2D;i++){
    Position2D *pos2D = (Position2D*) Array_Position2D->At(i); 
    pos2D -> ReadFromDB();
  }
  for(int i=0;i<numPos4x1D;i++){
    Position4x1D *pos4x1D = (Position4x1D*) Array_Position4x1D->At(i); 
    pos4x1D -> ReadFromDB();
  }
  for(int i=0;i<numTilt1D;i++){
    Tilt1D *tilt1D = (Tilt1D*) Array_Tilt1D->At(i); 
    tilt1D -> ReadFromDB();
  }
  for(int i=0;i<numDist;i++){
    Distance *dist = (Distance*) Array_Dist->At(i); 
    dist -> ReadFromDB();
  }
  for(int i=0;i<numTemp;i++){
    Temperature *temp = (Temperature*) Array_Temp->At(i); 
    temp -> ReadFromDB();
  }
}

//-----------------------------------------------------------------------------

void AlignmentEvent::addPos2D(TString ID)
{
  Position2D *pos2D = new( (*Array_Position2D)[numPos2D] ) Position2D();
  pos2D->fID = ID; 
  numPos2D++;
}

//-----------------------------------------------------------------------------

void AlignmentEvent::add4x1D(TString ID)
{
  Position4x1D *pos4x1D = new( (*Array_Position4x1D)[numPos4x1D] ) Position4x1D();
  pos4x1D->fID = ID; 
  numPos4x1D++;
}


//-----------------------------------------------------------------------------
void AlignmentEvent::addTilt1D(TString ID)
{
  Tilt1D *tilt1D = new( (*Array_Tilt1D)[numTilt1D] ) Tilt1D();
  tilt1D->fID = ID; 
  numTilt1D++;
}

//-----------------------------------------------------------------------------
void AlignmentEvent::addDist(TString ID)
{
  Distance *dist = new( (*Array_Dist)[numDist] ) Distance();
  dist->fID = ID; 
  numDist++;
}

//-----------------------------------------------------------------------------
void AlignmentEvent::addTemp(TString ID)
{
  Temperature *temp = new( (*Array_Temp)[numTemp] ) Temperature();
  temp->fID = ID; 
  numTemp++;
}

//-----------------------------------------------------------------------------

void AlignmentEvent::DumpIt()
{
  // Dump to screen all Event info
  Header->DumpIt();
  for(int i=0;i<numPos2D;i++){
    Position2D *pos2D = (Position2D*) Array_Position2D->At(i); 
    pos2D -> DumpIt(pos2D->fID);
  }

  for(int i=0;i<numPos4x1D;i++){
    Position4x1D *pos4x1D = (Position4x1D*) Array_Position4x1D->At(i); 
    pos4x1D -> DumpIt(pos4x1D->fID);
  }
  for(int i=0;i<numTilt1D;i++){
    Tilt1D *tilt1D = (Tilt1D*) Array_Tilt1D->At(i); 
    tilt1D -> DumpIt(tilt1D->fID);
  }
  for(int i=0;i<numDist;i++){
    Distance *dist = (Distance*) Array_Dist->At(i); 
    dist -> DumpIt(dist->fID);
  }
  for(int i=0;i<numTemp;i++){
    Temperature *temp = (Temperature*) Array_Temp->At(i); 
    temp -> DumpIt(temp->fID);
  }
}

//-----------------------------------------------------------------------------

void EventHeader::ReadTimeStampFromDB()
{
  // temporary dummy method. need to replace by SQL query
  fTimeStamp = 1234567;
}

//-----------------------------------------------------------------------------

void EventHeader::DumpIt()
{
  cout<<endl;
  cout<<"------------------------------- Event Header ------------------------------"<<endl;
  cout<<"Time stamp    = "<<fTimeStamp<<endl;
  cout<<"Run number    = "<<fRunNum<<endl;
  cout<<"Event number  = "<<fEvtNum<<endl;
  cout<<endl;
}

//-----------------------------------------------------------------------------

void Position2D::ReadFromDB()
{
  // temporary dummy method. need to replace by SQL query
  fX         = 10.5;
  fY         = 10.5;
  fX_error   = 0.5;
  fY_error   = 0.5;
  //fID      = "Position2D 1";
  fTimeStamp = 1234567;
}

//-----------------------------------------------------------------------------

void Position2D::DumpIt(TString Name)
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

void Position4x1D::ReadFromDB()
{
  // temporary dummy method. need to replace by SQL query
  fDCOPS_up          = 1.5;
  fDCOPS_down        = 1.5;
  fDCOPS_left        = 1.5;
  fDCOPS_right       = 1.5;
  fDCOPS_up_error    = 0.5;
  fDCOPS_down_error  = 0.5;
  fDCOPS_left_error  = 0.5;
  fDCOPS_right_error = 0.5;
  //fID              = "Position4x1D 1";
  fTimeStamp         = 1234567;
}

//-----------------------------------------------------------------------------

void Position4x1D::DumpIt(TString Name)
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

void Tilt1D::ReadFromDB()
{
  // temporary dummy method. need to replace by SQL query
  fTilt       = 3.0;
  fTilt_error = 0.1;
  //fID       = "Tilt1D 1";
  fTimeStamp  = 1234567;
}

//-----------------------------------------------------------------------------

void Tilt1D::DumpIt(TString Name)
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
void Distance::ReadFromDB()
{
  // temporary dummy method. need to replace by SQL query
  fDistance       = 10.5;
  fDistance_error = 0.5;
  //fID           = "Distance 1";
  fTimeStamp      = 1234567;
}

//-----------------------------------------------------------------------------

void Distance::DumpIt(TString Name)
{
  TString dashes = "------------------------------";
  TString line = dashes+Name+dashes;
  cout<<endl;
  cout<<line<<endl;
  cout<<"Time stamp     = "<<fTimeStamp<<endl;
  cout<<"Distance       = "<<fDistance<<endl;
  cout<<"Distance error = "<<fDistance_error<<endl;
  cout<<"ID             = "<<fID<<endl;
  cout<<endl;
}

//-----------------------------------------------------------------------------

void Temperature::ReadFromDB()
{
  // temporary dummy method. need to replace by SQL query
  fTemperature       = 10.5;
  fTemperature_error = 0.5;
  //fID              = "Temperature 1";
  fTimeStamp         = 1234567;
}

//-----------------------------------------------------------------------------

void Temperature::DumpIt(TString Name)
{
  TString dashes = "------------------------------";
  TString line = dashes+Name+dashes;
  cout<<endl;
  cout<<line<<endl;
  cout<<"Time stamp        = "<<fTimeStamp<<endl;
  cout<<"Temperature       = "<<fTemperature<<endl;
  cout<<"Temperature error = "<<fTemperature_error<<endl;
  cout<<"ID                = "<<fID<<endl;
  cout<<endl;
}

//-----------------------------------------------------------------------------

