#include "../interface/CocoaDaqReaderRoot.h"
#define private public
#include "Alignment/CocoaDaq/interface/AlignmentEvent.h"
#define private private

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurements.h"

using namespace std;
#include <iostream>

#include "TROOT.h"
#include "TBranch.h"
#include "TClonesArray.h"


//----------------------------------------------------------------------
CocoaDaqReaderRoot::CocoaDaqReaderRoot(const std::string& m_inFileName )
{

  // Open root file
  TFile *file = new TFile("cocoaDaqTest.root"); 

  // Read TTree named "Alignment_All_Cocoa" in memory. 
  TTree *tree = (TTree*)file->Get("Alignment_All_Cocoa");

  int Nev = tree->GetEntries(); // number of entries in Tree
  int nb  = 0;   // dummy, number of bytes

  // LinkEvent object must be created before setting the branch address
  AlignmentEvent *event = new AlignmentEvent();

  // link pointer to Tree branch (see Alignment_Cocoa_AllWriter.cc)
  tree->SetBranchAddress("Alignment_All", &event);  

  std::cout << "got tree branch " << std::endl;
  // Loop over all events
  for(int ev=0;ev<Nev;ev++){
    nb = tree->GetEntry(ev);  // read in entire event

    // Every n events, dump one to screen
    if(ev%1 == 0) {

      event->DumpIt();

      // Example to access Sensir2D information
      cout<<"Event "<<ev<<endl;
      for(int i=0; i<event->numPos2D; i++) {
         Position2D* pos2D = (Position2D*) event->Array_Position2D->At(i);
         cout<<"Link 2D sensor "<<i<<" has ID = "<<pos2D->fID
	     <<" and (x,y) = ("<<pos2D->fX<<","<<pos2D->fY<<")"<<endl;
      }
      // Example to access Endcap DCOPS pos4x1D information
      cout<<"Event "<<ev<<endl;
      for(int i=0; i<event->numPos4x1D; i++) {
         Position4x1D* pos4x1D = (Position4x1D*) event->Array_Position4x1D->At(i);
         cout<<"Endcap 4x1D sensor "<<i<<" has ID = "<<pos4x1D->fID
	     <<" and (u,d,l,t) = ("<<pos4x1D->fDCOPS_up<<","
             <<pos4x1D->fDCOPS_down<<","<<pos4x1D->fDCOPS_left<<","
             <<pos4x1D->fDCOPS_right<<")"<<endl;
      }
    } // if event is multiple of 50
  } // loop over all entries

  file->Close();


  // Open root file
  theFile = new TFile(m_inFileName.c_str()); 

  // Read TTree named "Alignment_Link_Cocoa" in memory.  !! SHOULD BE CALLED Alignment_Cocoa
  theTree = (TTree*)theFile->Get("Alignment_All_Cocoa");

  std::cout << "CocoaDaqReaderRoot tree opened " << theTree << std::endl;

  nev = theTree->GetEntries(); // number of entries in Tree
  nextEvent = 0;

  // LinkEvent object must be created before setting the branch address
  theEvent = new AlignmentEvent();

  // link pointer to Tree branch
  theTree->SetBranchAddress("Alignment_All", &theEvent);  //  !! SHOULD BE CALLED Alignment_Cocoa

}

//----------------------------------------------------------------------
CocoaDaqReaderRoot::~CocoaDaqReaderRoot()
{
  theFile->Close();
}

//----------------------------------------------------------------------
bool CocoaDaqReaderRoot::ReadNextEvent()
{
  return ReadEvent( nextEvent );
}


//----------------------------------------------------------------------
bool CocoaDaqReaderRoot::ReadEvent( int nev )
{
  std::vector<OpticalAlignMeasurementInfo> measList;

  int nb  = 0;   // dummy, number of bytes
  // Loop over all events
  std::cout << "CocoaDaqReaderRoot::ReadEvent " << ++nev << " tree " << theTree << std::endl;
  nb = theTree->GetEntry(nev);  // read in entire event

  std::cout << "CocoaDaqReaderRott reading event " << nev << " " << nb << std::endl;
  if( nb == 0 ) return 0; //end of file reached??

  // Every n events, dump one to screen
  if(nev%50 == 0) theEvent->DumpIt();
  
  // Example to access Link information
  cout<<" Event "<< nev<<endl;
  
  for(int ii=0; ii<theEvent->GetNumPos2D(); ii++) {
    Position2D* pos2D = (Position2D*) theEvent->GetArray_Position2D()->At(ii);
    cout<<"Link 2D sensor "<<ii<<" has ID = "<<pos2D->fID
	<<" and (x,y) = ("<<pos2D->fX<<","<<pos2D->fY<<")"<<endl;
     measList.push_back( GetMeasFromPosition2D( pos2D ) );
  }
  for(int ii=0; ii<theEvent->GetNumPos4x1D(); ii++) {
    Position4x1D* pos4x1D = (Position4x1D*) theEvent->GetArray_Position4x1D()->At(ii);
     measList.push_back( GetMeasFromPosition4x1D( pos4x1D ) );
  }
  for(int ii=0; ii<theEvent->GetNumTilt1D(); ii++) {
    Tilt1D* tilt1D = (Tilt1D*) theEvent->GetArray_Tilt1D()->At(ii);
     measList.push_back( GetMeasFromTilt1D( tilt1D ) );
  }
  for(int ii=0; ii<theEvent->GetNumDist(); ii++) {
    Distance* dist = (Distance*) theEvent->GetArray_Dist()->At(ii);
     measList.push_back( GetMeasFromDist( dist ) );
  }

  nextEvent = nev + 1;

  BuildMeasurementsFromOptAlign( measList );

  return 1;

}

//----------------------------------------------------------------------
OpticalAlignMeasurementInfo CocoaDaqReaderRoot::GetMeasFromPosition2D( Position2D* pos2D )
{
  OpticalAlignMeasurementInfo meas;
  
  meas.type_ = "SENSOR2D";
  meas.name_ = pos2D->fID;
  //-   std::vector<std::string> measObjectNames_;
  std::vector<bool> isSimu;
  for( size_t jj = 0; jj < 2; jj++ ){
    isSimu.push_back(false); 
  }
  meas.isSimulatedValue_ = isSimu; 
  std::vector<OpticalAlignParam> paramList;
  OpticalAlignParam oaParam1;
  oaParam1.name_ = "H:";
  oaParam1.value_ = pos2D->GetX();
  oaParam1.error_ = pos2D->GetXerror();
  paramList.push_back(oaParam1);
  
  OpticalAlignParam oaParam2;
  oaParam2.name_ = "V:";
  oaParam2.value_ = pos2D->GetY();
  oaParam2.error_ = pos2D->GetYerror();
  paramList.push_back(oaParam2);
  
  meas.values_ = paramList;

  return meas;
}


//----------------------------------------------------------------------
OpticalAlignMeasurementInfo CocoaDaqReaderRoot::GetMeasFromPosition4x1D( Position4x1D* pos4x1D )
{
  OpticalAlignMeasurementInfo meas;
  
  meas.type_ = "COPS";
  meas.name_ = pos4x1D->fID;
  //-   std::vector<std::string> measObjectNames_;
  std::vector<bool> isSimu;
  for( size_t jj = 0; jj < 4; jj++ ){
    isSimu.push_back(false); 
  }
  meas.isSimulatedValue_ = isSimu; 

  std::vector<OpticalAlignParam> paramList;
  OpticalAlignParam oaParam1;
  oaParam1.name_ = "U:";
  oaParam1.value_ = pos4x1D->GetUp();
  oaParam1.error_ = pos4x1D->GetUpError();
  paramList.push_back(oaParam1);

  OpticalAlignParam oaParam2;
  oaParam2.name_ = "U:";
  oaParam2.value_ = pos4x1D->GetDown();
  oaParam2.error_ = pos4x1D->GetDownError();
  paramList.push_back(oaParam2);

  OpticalAlignParam oaParam3;
  oaParam3.name_ = "U:";
  oaParam3.value_ = pos4x1D->GetRight();
  oaParam3.error_ = pos4x1D->GetRightError();
  paramList.push_back(oaParam3);

  OpticalAlignParam oaParam4;
  oaParam4.name_ = "U:";
  oaParam4.value_ = pos4x1D->GetLeft();
  oaParam4.error_ = pos4x1D->GetLeftError();
  paramList.push_back(oaParam4);
  
  meas.values_ = paramList;

  return meas;

}

//----------------------------------------------------------------------
OpticalAlignMeasurementInfo CocoaDaqReaderRoot::GetMeasFromTilt1D( Tilt1D* tilt1D )
{
  OpticalAlignMeasurementInfo meas;
  
  meas.type_ = "SENSOR2D";
  meas.name_ = tilt1D->fID;
  //-   std::vector<std::string> measObjectNames_;
  std::vector<bool> isSimu;
  for( size_t jj = 0; jj < 2; jj++ ){
    isSimu.push_back(false); 
  }
  meas.isSimulatedValue_ = isSimu; 
  std::vector<OpticalAlignParam> paramList;
  OpticalAlignParam oaParam;
  oaParam.name_ = "T:";
  oaParam.value_ = tilt1D->GetTilt();
  oaParam.error_ = tilt1D->GetTiltError();
  paramList.push_back(oaParam);
  
  meas.values_ = paramList;


  return meas;

}


//----------------------------------------------------------------------
OpticalAlignMeasurementInfo CocoaDaqReaderRoot::GetMeasFromDist( Distance* dist )
{
  OpticalAlignMeasurementInfo meas;
  
  meas.type_ = "SENSOR2D";
  meas.name_ = dist->fID;
  //-   std::vector<std::string> measObjectNames_;
  std::vector<bool> isSimu;
  for( size_t jj = 0; jj < 2; jj++ ){
    isSimu.push_back(false); 
  }
  meas.isSimulatedValue_ = isSimu; 
  std::vector<OpticalAlignParam> paramList;
  OpticalAlignParam oaParam;
  oaParam.name_ = "D:";
  oaParam.value_ = dist->GetDistance();
  oaParam.error_ = dist->GetDistanceError();
  paramList.push_back(oaParam);

  meas.values_ = paramList;

  return meas;

}

void CocoaDaqReaderRoot::BuildMeasurementsFromOptAlign( std::vector<OpticalAlignMeasurementInfo>& measList )
{

}
