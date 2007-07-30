#include "../interface/CocoaDaqReaderRoot.h"
#include "Alignment/CocoaDaq/interface/CocoaDaqRootEvent.h"

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
  theFile = new TFile(m_inFileName.c_str()); 

  // Read TTree named "CocoaDaq" in memory.  !! SHOULD BE CALLED Alignment_Cocoa
  theTree = (TTree*)theFile->Get("CocoaDaq");

  nev = theTree->GetEntries(); // number of entries in Tree
  nextEvent = 0;

  // Event object must be created before setting the branch address
  theEvent = new CocoaDaqRootEvent();

  // link pointer to Tree branch
  theTree->SetBranchAddress("CocoaDaq", &theEvent);  //  !! SHOULD BE CALLED Alignment_Cocoa

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
  nb = theTree->GetEntry(nev);  // read in entire event

  std::cout << "CocoaDaqReaderRoot reading event " << nev << " " << nb << std::endl;
  if( nb == 0 ) return 0; //end of file reached??

  // Every n events, dump one to screen
  if(nev%50 == 0) theEvent->DumpIt();
  
  cout<<" Event "<< nev <<endl;
  
  for(int ii=0; ii<theEvent->GetNumPos2D(); ii++) {
    AliDaqPosition2D* pos2D = (AliDaqPosition2D*) theEvent->GetArray_Position2D()->At(ii);
    cout<<"2D sensor "<<ii<<" has ID = "<<pos2D->GetID()
	<<" and (x,y) = ("<<pos2D->GetX()<<","<<pos2D->GetY()<<")"<<endl;
     measList.push_back( GetMeasFromPosition2D( pos2D ) );
  }
  for(int ii=0; ii<theEvent->GetNumPosCOPS(); ii++) {
    AliDaqPositionCOPS* posCOPS = (AliDaqPositionCOPS*) theEvent->GetArray_PositionCOPS()->At(ii);
     measList.push_back( GetMeasFromPositionCOPS( posCOPS ) );
  }
  for(int ii=0; ii<theEvent->GetNumTilt(); ii++) {
    AliDaqTilt* tilt = (AliDaqTilt*) theEvent->GetArray_Tilt()->At(ii);
     measList.push_back( GetMeasFromTilt( tilt ) );
  }
  for(int ii=0; ii<theEvent->GetNumDist(); ii++) {
    AliDaqDistance* dist = (AliDaqDistance*) theEvent->GetArray_Dist()->At(ii);
     measList.push_back( GetMeasFromDist( dist ) );
  }

  nextEvent = nev + 1;

  BuildMeasurementsFromOptAlign( measList );

  return 1;

}

//----------------------------------------------------------------------
OpticalAlignMeasurementInfo CocoaDaqReaderRoot::GetMeasFromPosition2D( AliDaqPosition2D* pos2D )
{
  OpticalAlignMeasurementInfo meas;
  
  meas.type_ = "SENSOR2D";
  meas.name_ = pos2D->GetID();
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
OpticalAlignMeasurementInfo CocoaDaqReaderRoot::GetMeasFromPositionCOPS( AliDaqPositionCOPS* posCOPS )
{
  OpticalAlignMeasurementInfo meas;
  
  meas.type_ = "COPS";
  meas.name_ = posCOPS->GetID();
  //-   std::vector<std::string> measObjectNames_;
  std::vector<bool> isSimu;
  for( size_t jj = 0; jj < 4; jj++ ){
    isSimu.push_back(false); 
  }
  meas.isSimulatedValue_ = isSimu; 

  std::vector<OpticalAlignParam> paramList;
  OpticalAlignParam oaParam1;
  oaParam1.name_ = "U:";
  oaParam1.value_ = posCOPS->GetUp();
  oaParam1.error_ = posCOPS->GetUpError();
  paramList.push_back(oaParam1);

  OpticalAlignParam oaParam2;
  oaParam2.name_ = "U:";
  oaParam2.value_ = posCOPS->GetDown();
  oaParam2.error_ = posCOPS->GetDownError();
  paramList.push_back(oaParam2);

  OpticalAlignParam oaParam3;
  oaParam3.name_ = "U:";
  oaParam3.value_ = posCOPS->GetRight();
  oaParam3.error_ = posCOPS->GetRightError();
  paramList.push_back(oaParam3);

  OpticalAlignParam oaParam4;
  oaParam4.name_ = "U:";
  oaParam4.value_ = posCOPS->GetLeft();
  oaParam4.error_ = posCOPS->GetLeftError();
  paramList.push_back(oaParam4);
  
  meas.values_ = paramList;

  return meas;

}

//----------------------------------------------------------------------
OpticalAlignMeasurementInfo CocoaDaqReaderRoot::GetMeasFromTilt( AliDaqTilt* tilt )
{
  OpticalAlignMeasurementInfo meas;
  
  meas.type_ = "SENSOR2D";
  meas.name_ = tilt->GetID();
  //-   std::vector<std::string> measObjectNames_;
  std::vector<bool> isSimu;
  for( size_t jj = 0; jj < 2; jj++ ){
    isSimu.push_back(false); 
  }
  meas.isSimulatedValue_ = isSimu; 
  std::vector<OpticalAlignParam> paramList;
  OpticalAlignParam oaParam;
  oaParam.name_ = "T:";
  oaParam.value_ = tilt->GetTilt();
  oaParam.error_ = tilt->GetTiltError();
  paramList.push_back(oaParam);
  
  meas.values_ = paramList;


  return meas;

}


//----------------------------------------------------------------------
OpticalAlignMeasurementInfo CocoaDaqReaderRoot::GetMeasFromDist( AliDaqDistance* dist )
{
  OpticalAlignMeasurementInfo meas;
  
  meas.type_ = "SENSOR2D";
  meas.name_ = dist->GetID();
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
