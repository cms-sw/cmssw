#include "../interface/CocoaDaqReaderRoot.h"
#include "TFile.h" 
#include "Alignment/CocoaDaq/interface/CocoaDaqRootEvent.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"

#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurements.h"

#include <iostream>

#include "TClonesArray.h"


//----------------------------------------------------------------------
CocoaDaqReaderRoot::CocoaDaqReaderRoot(const std::string& m_inFileName )
{
  if ( ALIUtils::debug >= 3) std::cout << " CocoaDaqReaderRoot opening file: " << m_inFileName << std::endl;
  // Open root file
  theFile = new TFile(m_inFileName.c_str()); 
  if( !theTree ) {
    std::cerr << " CocoaDaqReaderRoot TTree file not found " << m_inFileName << std::endl;
    std::exception();
  }
  
  // Read TTree named "CocoaDaq" in memory.  !! SHOULD BE CALLED Alignment_Cocoa
   theTree = (TTree*)theFile->Get("CocoaDaq");
  //  theTree = (TTree*)theFile->Get("Alignment_Link_Cocoa");
  
  if( !theTree ) {
    std::cerr << " CocoaDaqReaderRoot TTree in file " << m_inFileName << " should be called 'CocoaDaq' " << std::endl;
    std::exception();
  }
  TBranch *branch = theTree->GetBranch("Alignment_Cocoa");

  nev = branch->GetEntries(); // number of entries in Tree
  //if ( ALIUtils::debug >= 2) std::cout << "CocoaDaqReaderRoot::CocoaDaqReaderRoot:  number of entries in Tree " << nev << std::endl;
 
  nextEvent = 0;

  // Event object must be created before setting the branch address
  theEvent = new CocoaDaqRootEvent();

  // link pointer to Tree branch
   theTree->SetBranchAddress("Alignment_Cocoa", &theEvent);  //  !! SHOULD BE CALLED Alignment_Cocoa
  // theTree->SetBranchAddress("Alignment_Link", &theEvent);  //  !! SHOULD BE CALLED Alignment_Cocoa

  CocoaDaqReader::SetDaqReader( this );

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
 
  if ( ALIUtils::debug >= 3) std::cout << "CocoaDaqReaderRoot reading event " << nev << " " << nb << std::endl;
  if( nb == 0 ) return 0; //end of file reached??

  // Every n events, dump one to screen
  int n = 1;
  if(nev%n == 0 &&  ALIUtils::debug >= 3 ) theEvent->DumpIt();
  
  //if ( ALIUtils::debug >= 3) std::cout<<" CocoaDaqReaderRoot::ReadEvent "<< nev <<std::endl;

   if ( ALIUtils::debug >= 3) std::cout<<" CocoaDaqReaderRoot::ReadEvent npos2D "<< theEvent->GetNumPos2D() << " nCOPS " << theEvent->GetNumPosCOPS() << std::endl;
  
  for(int ii=0; ii<theEvent->GetNumPos2D(); ii++) {
    AliDaqPosition2D* pos2D = (AliDaqPosition2D*) theEvent->GetArray_Position2D()->At(ii);
    if ( ALIUtils::debug >= 4) std::cout<<"2D sensor "<<ii<<" has ID = "<<pos2D->GetID()<< std::endl;
      pos2D->DumpIt("2DSENSOR"); 
     measList.push_back( GetMeasFromPosition2D( pos2D ) );
  }
  for(int ii=0; ii<theEvent->GetNumPosCOPS(); ii++) {
    AliDaqPositionCOPS* posCOPS = (AliDaqPositionCOPS*) theEvent->GetArray_PositionCOPS()->At(ii);
    measList.push_back( GetMeasFromPositionCOPS( posCOPS ) );
    if ( ALIUtils::debug >= 4) {
      std::cout<<"COPS sensor "<<ii<<" has ID = "<<posCOPS->GetID()<< std::endl;
      posCOPS->DumpIt("COPS"); 
    }
  }
  for(int ii=0; ii<theEvent->GetNumTilt(); ii++) {
    AliDaqTilt* tilt = (AliDaqTilt*) theEvent->GetArray_Tilt()->At(ii);
    measList.push_back( GetMeasFromTilt( tilt ) );
     if ( ALIUtils::debug >= 4) {
       std::cout<<"TILT sensor "<<ii<<" has ID = "<<tilt->GetID()<< std::endl;
       tilt->DumpIt("TILT"); 
     }
     
  }
  for(int ii=0; ii<theEvent->GetNumDist(); ii++) {
    AliDaqDistance* dist = (AliDaqDistance*) theEvent->GetArray_Dist()->At(ii);
    measList.push_back( GetMeasFromDist( dist ) );
    if ( ALIUtils::debug >= 4) {
      std::cout<<"DIST sensor "<<ii<<" has ID = "<<dist->GetID()<< std::endl;
      dist->DumpIt("DIST"); 
    }
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
  oaParam1.value_ = pos2D->GetX()/100.;
  oaParam1.error_ = pos2D->GetXerror()/100.;
  paramList.push_back(oaParam1);
  
  OpticalAlignParam oaParam2;
  oaParam2.name_ = "V:";
  oaParam2.value_ = pos2D->GetY()/100.;
  oaParam2.error_ = pos2D->GetYerror()/100.;
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
  oaParam1.value_ = posCOPS->GetUp()/100.;
  oaParam1.error_ = posCOPS->GetUpError()/100.;
  paramList.push_back(oaParam1);

  OpticalAlignParam oaParam2;
  oaParam2.name_ = "U:";
  oaParam2.value_ = posCOPS->GetDown()/100.;
  oaParam2.error_ = posCOPS->GetDownError()/100.;
  paramList.push_back(oaParam2);

  OpticalAlignParam oaParam3;
  oaParam3.name_ = "U:";
  oaParam3.value_ = posCOPS->GetRight()/100.;
  oaParam3.error_ = posCOPS->GetRightError()/100.;
  paramList.push_back(oaParam3);

  OpticalAlignParam oaParam4;
  oaParam4.name_ = "U:";
  oaParam4.value_ = posCOPS->GetLeft()/100.;
  oaParam4.error_ = posCOPS->GetLeftError()/100.;
  paramList.push_back(oaParam4);
  
  meas.values_ = paramList;

  return meas;

}

//----------------------------------------------------------------------
OpticalAlignMeasurementInfo CocoaDaqReaderRoot::GetMeasFromTilt( AliDaqTilt* tilt )
{
  OpticalAlignMeasurementInfo meas;
  
  meas.type_ = "TILTMETER";
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
  
  meas.type_ = "DISTANCEMETER";
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
  oaParam.value_ = dist->GetDistance()/100.;
  oaParam.error_ = dist->GetDistanceError()/100.;
  paramList.push_back(oaParam);

  meas.values_ = paramList;

  return meas;
}


//----------------------------------------------------------------------
void CocoaDaqReaderRoot::BuildMeasurementsFromOptAlign( std::vector<OpticalAlignMeasurementInfo>& measList )
{
  if ( ALIUtils::debug >= 3) std::cout << "@@@ CocoaDaqReaderRoot::BuildMeasurementsFromOptAlign " << std::endl;

 //set date and time of current measurement
  //  if( wordlist[0] == "DATE:" ) {
  //   Measurement::setCurrentDate( wordlist ); 
  // } 

  //---------- loop measurements read from ROOT and check for corresponding measurement in Model
  //  ALIint nMeasModel = Model::MeasurementList().size();
  ALIint nMeasRoot = measList.size();
  if(ALIUtils::debug >= 4) {
    std::cout << " Building " << nMeasRoot << " measurements from ROOT file " << std::endl;
  }

  //--- Loop to Measurements in Model and check for corresponding measurement in ROOT
  std::vector< Measurement* >::const_iterator vmcite;
  for( vmcite = Model::MeasurementList().begin();  vmcite != Model::MeasurementList().end(); vmcite++ ) {
    ALIint fcolon = (*vmcite)->name().find(':');
    ALIstring oname = (*vmcite)->name();
    oname = oname.substr(fcolon+1,oname.length());
    
    //---------- loop measurements read from ROOT 
    ALIint ii;
    for(ii = 0; ii < nMeasRoot; ii++) {
      OpticalAlignMeasurementInfo measInfo = measList[ii];
      std::cout << " measurement name ROOT " << measInfo.name_ << " Model= " << (*vmcite)->name() << " short " << oname << std::endl;
      
      if( oname == measInfo.name_ ) {
	//-------- Measurement found, fill data
	//---- Check that type is the same
	if( (*vmcite)->type() != measInfo.type_ ) {
	  std::cerr << "!!! Measurement from ROOT file: type in file is " 
		    <<measInfo.type_ << " and should be " << (*vmcite)->type() << std::endl;
	  exit(1);
	}
	
	std::cout << " NOBJECTS IN MEAS " << (*vmcite)->OptOList().size() << " NMEAS " << Model::MeasurementList().size() << std::endl;
	
	std::vector<OpticalAlignParam> measValues = measInfo.values_;
	
	for( size_t jj= 0; jj < measValues.size(); jj++ ){
	  (*vmcite)->fillData( jj, &(measValues[jj]) );
	}

	std::cout << " NOBJECTS IN MEAS after " << (*vmcite)->OptOList().size() << " NMEAS " << Model::MeasurementList().size()  << std::endl;

	break;
      }
    }
    if (ii==nMeasRoot) {
      std::cerr << "!!! Reading measurement from file: measurement not found! Type in list is "  <<  oname  << std::endl;
      exit(1);
    }
  }

}

