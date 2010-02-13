// PixelPopConCalibChecker.cc
//
// EDAnalyzer to check calibration configuration objects transferred to database
//
// M. Eads
// Aug 2008

#include <iostream>

#include "CondTools/SiPixel/test/PixelPopConCalibChecker.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CondFormats/DataRecord/interface/SiPixelCalibConfigurationRcd.h"
#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"

using namespace std;

//
// constructors and destructor
//
PixelPopConCalibChecker::PixelPopConCalibChecker(const edm::ParameterSet& iConfig)

{
	_filename = iConfig.getParameter<string>("filename");
	_messageLevel = iConfig.getUntrackedParameter("messageLevel", 0);
	if (_messageLevel > 0)
		cout << "********* PixelPopConCalibChecker ************" << endl;

}


PixelPopConCalibChecker::~PixelPopConCalibChecker()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
PixelPopConCalibChecker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   bool isSame = true;
   
   // get the run and event number from the EmptyIOVSource
   if (_messageLevel > 0) {
	   EventID eventID = iEvent.id();
	   cout << "Comparing SiPixelCalibConfiguration object from database for run " << eventID.run()
   	    	<< " to calib.dat file at " << _filename << endl;
   } // if (_messageLevel > 0)
   
   
   // get the calib config object in the database from the event setup
   ESHandle<SiPixelCalibConfiguration> calibES;
   iSetup.get<SiPixelCalibConfigurationRcd>().get(calibES);
   
   // get the calib config object from the calib.dat file
   pos::PixelCalibConfiguration fancyCalib(_filename);
   SiPixelCalibConfiguration *calibFile = new SiPixelCalibConfiguration(fancyCalib);
   
   // check if mode matches
   string modeES = calibES->getCalibrationMode();
   string modeFile = calibFile->getCalibrationMode();
   if (_messageLevel > 1) {
	   cout << "*** Checking calibration mode ***" << endl;
	   cout << "  mode from DB:   " << modeES << endl;
	   cout << "  mode from file: " << modeFile << endl;
   }
   if (modeES != modeFile) {
	   isSame = false;
	   if (_messageLevel > 0) {
		   cout << "Mismatch in calibration mode!" << endl;
		   cout << "  " << modeES << " in database, " << modeFile 
		   	<< " in file" << endl;
	   }
   }
   
   // check if the number of triggers matches
   short ntriggerES = calibES->getNTriggers();
   short ntriggerFile = calibFile->getNTriggers();
   if (_messageLevel > 1) {
	   cout << "*** Checking number of triggers ***" << endl;
	   cout << "  NTriggers from DB:   " << ntriggerES << endl;
	   cout << "  NTriggers from file: " << ntriggerFile << endl;
   }
   if (ntriggerES != ntriggerFile) {
	   isSame = false;
	   if (_messageLevel > 0) {
		   cout << "Mismatch in number of triggers!" << endl;
		   cout << "  " << ntriggerES << " in database, " << ntriggerFile 
		   << " in file" << endl;
	   }
   }

   // check if vcal values match
   vector<short> vcalES = calibES->getVCalValues();
   vector<short> vcalFile = calibFile->getVCalValues();
   if (_messageLevel > 1) {
	   cout << "*** Checking vcal values ***" << endl;
	   cout << "  vcal values from DB:   ";
	   for (vector<short>::const_iterator it = vcalES.begin(); 
	   		it != vcalES.end(); ++it)
		   cout << *it << ", ";
	   cout << endl;
	   cout << "  vcal values from file: ";
	   for (vector<short>::const_iterator it = vcalFile.begin(); 
	   	   		it != vcalFile.end(); ++it)
	   		   cout << *it << ", ";
	   cout << endl;
   }
   if (vcalES != vcalFile) {
	   isSame = false;
	   if (_messageLevel > 0) {
		   cout << "Mismatch in vcal values!" << endl;
	   }
   }

   // check if column values match
   vector<short> colES = calibES->getColumnPattern();
   vector<short> colFile = calibFile->getColumnPattern();
   if (_messageLevel > 1) {
	   cout << "*** Checking column pattern values ***" << endl;
	   cout << "  column pattern from DB:   ";
	   for (vector<short>::const_iterator it = colES.begin(); 
	   		it != colES.end(); ++it)
		   cout << *it << ", ";
	   cout << endl;
	   cout << "  column pattern from file: ";
	   for (vector<short>::const_iterator it = colFile.begin(); 
	   	   		it != colFile.end(); ++it)
	   		   cout << *it << ", ";
	   cout << endl;
   }
   if (colES != colFile) {
	   isSame = false;
	   if (_messageLevel > 0) {
		   cout << "Mismatch in column pattern!" << endl;
	   }
   }

   // check if row values match
   vector<short> rowES = calibES->getRowPattern();
   vector<short> rowFile = calibFile->getRowPattern();
   if (_messageLevel > 1) {
	   cout << "*** Checking row pattern values ***" << endl;
	   cout << "  row pattern from DB:   ";
	   for (vector<short>::const_iterator it = rowES.begin(); 
	   		it != rowES.end(); ++it)
		   cout << *it << ", ";
	   cout << endl;
	   cout << "  row pattern from file: ";
	   for (vector<short>::const_iterator it = rowFile.begin(); 
	   	   		it != rowFile.end(); ++it)
	   		   cout << *it << ", ";
	   cout << endl;
   }
   if (rowES != rowFile) {
	   isSame = false;
	   if (_messageLevel > 0) {
		   cout << "Mismatch in row pattern!" << endl;
	   }
   }

   cout << endl;
   if (isSame) {
	   cout << "*** Calibration configuration in database and file match. Go forth and calibrate." << endl;
   }
   else {
	   cout << "*** WARNING! Calibration configuration is database and file DO NOT match!" << endl;
   }

} // PixelPopConCalibChecker::analyze()

void
//PixelPopConCalibChecker::beginJob(const edm::EventSetup&) 
PixelPopConCalibChecker::beginJob() 
{
	
} // void PixelPopConCalibChecker::beginJob(const edm::EventSetup&)

void
PixelPopConCalibChecker::endJob() 
{

} // void PixelPopConCalibChecker::endJob()


//define this as a plug-in
DEFINE_FWK_MODULE(PixelPopConCalibChecker);
