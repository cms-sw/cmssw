// -*- C++ -*-
//
// Package:    PixelPopConCalibChecker
// Class:      PixelPopConCalibChecker
//
/**\class PixelPopConCalibChecker PixelPopConCalibChecker.h SiPixel/test/PixelPopConCalibChecker.h

 Description: Test analyzer for checking calib configuration objects written to db

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  M. Eads
//         Created:  August 2008
//
//

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"
#include "CondFormats/DataRecord/interface/SiPixelCalibConfigurationRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//
class PixelPopConCalibChecker : public edm::one::EDAnalyzer<> {
public:
  explicit PixelPopConCalibChecker(const edm::ParameterSet&);
  ~PixelPopConCalibChecker() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  const edm::ESGetToken<SiPixelCalibConfiguration, SiPixelCalibConfigurationRcd> gainCalibToken_;
  const std::string _filename;
  const int _messageLevel;
};

using namespace std;

//
// constructors and destructor
//
PixelPopConCalibChecker::PixelPopConCalibChecker(const edm::ParameterSet& iConfig)
    : gainCalibToken_(esConsumes()),
      _filename(iConfig.getParameter<string>("filename")),
      _messageLevel(iConfig.getUntrackedParameter("messageLevel", 0)) {
  if (_messageLevel > 0)
    edm::LogPrint("PixelPopConCalibChecker") << "********* PixelPopConCalibChecker ************" << endl;
}

PixelPopConCalibChecker::~PixelPopConCalibChecker() = default;

//
// member functions
//

// ------------ method called to for each event  ------------
void PixelPopConCalibChecker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  bool isSame = true;

  // get the run and event number from the EmptyIOVSource
  if (_messageLevel > 0) {
    EventID eventID = iEvent.id();
    edm::LogPrint("PixelPopConCalibChecker") << "Comparing SiPixelCalibConfiguration object from database for run "
                                             << eventID.run() << " to calib.dat file at " << _filename << endl;
  }  // if (_messageLevel > 0)

  // get the calib config object in the database from the event setup
  const SiPixelCalibConfiguration* calibES = &iSetup.getData(gainCalibToken_);

  // get the calib config object from the calib.dat file
  pos::PixelCalibConfiguration fancyCalib(_filename);
  SiPixelCalibConfiguration* calibFile = new SiPixelCalibConfiguration(fancyCalib);

  // check if mode matches
  string modeES = calibES->getCalibrationMode();
  string modeFile = calibFile->getCalibrationMode();
  if (_messageLevel > 1) {
    edm::LogPrint("PixelPopConCalibChecker") << "*** Checking calibration mode ***" << endl;
    edm::LogPrint("PixelPopConCalibChecker") << "  mode from DB:   " << modeES << endl;
    edm::LogPrint("PixelPopConCalibChecker") << "  mode from file: " << modeFile << endl;
  }
  if (modeES != modeFile) {
    isSame = false;
    if (_messageLevel > 0) {
      edm::LogPrint("PixelPopConCalibChecker") << "Mismatch in calibration mode!" << endl;
      edm::LogPrint("PixelPopConCalibChecker") << "  " << modeES << " in database, " << modeFile << " in file" << endl;
    }
  }

  // check if the number of triggers matches
  short ntriggerES = calibES->getNTriggers();
  short ntriggerFile = calibFile->getNTriggers();
  if (_messageLevel > 1) {
    edm::LogPrint("PixelPopConCalibChecker") << "*** Checking number of triggers ***" << endl;
    edm::LogPrint("PixelPopConCalibChecker") << "  NTriggers from DB:   " << ntriggerES << endl;
    edm::LogPrint("PixelPopConCalibChecker") << "  NTriggers from file: " << ntriggerFile << endl;
  }
  if (ntriggerES != ntriggerFile) {
    isSame = false;
    if (_messageLevel > 0) {
      edm::LogPrint("PixelPopConCalibChecker") << "Mismatch in number of triggers!" << endl;
      edm::LogPrint("PixelPopConCalibChecker")
          << "  " << ntriggerES << " in database, " << ntriggerFile << " in file" << endl;
    }
  }

  // check if vcal values match
  vector<short> vcalES = calibES->getVCalValues();
  vector<short> vcalFile = calibFile->getVCalValues();
  if (_messageLevel > 1) {
    edm::LogPrint("PixelPopConCalibChecker") << "*** Checking vcal values ***" << endl;
    edm::LogPrint("PixelPopConCalibChecker") << "  vcal values from DB:   ";
    for (vector<short>::const_iterator it = vcalES.begin(); it != vcalES.end(); ++it)
      edm::LogPrint("PixelPopConCalibChecker") << *it << ", ";
    edm::LogPrint("PixelPopConCalibChecker") << endl;
    edm::LogPrint("PixelPopConCalibChecker") << "  vcal values from file: ";
    for (vector<short>::const_iterator it = vcalFile.begin(); it != vcalFile.end(); ++it)
      edm::LogPrint("PixelPopConCalibChecker") << *it << ", ";
    edm::LogPrint("PixelPopConCalibChecker") << endl;
  }
  if (vcalES != vcalFile) {
    isSame = false;
    if (_messageLevel > 0) {
      edm::LogPrint("PixelPopConCalibChecker") << "Mismatch in vcal values!" << endl;
    }
  }

  // check if column values match
  vector<short> colES = calibES->getColumnPattern();
  vector<short> colFile = calibFile->getColumnPattern();
  if (_messageLevel > 1) {
    edm::LogPrint("PixelPopConCalibChecker") << "*** Checking column pattern values ***" << endl;
    edm::LogPrint("PixelPopConCalibChecker") << "  column pattern from DB:   ";
    for (vector<short>::const_iterator it = colES.begin(); it != colES.end(); ++it)
      edm::LogPrint("PixelPopConCalibChecker") << *it << ", ";
    edm::LogPrint("PixelPopConCalibChecker") << endl;
    edm::LogPrint("PixelPopConCalibChecker") << "  column pattern from file: ";
    for (vector<short>::const_iterator it = colFile.begin(); it != colFile.end(); ++it)
      edm::LogPrint("PixelPopConCalibChecker") << *it << ", ";
    edm::LogPrint("PixelPopConCalibChecker") << endl;
  }
  if (colES != colFile) {
    isSame = false;
    if (_messageLevel > 0) {
      edm::LogPrint("PixelPopConCalibChecker") << "Mismatch in column pattern!" << endl;
    }
  }

  // check if row values match
  vector<short> rowES = calibES->getRowPattern();
  vector<short> rowFile = calibFile->getRowPattern();
  if (_messageLevel > 1) {
    edm::LogPrint("PixelPopConCalibChecker") << "*** Checking row pattern values ***" << endl;
    edm::LogPrint("PixelPopConCalibChecker") << "  row pattern from DB:   ";
    for (vector<short>::const_iterator it = rowES.begin(); it != rowES.end(); ++it)
      edm::LogPrint("PixelPopConCalibChecker") << *it << ", ";
    edm::LogPrint("PixelPopConCalibChecker") << endl;
    edm::LogPrint("PixelPopConCalibChecker") << "  row pattern from file: ";
    for (vector<short>::const_iterator it = rowFile.begin(); it != rowFile.end(); ++it)
      edm::LogPrint("PixelPopConCalibChecker") << *it << ", ";
    edm::LogPrint("PixelPopConCalibChecker") << endl;
  }
  if (rowES != rowFile) {
    isSame = false;
    if (_messageLevel > 0) {
      edm::LogPrint("PixelPopConCalibChecker") << "Mismatch in row pattern!" << endl;
    }
  }

  edm::LogPrint("PixelPopConCalibChecker") << endl;
  if (isSame) {
    edm::LogPrint("PixelPopConCalibChecker")
        << "*** Calibration configuration in database and file match. Go forth and calibrate." << endl;
  } else {
    edm::LogPrint("PixelPopConCalibChecker")
        << "*** WARNING! Calibration configuration is database and file DO NOT match!" << endl;
  }

}  // PixelPopConCalibChecker::analyze()

//define this as a plug-in
DEFINE_FWK_MODULE(PixelPopConCalibChecker);
