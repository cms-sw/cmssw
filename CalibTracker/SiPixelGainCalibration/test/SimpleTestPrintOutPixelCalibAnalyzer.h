// -*- C++ -*-
//
// Package:    SimpleTestPrintOutPixelCalibAnalyzer
// Class:      SimpleTestPrintOutPixelCalibAnalyzer
//
/**\class SimpleTestPrintOutPixelCalibAnalyzer CalibTracker/SiPixelGainCalibration/test/SimpleTestPrintOutPixelCalibAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Mon Nov  5 16:56:35 CET 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigi.h"

//
// class decleration
//

class SimpleTestPrintOutPixelCalibAnalyzer : public edm::EDAnalyzer {
public:
  explicit SimpleTestPrintOutPixelCalibAnalyzer(const edm::ParameterSet&);
  ~SimpleTestPrintOutPixelCalibAnalyzer();

private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void printInfo(const edm::Event&,
                         const edm::EventSetup&);  // print method added by Freya, this way the analyzer stays clean
  virtual void endJob();

  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::DetSetVector<SiPixelCalibDigi> > tPixelCalibDigi;
};
