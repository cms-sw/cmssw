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
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigi.h"

//
// class decleration
//

class SimpleTestPrintOutPixelCalibAnalyzer : public edm::global::EDAnalyzer<> {
public:
  explicit SimpleTestPrintOutPixelCalibAnalyzer(const edm::ParameterSet&);
  ~SimpleTestPrintOutPixelCalibAnalyzer() = default;

private:
  virtual void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override;
  virtual void printInfo(const edm::Event&, const edm::EventSetup&)
      const;  // print method added by Freya, this way the analyzer stays clean
  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::DetSetVector<SiPixelCalibDigi> > tPixelCalibDigi;
};
