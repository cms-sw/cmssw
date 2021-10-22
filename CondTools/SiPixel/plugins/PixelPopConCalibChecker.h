#ifndef CondTools_SiPixel_PixelPopConCalibChecker_H
#define CondTools_SiPixel_PixelPopConCalibChecker_H

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

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CondFormats/DataRecord/interface/SiPixelCalibConfigurationRcd.h"

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
  edm::ESGetToken<SiPixelCalibConfiguration, SiPixelCalibConfigurationRcd> gainCalibToken_;
  std::string _filename;
  int _messageLevel;
};

#endif
