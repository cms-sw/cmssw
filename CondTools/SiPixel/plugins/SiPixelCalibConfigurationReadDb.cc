// -*- C++ -*-
//
// Package:    SiPixelCalibConfigurationReadDb
// Class:      SiPixelCalibConfigurationReadDb
//
/**\class SiPixelCalibConfigurationReadDb SiPixelCalibConfigurationReadDb.cc CalibTracker/SiPixelTools/plugins/SiPixelCalibConfigurationReadDb.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Thu Sep 20 12:13:20 CEST 2007
// $Id: SiPixelCalibConfigurationReadDb.cc,v 1.2 2009/02/10 09:27:50 fblekman Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/SiPixelCalibConfigurationRcd.h"
#include <iostream>
//
// class decleration
//

class SiPixelCalibConfigurationReadDb : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelCalibConfigurationReadDb(const edm::ParameterSet&);
  ~SiPixelCalibConfigurationReadDb() override;

private:
  const edm::ESGetToken<SiPixelCalibConfiguration, SiPixelCalibConfigurationRcd> calibConfigToken;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  bool verbose_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelCalibConfigurationReadDb::SiPixelCalibConfigurationReadDb(const edm::ParameterSet& iConfig)
    : calibConfigToken(esConsumes()), verbose_(iConfig.getParameter<bool>("verbosity")) {
  //now do what ever initialization is needed
}

SiPixelCalibConfigurationReadDb::~SiPixelCalibConfigurationReadDb() = default;

//
// member functions
//

// ------------ method called to for each event  ------------
void SiPixelCalibConfigurationReadDb::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  LogInfo("") << " examining SiPixelCalibConfiguration database object..." << std::endl;

  const SiPixelCalibConfiguration* calib = &iSetup.getData(calibConfigToken);
  edm::LogPrint("SiPixelCalibConfigurationReadDb") << "calibration type: " << calib->getCalibrationMode() << std::endl;
  edm::LogPrint("SiPixelCalibConfigurationReadDb") << "number of triggers: " << calib->getNTriggers() << std::endl;
  std::vector<short> vcalvalues = calib->getVCalValues();
  edm::LogPrint("SiPixelCalibConfigurationReadDb") << "number of VCAL: " << vcalvalues.size() << std::endl;
  int ngoodcols = 0;
  int ngoodrows = 0;
  for (uint32_t i = 0; i < vcalvalues.size(); ++i) {
    if (verbose_) {
      edm::LogPrint("SiPixelCalibConfigurationReadDb")
          << "Vcal values " << i << "," << i + 1 << " : " << vcalvalues[i] << ",";
    }
    ++i;
    if (verbose_) {
      if (i < vcalvalues.size())
        edm::LogPrint("SiPixelCalibConfigurationReadDb") << vcalvalues[i];
      edm::LogPrint("SiPixelCalibConfigurationReadDb") << std::endl;
    }
  }
  if (verbose_)
    edm::LogPrint("SiPixelCalibConfigurationReadDb") << "column patterns:" << std::endl;
  for (uint32_t i = 0; i < calib->getColumnPattern().size(); ++i) {
    if (calib->getColumnPattern()[i] != -1) {
      if (verbose_)
        edm::LogPrint("SiPixelCalibConfigurationReadDb") << calib->getColumnPattern()[i];
      ngoodcols++;
    }
    if (verbose_) {
      if (i != 0)
        edm::LogPrint("SiPixelCalibConfigurationReadDb") << " ";
      if (calib->getColumnPattern()[i] == -1)
        edm::LogPrint("SiPixelCalibConfigurationReadDb") << "- ";
    }
  }
  if (verbose_) {
    edm::LogPrint("SiPixelCalibConfigurationReadDb") << std::endl;
    edm::LogPrint("SiPixelCalibConfigurationReadDb") << "row patterns:" << std::endl;
  }
  for (uint32_t i = 0; i < calib->getRowPattern().size(); ++i) {
    if (calib->getRowPattern()[i] != -1) {
      if (verbose_)
        edm::LogPrint("SiPixelCalibConfigurationReadDb") << calib->getRowPattern()[i];
      ngoodrows++;
    }
    if (verbose_) {
      if (i != 0)
        edm::LogPrint("SiPixelCalibConfigurationReadDb") << " ";
      if (calib->getRowPattern()[i] == -1)
        edm::LogPrint("SiPixelCalibConfigurationReadDb") << "- ";
    }
  }
  if (verbose_) {
    edm::LogPrint("SiPixelCalibConfigurationReadDb") << std::endl;
    edm::LogPrint("SiPixelCalibConfigurationReadDb") << "number of row patterns: " << ngoodrows << std::endl;
    edm::LogPrint("SiPixelCalibConfigurationReadDb") << "number of column patterns: " << ngoodcols << std::endl;
  }
  edm::LogPrint("SiPixelCalibConfigurationReadDb")
      << "this payload is designed to run on " << vcalvalues.size() * ngoodcols * ngoodrows * calib->getNTriggers()
      << " events." << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelCalibConfigurationReadDb);
