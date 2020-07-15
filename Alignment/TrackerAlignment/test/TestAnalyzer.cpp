// -*- C++ -*-
//
// Package:    TestAnalyzer
// Class:      TestAnalyzer
//
//
// Description: Module to test the Alignment software
//
//
// Original Author:  Frederic Ronga
//         Created:  March 16, 2006
//

// system include files
#include <string>
#include <TTree.h>
#include <TFile.h>
#include <TRotMatrix.h>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//
//
// class declaration
//

class TestAnalyzer : public edm::EDAnalyzer {
public:
  explicit TestAnalyzer(const edm::ParameterSet&);
  ~TestAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  TTree* theTree;
  TFile* theFile;
  float x_, y_, z_, phi_, theta_, length_, thick_, width_;
  int Id_;
  TRotMatrix* rot_;
};

//
// constructors and destructor
//
TestAnalyzer::TestAnalyzer(const edm::ParameterSet& iConfig) {
  // Open root file and define tree
  std::string fileName = iConfig.getUntrackedParameter<std::string>("fileName", "test.root");
  theFile = new TFile(fileName.c_str(), "RECREATE");
  theTree = new TTree("theTree", "Detector units positions");

  theTree->Branch("Id", &Id_, "Id/I");
  theTree->Branch("x", &x_, "x/F");
  theTree->Branch("y", &y_, "y/F");
  theTree->Branch("z", &z_, "z/F");
  theTree->Branch("phi", &phi_, "phi/F");
  theTree->Branch("theta", &theta_, "theta/F");
  theTree->Branch("length", &length_, "length/F");
  theTree->Branch("width", &width_, "width/F");
  theTree->Branch("thick", &thick_, "thick/F");
  rot_ = nullptr;
  theTree->Branch("rot", "TRotMatrix", &rot_);
}

TestAnalyzer::~TestAnalyzer() {
  theTree->Write();
  theFile->Close();
}

void TestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogInfo("TrackerAlignment") << "Starting!";

  //
  // Retrieve tracker geometry from event setup
  //
  edm::ESHandle<TrackerGeometry> trackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);

  // Now loop on detector units, and store position and orientation
  for (auto iGeomDet = trackerGeometry->dets().begin(); iGeomDet != trackerGeometry->dets().end(); iGeomDet++) {
    Id_ = (*iGeomDet)->geographicalId().rawId();
    x_ = (*iGeomDet)->position().x();
    y_ = (*iGeomDet)->position().y();
    z_ = (*iGeomDet)->position().z();
    phi_ = (*iGeomDet)->surface().normalVector().phi();
    theta_ = (*iGeomDet)->surface().normalVector().theta();
    length_ = (*iGeomDet)->surface().bounds().length();
    width_ = (*iGeomDet)->surface().bounds().width();
    thick_ = (*iGeomDet)->surface().bounds().thickness();

    double matrix[9] = {(*iGeomDet)->rotation().xx(),
                        (*iGeomDet)->rotation().xy(),
                        (*iGeomDet)->rotation().xz(),
                        (*iGeomDet)->rotation().yx(),
                        (*iGeomDet)->rotation().yy(),
                        (*iGeomDet)->rotation().yz(),
                        (*iGeomDet)->rotation().zx(),
                        (*iGeomDet)->rotation().zy(),
                        (*iGeomDet)->rotation().zz()};
    rot_ = new TRotMatrix("rot", "rot", matrix);

    theTree->Fill();
  }

  edm::LogInfo("TrackerAlignment") << "Done!";
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestAnalyzer);
