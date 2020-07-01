// -*- C++ -*-
//
// Package:    TestMisalign
// Class:      TestMisalign
//
//
// Description: Module to test the Alignment software
//
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

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

//
//
// class declaration
//

class TestMisalign : public edm::EDAnalyzer {
public:
  explicit TestMisalign(const edm::ParameterSet&);
  ~TestMisalign() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void fillTree(const GeomDet* geomDet);

  // ----------member data ---------------------------
  TTree* theTree;
  TFile* theFile;
  float x, y, z, phi, theta, length, thick, width;
  int Id_;
  TRotMatrix* rot;
};

//
// constructors and destructor
//
TestMisalign::TestMisalign(const edm::ParameterSet& iConfig) {
  // Open root file and define tree
  std::string fileName = iConfig.getUntrackedParameter<std::string>("fileName", "test.root");
  theFile = new TFile(fileName.c_str(), "RECREATE");
  theTree = new TTree("theTree", "Detector units positions");

  theTree->Branch("Id", &Id_, "Id/I");
  theTree->Branch("x", &x, "x/F");
  theTree->Branch("y", &y, "y/F");
  theTree->Branch("z", &z, "z/F");
  theTree->Branch("phi", &phi, "phi/F");
  theTree->Branch("theta", &theta, "theta/F");
  theTree->Branch("length", &length, "length/F");
  theTree->Branch("width", &width, "width/F");
  theTree->Branch("thick", &thick, "thick/F");
  rot = nullptr;
  theTree->Branch("rot", "TRotMatrix", &rot);
}

TestMisalign::~TestMisalign() {
  theTree->Write();
  theFile->Close();
}

void TestMisalign::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogInfo("MuonAlignment") << "Starting!";

  //
  // Retrieve muon geometry from event setup
  //
  edm::ESHandle<DTGeometry> pDT;
  edm::ESHandle<CSCGeometry> pCSC;

  iSetup.get<MuonGeometryRecord>().get(pDT);
  iSetup.get<MuonGeometryRecord>().get(pCSC);

  // Now loop on detector units, and store position and orientation
  for (auto iGeomDet = pDT->dets().begin(); iGeomDet != pDT->dets().end(); iGeomDet++)
    this->fillTree(*iGeomDet);
  for (auto iGeomDet = pCSC->dets().begin(); iGeomDet != pCSC->dets().end(); iGeomDet++)
    this->fillTree(*iGeomDet);

  edm::LogInfo("MuonAlignment") << "Done!";
}

//__________________________________________________________________________________________________
void TestMisalign::fillTree(const GeomDet* geomDet) {
  Id_ = geomDet->geographicalId().rawId();
  x = geomDet->position().x();
  y = geomDet->position().y();
  z = geomDet->position().z();
  phi = geomDet->surface().normalVector().phi();
  theta = geomDet->surface().normalVector().theta();
  length = geomDet->surface().bounds().length();
  width = geomDet->surface().bounds().width();
  thick = geomDet->surface().bounds().thickness();

  double matrix[9] = {geomDet->rotation().xx(),
                      geomDet->rotation().xy(),
                      geomDet->rotation().xz(),
                      geomDet->rotation().yx(),
                      geomDet->rotation().yy(),
                      geomDet->rotation().yz(),
                      geomDet->rotation().zx(),
                      geomDet->rotation().zy(),
                      geomDet->rotation().zz()};
  rot = new TRotMatrix("rot", "rot", matrix);

  theTree->Fill();
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestMisalign);
