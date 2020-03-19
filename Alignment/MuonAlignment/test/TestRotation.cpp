// -*- C++ -*-
//
// Package:    TestRotation
// Class:      TestRotation
//
//
// Description: Module to test the Alignment software
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
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include "DataFormats/GeometrySurface/interface/Surface.h"

#include <vector>

//
//
// class declaration
//

class TestRotation : public edm::EDAnalyzer {
public:
  explicit TestRotation(const edm::ParameterSet&);
  ~TestRotation();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  // ----------member data ---------------------------
  TTree* theTree;
  TFile* theFile;
  float x, y, z, phi, theta, length, thick, width;
  TRotMatrix* dir;

  //  typedef Surface::RotationType    RotationType;
  //  typedef Surface::PositionType    PositionType;
};

//
// constructors and destructor
//
TestRotation::TestRotation(const edm::ParameterSet& iConfig) {
  // Open root file and define tree
  std::string fileName = iConfig.getUntrackedParameter<std::string>("fileName", "test.root");
  theFile = new TFile(fileName.c_str(), "RECREATE");
  theTree = new TTree("theTree", "Detector units positions");

  theTree->Branch("x", &x, "x/F");
  theTree->Branch("y", &y, "y/F");
  theTree->Branch("z", &z, "z/F");
  theTree->Branch("phi", &phi, "phi/F");
  theTree->Branch("theta", &theta, "theta/F");
  theTree->Branch("length", &length, "length/F");
  theTree->Branch("width", &width, "width/F");
  theTree->Branch("thick", &thick, "thick/F");
  dir = 0;
  theTree->Branch("dir", "TRotMatrix", &dir);
}

TestRotation::~TestRotation() {
  theTree->Write();
  theFile->Close();
}

void TestRotation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogInfo("MuonAlignment") << "Starting!";

  /*
  //
  // Build alignable muon geometry from event setup
  //

  edm::ESTransientHandle<DDCompactView> cpv;
  iRecord.getRecord<IdealGeometryRecord>().get( cpv );

  DTGeometryBuilderFromDDD  DTGeometryBuilder;
  CSCGeometryBuilderFromDDD CSCGeometryBuilder;

  theDTGeometry   = std::shared_ptr<DTGeometry>( DTGeometryBuilder.build( &(*cpv) ) );
  theCSCGeometry  = std::shared_ptr<CSCGeometry>( CSCGeometryBuilder.build( &(*cpv) ) );

  AlignableMuon* theAlignableMuon = new AlignableMuon( &(*theDTGeometry) , &(*theCSCGeometry) );
*/

  //
  // Retrieve ideal geometry from event setup
  //
  edm::ESHandle<DTGeometry> dtGeometry;
  edm::ESHandle<CSCGeometry> cscGeometry;
  iSetup.get<MuonGeometryRecord>().get(dtGeometry);
  iSetup.get<MuonGeometryRecord>().get(cscGeometry);

  AlignableMuon* theAlignableMuon = new AlignableMuon(&(*dtGeometry), &(*cscGeometry));

  // Apply alignment

  const auto& theDTWheels = theAlignableMuon->DTWheels();
  std::cout << "Number of wheels=" << theDTWheels.size() << std::endl;

  for (const auto& iter : theDTWheels) {
    std::cout << "------------------------" << std::endl << " BEFORE ROTATION " << std::endl;

    align::GlobalPoint pos_i = iter->globalPosition();
    align::RotationType dir_i = iter->globalRotation();

    std::cout << "x=" << pos_i.x() << ",  y=" << pos_i.y() << ",  z=" << pos_i.z() << std::endl;
    std::cout << "xx=" << dir_i.xx() << ",  yx=" << dir_i.yx() << ",  zx=" << dir_i.zx() << std::endl;
    std::cout << "xy=" << dir_i.xy() << ",  yy=" << dir_i.yy() << ",  zy=" << dir_i.zy() << std::endl;
    std::cout << "xz=" << dir_i.xz() << ",  yz=" << dir_i.yz() << ",  zz=" << dir_i.zz() << std::endl;

    // x      = iter->surface().position().x();
    // y      = iter->surface().position().y();
    // z      = iter->surface().position().z();
    // std::cout << "X=" << x << ", Y= " <<  y << ", Z=" << z  << std::endl ;

    double deltaPhi = 3.1415926 / 180 * 45;

    iter->rotateAroundGlobalZ(deltaPhi);

    std::cout << "------------------------" << std::endl << " AFTER ROTATION " << std::endl;

    align::GlobalPoint pos_f = iter->globalPosition();
    align::RotationType dir_f = iter->globalRotation();

    std::cout << "x=" << pos_f.x() << ",  y=" << pos_f.y() << ",  z=" << pos_f.z() << std::endl;
    std::cout << "xx=" << dir_f.xx() << ",  yx=" << dir_f.yx() << ",  zx=" << dir_f.zx() << std::endl;
    std::cout << "xy=" << dir_f.xy() << ",  yy=" << dir_f.yy() << ",  zy=" << dir_f.zy() << std::endl;
    std::cout << "xz=" << dir_f.xz() << ",  yz=" << dir_f.yz() << ",  zz=" << dir_f.zz() << std::endl;
    std::cout << "------------------------" << std::endl;
  }

  //  delete theAlignableMuon ;

  edm::LogInfo("MuonAlignment") << "Done!";
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestRotation);
