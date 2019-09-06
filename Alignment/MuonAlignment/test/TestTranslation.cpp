// -*- C++ -*-
//
// Package:    TestTranslation
// Class:      TestTranslation
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

class TestTranslation : public edm::EDAnalyzer {
public:
  explicit TestTranslation(const edm::ParameterSet&);
  ~TestTranslation();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  // ----------member data ---------------------------
  TTree* theTree;
  TFile* theFile;
  float x, y, z, phi, theta, length, thick, width;
  TRotMatrix* dir;

  typedef Surface::RotationType RotationType;
  typedef Surface::PositionType PositionType;

  void apply(Alignable*);
};

//
// constructors and destructor
//
TestTranslation::TestTranslation(const edm::ParameterSet& iConfig) {
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

TestTranslation::~TestTranslation() {
  theTree->Write();
  theFile->Close();
}

void TestTranslation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

  // Apply  alignment
  for (const auto& iter : theAlignableMuon->DTChambers())
    apply(iter);
  for (const auto& iter : theAlignableMuon->CSCEndcaps())
    apply(iter);

  edm::LogInfo("MuonAlignment") << "Done!";
}

void TestTranslation::apply(Alignable* it) {
  std::cout << "------------------------" << std::endl << " BEFORE TRANSLATION " << std::endl;

  align::GlobalPoint pos_i = (it)->globalPosition();
  //          RotationType dir_i  = (it)->globalRotation();

  std::cout << "x=" << pos_i.x() << ",  y=" << pos_i.y() << ",  z=" << pos_i.z() << std::endl;

  double dx = 1.0;
  double dy = 2.0;
  double dz = 3.0;
  align::GlobalVector dr(dx, dy, dz);
  it->move(dr);

  std::cout << "------------------------" << std::endl << " AFTER TRANSLATION " << std::endl;

  align::GlobalPoint pos_f = (it)->globalPosition();
  //          RotationType dir_f = (it)->globalRotation();

  std::cout << "x=" << pos_f.x() << ",  y=" << pos_f.y() << ",  z=" << pos_f.z() << std::endl;

  std::cout << "------------------------" << std::endl;
}

//  delete theAlignableMuon ;

//define this as a plug-in
DEFINE_FWK_MODULE(TestTranslation);
