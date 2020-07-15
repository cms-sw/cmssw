// -*- C++ -*-
//
// Package:    TestAlign
// Class:      TestAlign
//
//
// Description: Module to test the Alignment software
//
//
// system include files
#include <string>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "Alignment/MuonAlignment/interface/MuonAlignment.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"

#include "DataFormats/GeometrySurface/interface/Surface.h"

#include <vector>

//
//
// class declaration
//

class TestAlign : public edm::EDAnalyzer {
public:
  explicit TestAlign(const edm::ParameterSet&);

  ~TestAlign() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  //typedef Surface::RotationType    RotationType;
  //typedef Surface::PositionType    PositionType;
};

//
// constructors and destructor
//
TestAlign::TestAlign(const edm::ParameterSet& iConfig) { edm::LogInfo("MuonAlignment") << "Starting!"; }

TestAlign::~TestAlign() { edm::LogInfo("MuonAlignment") << "Ending!"; }

void TestAlign::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Instantiate the helper class
  MuonAlignment align(iSetup);

  // Get the AlignableMuon pointer
  AlignableMuon* theAlignableMuon = align.getAlignableMuon();

  // Apply  alignment
  std::vector<double> displacement;
  displacement.push_back(1.0);
  displacement.push_back(0.0);
  displacement.push_back(0.0);

  std::vector<double> rotation;
  rotation.push_back(0.0);
  rotation.push_back(0.0);
  rotation.push_back(1.64);

  // Loop over DT chamber to apply alignment corrections
  for (const auto& iter : theAlignableMuon->DTChambers()) {
    // Print inital position/orientation
    const align::GlobalPoint& pos_i = iter->globalPosition();
    align::RotationType dir_i = iter->globalRotation();

    std::cout << "Initial pos: x=" << pos_i.x() << ",  y=" << pos_i.y() << ",  z=" << pos_i.z() << std::endl;
    std::cout << "Initial ori: x=" << dir_i.xx() << ",  y=" << dir_i.yy() << ",  z=" << dir_i.zz() << std::endl;

    // Move DT chamber
    DetId detid = iter->geomDetId();
    align.moveAlignableGlobalCoord(detid, displacement, rotation);

    // Print final position/orientation
    const align::GlobalPoint& pos_f = iter->globalPosition();
    align::RotationType dir_f = iter->globalRotation();

    std::cout << "Final pos: x=" << pos_f.x() << ",  y=" << pos_f.y() << ",  z=" << pos_f.z() << std::endl;
    std::cout << "Final ori: x=" << dir_f.xx() << ",  y=" << dir_f.yy() << ",  z=" << dir_f.zz() << std::endl;
    std::cout << "------------------------" << std::endl;
  }

  // Saves to DB
  //  align.saveToDB();
}
//define this as a plug-in
DEFINE_FWK_MODULE(TestAlign);
