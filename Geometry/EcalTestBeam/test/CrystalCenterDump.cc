// -*- C++ -*-
//
// Package:    CrystalCenterDump
// Class:      CrystalCenterDump
//
/**\class CrystalCenterDump CrystalCenterDump.cc test/CrystalCenterDump/src/CrystalCenterDump.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//

// system include files
#include <memory>
#include <cmath>
#include <iomanip>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include <fstream>

#include "CLHEP/Vector/ThreeVector.h"

class CrystalCenterDump : public edm::one::EDAnalyzer<> {
public:
  explicit CrystalCenterDump(const edm::ParameterSet&);
  ~CrystalCenterDump() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  void build(const CaloGeometry& cg, DetId::Detector det, int subdetn, const char* name);
  int pass_;

  double A_;
  double B_;
  double beamEnergy_;

  double crystalDepth() { return A_ * (B_ + log(beamEnergy_)); }

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
};

CrystalCenterDump::CrystalCenterDump(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  pass_ = 0;

  A_ = iConfig.getUntrackedParameter<double>("Afac", 0.89);
  B_ = iConfig.getUntrackedParameter<double>("Bfac", 5.7);
  beamEnergy_ = iConfig.getUntrackedParameter<double>("BeamEnergy", 120.);

  edm::LogVerbatim("CrysInfo") << "Position computed according to the depth " << crystalDepth() << " based on:"
                               << "\n A = " << A_ << " cm "
                               << "\n B = " << B_ << "\n BeamEnergy = " << beamEnergy_ << " GeV";

  geometryToken_ = esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag{});
}

CrystalCenterDump::~CrystalCenterDump() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void CrystalCenterDump::build(const CaloGeometry& cg, DetId::Detector det, int subdetn, const char* name) {
  std::fstream f(name, std::ios_base::out);
  const CaloSubdetectorGeometry* geom(cg.getSubdetectorGeometry(det, subdetn));

  const std::vector<DetId>& ids = geom->getValidDetIds(det, subdetn);
  for (auto id : ids) {
    auto cell = geom->getGeometry(id);
    if (det == DetId::Ecal) {
      if (subdetn == EcalBarrel) {
        EBDetId ebid(id.rawId());
        if (ebid.ism() == 1) {
          float depth = (crystalDepth());
          double crysX = (cell)->getPosition(depth).x();
          double crysY = (cell)->getPosition(depth).y();
          double crysZ = (cell)->getPosition(depth).z();

          CLHEP::Hep3Vector crysPos(crysX, crysY, crysZ);
          double crysEta = crysPos.eta();
          double crysTheta = crysPos.theta();
          double crysPhi = crysPos.phi();

          edm::LogVerbatim("CrysPos") << ebid.ic() << " x = " << crysX << " y = " << crysY << " z = " << crysZ << " \n "
                                      << " eta = " << crysEta << " phi = " << crysPhi << " theta = " << crysTheta;
          f << std::setw(4) << ebid.ic() << " " << std::setw(8) << std::setprecision(6) << crysEta << " "
            << std::setw(8) << std::setprecision(6) << crysPhi << " " << std::endl;
        }
      }
    }
  }
  f.close();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void CrystalCenterDump::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogVerbatim("CrysPos") << "Writing the center (eta,phi) for crystals in barrel SM 1 ";

  const auto& pG = iSetup.getData(geometryToken_);
  //
  // get the ecal & hcal geometry
  //
  if (pass_ == 0) {
    build(pG, DetId::Ecal, EcalBarrel, "BarrelSM1CrystalCenter.dat");
  }

  pass_++;
}

//define this as a plug-in

DEFINE_FWK_MODULE(CrystalCenterDump);
