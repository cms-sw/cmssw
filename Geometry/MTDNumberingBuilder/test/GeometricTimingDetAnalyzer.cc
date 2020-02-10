// -*- C++ -*-
//
// Package:    GeometricDetAnalyzer
// Class:      GeometricDetAnalyzer
//
/**\class GeometricDetAnalyzer GeometricDetAnalyzer.cc test/GeometricDetAnalyzer/src/GeometricDetAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tommaso Boccali
//         Created:  Tue Jul 26 08:47:57 CEST 2005
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"

//
//
// class decleration
//

class GeometricTimingDetAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit GeometricTimingDetAnalyzer(const edm::ParameterSet&);
  ~GeometricTimingDetAnalyzer() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
  void dumpGeometricTimingDet(const GeometricTimingDet* det);
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
GeometricTimingDetAnalyzer::GeometricTimingDetAnalyzer(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
}

GeometricTimingDetAnalyzer::~GeometricTimingDetAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void GeometricTimingDetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::LogInfo("GeometricTimingDetAnalyzer") << "Beginning MTD GeometricTimingDet container dump ";
  //
  // get the GeometricTimingDet
  //
  edm::ESHandle<GeometricTimingDet> rDD;
  iSetup.get<IdealGeometryRecord>().get(rDD);
  edm::LogInfo("GeometricTimingDetAnalyzer")
      << " Top node is  " << rDD.product() << " " << rDD.product()->name() << std::endl;

  const auto& top = rDD.product();
  dumpGeometricTimingDet(top);

  std::vector<const GeometricTimingDet*> det = rDD->deepComponents();
  for (const auto& it : det) {
    dumpGeometricTimingDet(it);
  }
}

void GeometricTimingDetAnalyzer::dumpGeometricTimingDet(const GeometricTimingDet* det) {
  const GeometricTimingDet::Translation& trans = det->translation();

  const GeometricTimingDet::RotationMatrix& res = det->rotation();

  DD3Vector x, y, z;
  res.GetComponents(x, y, z);

  MTDDetId thisDet(det->geographicalID());

  edm::LogVerbatim("GeometricTimingDetAnalyzer").log([&](auto& log) {
    log << "\n---------------------------------------------------------------------------------------\n";
    log << "Module = " << det->name() << " type = " << det->type() << " rawId = " << det->geographicalID().rawId()
        << " Sub/side/RR = " << thisDet.mtdSubDetector() << " " << thisDet.mtdSide() << " " << thisDet.mtdRR() << "\n\n"
        << "      shape = " << det->shape() << "\n"
        << "    radLength " << det->radLength() << "\n"
        << "           xi " << det->xi() << "\n"
        << " PixelROCRows " << det->pixROCRows() << "\n"
        << "   PixROCCols " << det->pixROCCols() << "\n"
        << "   PixelROC_X " << det->pixROCx() << "\n"
        << "   PixelROC_Y " << det->pixROCy() << "\n"
        << "TrackerStereoDetectors " << (det->stereo() ? "true" : "false") << "\n"
        << "SiliconAPVNumber " << det->siliconAPVNum() << "\n"
        << "Siblings numbers = ";
    std::vector<int> nv = det->navType();
    for (auto sib : nv)
      log << sib << ", ";
    log << " And Contains  Daughters: " << det->deepComponents().size() << "\n\n";
    log << "Translation = " << std::fixed << std::setw(14) << trans.X() << " " << std::setw(14) << trans.Y() << " "
        << std::setw(14) << trans.Z() << "\n";
    log << "Rotation    = " << std::fixed << std::setw(14) << x.X() << " " << std::setw(14) << x.Y() << " "
        << std::setw(14) << x.Z() << " " << std::setw(14) << y.X() << " " << std::setw(14) << y.Y() << " "
        << std::setw(14) << y.Z() << " " << std::setw(14) << z.X() << " " << std::setw(14) << z.Y() << " "
        << std::setw(14) << z.Z() << "\n";
    log << "Phi = " << std::fixed << std::setw(14) << det->phi() << " Rho = " << std::fixed << std::setw(14)
        << det->rho() << "\n";
    log << "\n---------------------------------------------------------------------------------------\n";
  });

  DD3Vector colx(x.X(), x.Y(), x.Z());
  DD3Vector coly(y.X(), y.Y(), y.Z());
  DD3Vector colz(z.X(), z.Y(), z.Z());
  DDRotationMatrix result(colx, coly, colz);
  DD3Vector cx, cy, cz;
  result.GetComponents(cx, cy, cz);
  if (cx.Cross(cy).Dot(cz) < 0.5) {
    edm::LogInfo("GeometricTimingDetAnalyzer")
        << "Left Handed Rotation Matrix detected; making it right handed: " << det->name();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(GeometricTimingDetAnalyzer);
