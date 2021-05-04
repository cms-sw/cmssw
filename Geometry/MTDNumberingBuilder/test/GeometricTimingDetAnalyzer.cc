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
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"

#include "DataFormats/Math/interface/Rounding.h"

// Trivial using definition valid both for DDD and DD4hep

#include "DetectorDescription/DDCMS/interface/DDTranslation.h"
#include "DetectorDescription/DDCMS/interface/DDRotationMatrix.h"

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

private:
  edm::ESGetToken<GeometricTimingDet, IdealGeometryRecord> gtdToken_;
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
  gtdToken_ = esConsumes<GeometricTimingDet, IdealGeometryRecord>();
}

GeometricTimingDetAnalyzer::~GeometricTimingDetAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

using cms_rounding::roundIfNear0;

// ------------ method called to produce the data  ------------
void GeometricTimingDetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::LogInfo("GeometricTimingDetAnalyzer") << "Beginning MTD GeometricTimingDet container dump ";
  //
  // get the GeometricTimingDet
  //
  auto rDD = iSetup.getTransientHandle(gtdToken_);

  if (!rDD.isValid()) {
    edm::LogError("GeometricTimingDetAnalyzer") << "ESTransientHandle<GeometricTimingDet> rDD is not valid!";
    return;
  }

  edm::LogVerbatim("GeometricTimingDetAnalyzer")
      << "\n Top node is  " << rDD.product() << " " << rDD.product()->name() << std::endl;

  const auto& top = rDD.product();
  dumpGeometricTimingDet(top);

  edm::LogVerbatim("GeometricTimingDetAnalyzer") << " SubDetectors and layers:";

  std::vector<const GeometricTimingDet*> det;

  det = rDD->components();
  for (const auto& it : det) {
    dumpGeometricTimingDet(it);

    std::vector<const GeometricTimingDet*> layer;
    layer = it->components();
    for (const auto& lay : layer) {
      dumpGeometricTimingDet(lay);
    }
  }
  det.clear();

  edm::LogVerbatim("GeometricTimingDetAnalyzer") << " Modules:";

  det = rDD->deepComponents();
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

  auto fround = [&](double in) {
    std::stringstream ss;
    ss << std::fixed << std::setw(14) << roundIfNear0(in);
    return ss.str();
  };

  edm::LogVerbatim("GeometricTimingDetAnalyzer").log([&](auto& log) {
    log << "\n---------------------------------------------------------------------------------------\n";
    log << "Module = " << det->name() << " rawId = " << det->geographicalID().rawId()
        << " Sub/side/RR = " << thisDet.mtdSubDetector() << " " << thisDet.mtdSide() << " " << thisDet.mtdRR() << "\n\n"
        << " type  = " << det->type() << "\n"
        << " shape = " << det->shape() << "\n\n"
        << "    radLength " << det->radLength() << "\n"
        << "           xi " << det->xi() << "\n"
        << " PixelROCRows " << det->pixROCRows() << "\n"
        << "   PixROCCols " << det->pixROCCols() << "\n"
        << "   PixelROC_X " << det->pixROCx() << "\n"
        << "   PixelROC_Y " << det->pixROCy() << "\n"
        << " TrackerStereoDetectors " << (det->stereo() ? "true" : "false") << "\n"
        << " SiliconAPVNumber " << det->siliconAPVNum() << "\n\n"
        << " Siblings numbers = ";
    std::vector<int> nv = det->navType();
    for (auto sib : nv) {
      log << sib << ", ";
    }
    log << "\n"
        << " And Contains SubDetectors: " << det->components().size() << "\n"
        << " And Contains Daughters:    " << det->deepComponents().size() << "\n"
        << " Is leaf: " << det->isLeaf() << "\n\n"
        << " Translation = " << fround(trans.X()) << " " << fround(trans.Y()) << " " << fround(trans.Z()) << "\n"
        << " Rotation    = " << fround(x.X()) << " " << fround(x.Y()) << " " << fround(x.Z()) << " " << fround(y.X())
        << " " << fround(y.Y()) << " " << fround(y.Z()) << " " << fround(z.X()) << " " << fround(z.Y()) << " "
        << fround(z.Z()) << "\n"
        << " Phi = " << fround(det->phi()) << " Rho = " << fround(det->rho()) << "\n";
    log << "\n---------------------------------------------------------------------------------------\n";
  });

  edm::LogVerbatim("MTDUnitTest") << det->geographicalID().rawId() << fround(trans.X()) << fround(trans.Y())
                                  << fround(trans.Z()) << fround(x.X()) << fround(x.Y()) << fround(x.Z())
                                  << fround(y.X()) << fround(y.Y()) << fround(y.Z()) << fround(z.X()) << fround(z.Y())
                                  << fround(z.Z());

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
