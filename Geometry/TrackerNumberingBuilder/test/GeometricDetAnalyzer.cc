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

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// CLHEP Dependency
#include <CLHEP/Vector/ThreeVector.h>

//
//
// class decleration
//

class GeometricDetAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit GeometricDetAnalyzer(const edm::ParameterSet&);
  ~GeometricDetAnalyzer() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
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
GeometricDetAnalyzer::GeometricDetAnalyzer(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
}

GeometricDetAnalyzer::~GeometricDetAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void GeometricDetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::LogInfo("GeometricDetAnalyzer") << "Here I am ";
  //
  // get the GeometricDet
  //
  edm::ESHandle<GeometricDet> pDD;
  iSetup.get<IdealGeometryRecord>().get(pDD);
  edm::LogInfo("GeometricDetAnalyzer") << " Top node is  " << pDD.product();
  edm::LogInfo("GeometricDetAnalyzer") << " And Contains  Daughters: " << pDD->deepComponents().size();
  std::vector<const GeometricDet*> det = pDD->deepComponents();
  for (auto& it : det) {
    const DDRotationMatrix& res = it->rotation();
    DD3Vector x, y, z;
    res.GetComponents(x, y, z);
    DD3Vector colx(x.X(), x.Y(), x.Z());
    DD3Vector coly(y.X(), y.Y(), y.Z());
    DD3Vector colz(z.X(), z.Y(), z.Z());

    DDRotationMatrix result(colx, coly, colz);

    DD3Vector cx, cy, cz;
    result.GetComponents(cx, cy, cz);
    if (cx.Cross(cy).Dot(cz) < 0.5) {
      edm::LogInfo("GeometricDetAnalyzer")
          << "Left Handed Rotation Matrix detected; making it right handed: " << it->name();
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(GeometricDetAnalyzer);
