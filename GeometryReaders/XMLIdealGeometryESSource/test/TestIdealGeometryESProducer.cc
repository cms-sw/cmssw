// -*- C++ -*-
//
// Package:    TestIdealGeometryESProducer
// Class:      TestIdealGeometryESProducer
// 
/**\class TestIdealGeometryESProducer TestIdealGeometryESProducer.cc test/TestIdealGeometryESProducer/src/TestIdealGeometryESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Case
//         Created:  Tue Jan 16 2009
//
//

#include <memory>
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "CondFormats/Common/interface/FileBlob.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"

class TestIdealGeometryESProducer : public edm::one::EDAnalyzer<> {
public:
  explicit TestIdealGeometryESProducer( const edm::ParameterSet& );
  ~TestIdealGeometryESProducer() override;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}
};

TestIdealGeometryESProducer::TestIdealGeometryESProducer( const edm::ParameterSet& iConfig ) 
{
}

TestIdealGeometryESProducer::~TestIdealGeometryESProducer()
{
}

void
TestIdealGeometryESProducer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   std::cout << "Here I am " << std::endl;
   edm::ESTransientHandle<DDCompactView> pDD;
   iSetup.get<IdealGeometryRecord>().get(pDD);

   GeometryInfoDump gidump;
   gidump.dumpInfo( true, true, true, *pDD );
   std::cout << "finished" << std::endl;
}

DEFINE_FWK_MODULE(TestIdealGeometryESProducer);
