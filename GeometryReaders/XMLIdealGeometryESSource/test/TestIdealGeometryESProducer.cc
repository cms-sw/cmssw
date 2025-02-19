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
// $Id: TestIdealGeometryESProducer.cc,v 1.3 2010/07/19 16:16:15 case Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

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


//
// class decleration
//

class TestIdealGeometryESProducer : public edm::EDAnalyzer {
public:
  explicit TestIdealGeometryESProducer( const edm::ParameterSet& );
  ~TestIdealGeometryESProducer();

  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
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
TestIdealGeometryESProducer::TestIdealGeometryESProducer( const edm::ParameterSet& iConfig ) 
// :   label_(iConfig.getUntrackedParameter<std::string>("label","")),
//   isMagField_(iConfig.getUntrackedParameter<bool>("isMagField",false))
{
//   dumpHistory_=iConfig.getUntrackedParameter<bool>("dumpGeoHistory");
//   dumpPosInfo_=iConfig.getUntrackedParameter<bool>("dumpPosInfo");
//   dumpSpecs_=iConfig.getUntrackedParameter<bool>("dumpSpecs");
//   if ( isMagField_ ) {
//     label_ = "magfield";
//   }
}


TestIdealGeometryESProducer::~TestIdealGeometryESProducer()
{
 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
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


//define this as a plug-in
DEFINE_FWK_MODULE(TestIdealGeometryESProducer);
