// -*- C++ -*-
//
// Package:    PerfectGeometryAnalyzer
// Class:      PerfectGeometryAnalyzer
// 
/**\class PerfectGeometryAnalyzer PerfectGeometryAnalyzer.cc test/PerfectGeometryAnalyzer/src/PerfectGeometryAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tommaso Boccali
//         Created:  Tue Jul 26 08:47:57 CEST 2005
// $Id: PerfectGeometryAnalyzer.cc,v 1.10 2006/11/11 13:12:15 chrjones Exp $
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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"

#include "DataSvc/RefException.h"
#include "CoralBase/Exception.h"

//
// class decleration
//

class PerfectGeometryAnalyzer : public edm::EDAnalyzer {
public:
  explicit PerfectGeometryAnalyzer( const edm::ParameterSet& );
  ~PerfectGeometryAnalyzer();

  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  std::string label_;
  bool dumpHistory_;
  bool dumpPosInfo_;
  bool dumpSpecs_;
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
PerfectGeometryAnalyzer::PerfectGeometryAnalyzer( const edm::ParameterSet& iConfig ) :
   label_(iConfig.getUntrackedParameter<std::string>("label",""))
{
  dumpHistory_=iConfig.getUntrackedParameter<bool>("dumpGeoHistory");
  dumpPosInfo_=iConfig.getUntrackedParameter<bool>("dumpPosInfo");
  dumpSpecs_=iConfig.getUntrackedParameter<bool>("dumpSpecs");
}


PerfectGeometryAnalyzer::~PerfectGeometryAnalyzer()
{
 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PerfectGeometryAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   std::cout << "Here I am " << std::endl;
   edm::ESHandle<DDCompactView> pDD;
   iSetup.get<IdealGeometryRecord>().get(label_, pDD );
   GeometryInfoDump gidump;
   gidump.dumpInfo( dumpHistory_, dumpSpecs_, dumpPosInfo_, *pDD );
   std::cout << "finished" << std::endl;
}


//define this as a plug-in
DEFINE_FWK_MODULE(PerfectGeometryAnalyzer);
