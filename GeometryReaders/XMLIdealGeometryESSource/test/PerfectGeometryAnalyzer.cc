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
// $Id: PerfectGeometryAnalyzer.cc,v 1.13 2007/12/12 09:30:59 muzaffar Exp $
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

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//#include "DataSvc/RefException.h"
//#include "CoralBase/Exception.h"

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
  bool isMagField_;
  bool dumpHistory_;
  bool dumpPosInfo_;
  bool dumpSpecs_;
  std::string fname_;
  int nNodes_;
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
  label_(iConfig.getUntrackedParameter<std::string>("label","")),
  isMagField_(iConfig.getUntrackedParameter<bool>("isMagField",false)),
  dumpHistory_(iConfig.getUntrackedParameter<bool>("dumpGeoHistory",false)),
  dumpPosInfo_(iConfig.getUntrackedParameter<bool>("dumpPosInfo", false)),
  dumpSpecs_(iConfig.getUntrackedParameter<bool>("dumpSpecs", false)),
  fname_(iConfig.getUntrackedParameter<std::string>("outFileName", "GeoHistory")),
  nNodes_(iConfig.getUntrackedParameter<uint32_t>("numNodesToDump", 0))
{
  if ( isMagField_ ) {
    label_ = "magfield";
  }
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
   if (!isMagField_) {
     iSetup.get<IdealGeometryRecord>().get(label_, pDD );
   } else {
     iSetup.get<IdealMagneticFieldRecord>().get(label_, pDD );
   }
   if (pDD.description()) {
     std::cout << pDD.description()->type_ << " label: " << pDD.description()->label_ << std::endl;
   }
   GeometryInfoDump gidump;
   gidump.dumpInfo( dumpHistory_, dumpSpecs_, dumpPosInfo_, *pDD, fname_, nNodes_ );
   std::cout << "finished" << std::endl;
}


//define this as a plug-in
DEFINE_FWK_MODULE(PerfectGeometryAnalyzer);
