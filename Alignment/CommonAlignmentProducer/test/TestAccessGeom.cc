// -*- C++ -*-
//
// Package:    TestAccessGeom
// Class:      TestAccessGeom
// 
/**\class TestAccessGeom Alignment/CommonAlignmentProducer/test/TestAccessGeom.cc

 Description: <one line class summary>

 Implementation:
 Module accessing tracking geometries for tracker, DT and CSC
*/
//
// Original Author:  Gero Flucke
//         Created:  Sat Feb 16 20:56:04 CET 2008
// $Id: TestAccessGeom.cc,v 1.3 2010/01/05 11:22:37 mussgill Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  
#include "Geometry/DTGeometry/interface/DTGeometry.h"  
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"  

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <vector>
#include <string>
#include "TString.h"

//
// class declaration
//

class TestAccessGeom : public edm::EDAnalyzer {
   public:
      explicit TestAccessGeom(const edm::ParameterSet&);
      ~TestAccessGeom();


   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------

      const std::vector<std::string> tkGeomLabels_;
      const std::vector<std::string> dtGeomLabels_;
      const std::vector<std::string> cscGeomLabels_;
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
TestAccessGeom::TestAccessGeom(const edm::ParameterSet& iConfig)
  : tkGeomLabels_(iConfig.getParameter<std::vector<std::string> >("TrackerGeomLabels")),
    dtGeomLabels_(iConfig.getParameter<std::vector<std::string> >("DTGeomLabels")),
    cscGeomLabels_(iConfig.getParameter<std::vector<std::string> >("CSCGeomLabels"))
{
   //now do what ever initialization is needed

}


TestAccessGeom::~TestAccessGeom()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TestAccessGeom::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using std::vector;
  using std::string;

  edm::LogInfo("Test") << "@SUB=analyze" << "Try to  access " << tkGeomLabels_.size() 
		       << " Tracker-, " << dtGeomLabels_.size() << " DT- and "
		       << cscGeomLabels_.size() << " CSC-geometries.";

  for (vector<string>::const_iterator iL = tkGeomLabels_.begin(), iE = tkGeomLabels_.end();
       iL != iE; ++iL) {
    TString label(iL->c_str()); label.ReplaceAll(" ", ""); // fix for buggy framework
    edm::LogInfo("Test") << "Try access to tracker geometry with label '" << label << "'.";
    //*iL << "'.";
    edm::ESHandle<TrackerGeometry> tkGeomHandle;
    iSetup.get<TrackerDigiGeometryRecord>().get(label, tkGeomHandle);// *iL, tkGeomHandle);
    edm::LogInfo("Test") << "TrackerGeometry pointer: " << tkGeomHandle.product();
  }

  for (vector<string>::const_iterator iL = dtGeomLabels_.begin(), iE = dtGeomLabels_.end();
       iL != iE; ++iL) {
    TString label(iL->c_str()); label.ReplaceAll(" ", ""); // fix for buggy framework
    edm::LogInfo("Test") << "Try access to DT geometry with label '" << label << "'.";
    //*iL << "'.";
    edm::ESHandle<DTGeometry> dtGeomHandle;
    iSetup.get<MuonGeometryRecord>().get(label, dtGeomHandle);//*iL, dtGeomHandle);
    edm::LogInfo("Test") << "DTGeometry pointer: " << dtGeomHandle.product();
  }

  for (vector<string>::const_iterator iL = cscGeomLabels_.begin(), iE = cscGeomLabels_.end();
       iL != iE; ++iL) {
    TString label(iL->c_str()); label.ReplaceAll(" ", ""); // fix for buggy framework
    edm::LogInfo("Test") << "Try access to CSC geometry with label '" << label << "'.";
    //*iL << "'.";
    edm::ESHandle<CSCGeometry> cscGeomHandle;
    iSetup.get<MuonGeometryRecord>().get(label, cscGeomHandle); //*iL, cscGeomHandle);
    edm::LogInfo("Test") << "CSCGeometry pointer: " << cscGeomHandle.product();
  }


  edm::LogInfo("Test") << "@SUB=analyze" << "Succesfully accessed " << tkGeomLabels_.size() 
                       << " Tracker-, " << dtGeomLabels_.size() << " DT- and "
                       << cscGeomLabels_.size() << " CSC-geometries.";
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestAccessGeom);
