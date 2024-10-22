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
// $Id: TestAccessGeom.cc,v 1.2 2008/06/26 10:05:09 flucke Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

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

class TestAccessGeom : public edm::one::EDAnalyzer<> {
public:
  explicit TestAccessGeom(const edm::ParameterSet&);
  ~TestAccessGeom() = default;

private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  // ----------member data ---------------------------
  const std::vector<std::string> tkGeomLabels_;
  const std::vector<std::string> dtGeomLabels_;
  const std::vector<std::string> cscGeomLabels_;

  std::vector<edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord>> tkGeoTokens_;
  std::vector<edm::ESGetToken<DTGeometry, MuonGeometryRecord>> dtGeoTokens_;
  std::vector<edm::ESGetToken<CSCGeometry, MuonGeometryRecord>> cscGeoTokens_;
};

//
// constructors and destructor
//
TestAccessGeom::TestAccessGeom(const edm::ParameterSet& iConfig)
    : tkGeomLabels_(iConfig.getParameter<std::vector<std::string>>("TrackerGeomLabels")),
      dtGeomLabels_(iConfig.getParameter<std::vector<std::string>>("DTGeomLabels")),
      cscGeomLabels_(iConfig.getParameter<std::vector<std::string>>("CSCGeomLabels")) {
  //now do what ever initialization is needed

  for (std::vector<std::string>::const_iterator iL = tkGeomLabels_.begin(), iE = tkGeomLabels_.end(); iL != iE; ++iL) {
    auto index = std::distance(tkGeomLabels_.begin(), iL);
    TString label(iL->c_str());
    label.ReplaceAll(" ", "");  // fix for buggy framework
    tkGeoTokens_[index] = esConsumes(edm::ESInputTag("", label.Data()));
  }

  for (std::vector<std::string>::const_iterator iL = dtGeomLabels_.begin(), iE = dtGeomLabels_.end(); iL != iE; ++iL) {
    auto index = std::distance(dtGeomLabels_.begin(), iL);
    TString label(iL->c_str());
    label.ReplaceAll(" ", "");  // fix for buggy framework
    dtGeoTokens_[index] = esConsumes(edm::ESInputTag("", label.Data()));
  }

  for (std::vector<std::string>::const_iterator iL = cscGeomLabels_.begin(), iE = cscGeomLabels_.end(); iL != iE;
       ++iL) {
    auto index = std::distance(cscGeomLabels_.begin(), iL);
    TString label(iL->c_str());
    label.ReplaceAll(" ", "");  // fix for buggy framework
    cscGeoTokens_[index] = esConsumes(edm::ESInputTag("", label.Data()));
  }
}

//
// member functions
//

// ------------ method called to for each event  ------------
void TestAccessGeom::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using std::string;
  using std::vector;

  edm::LogInfo("Test") << "@SUB=analyze"
                       << "Try to  access " << tkGeomLabels_.size() << " Tracker-, " << dtGeomLabels_.size()
                       << " DT- and " << cscGeomLabels_.size() << " CSC-geometries.";

  for (vector<string>::const_iterator iL = tkGeomLabels_.begin(), iE = tkGeomLabels_.end(); iL != iE; ++iL) {
    TString label(iL->c_str());
    label.ReplaceAll(" ", "");  // fix for buggy framework
    edm::LogInfo("Test") << "Try access to tracker geometry with label '" << label << "'.";
    auto idx = std::distance(tkGeomLabels_.begin(), iL);
    //*iL << "'.";
    edm::ESHandle<TrackerGeometry> tkGeomHandle = iSetup.getHandle(tkGeoTokens_[idx]);
    edm::LogInfo("Test") << "TrackerGeometry pointer: " << tkGeomHandle.product();
  }

  for (vector<string>::const_iterator iL = dtGeomLabels_.begin(), iE = dtGeomLabels_.end(); iL != iE; ++iL) {
    TString label(iL->c_str());
    label.ReplaceAll(" ", "");  // fix for buggy framework
    edm::LogInfo("Test") << "Try access to DT geometry with label '" << label << "'.";
    auto idx = std::distance(dtGeomLabels_.begin(), iL);
    //*iL << "'.";
    edm::ESHandle<DTGeometry> dtGeomHandle = iSetup.getHandle(dtGeoTokens_[idx]);
    edm::LogInfo("Test") << "DTGeometry pointer: " << dtGeomHandle.product();
  }

  for (vector<string>::const_iterator iL = cscGeomLabels_.begin(), iE = cscGeomLabels_.end(); iL != iE; ++iL) {
    TString label(iL->c_str());
    label.ReplaceAll(" ", "");  // fix for buggy framework
    edm::LogInfo("Test") << "Try access to CSC geometry with label '" << label << "'.";
    auto idx = std::distance(cscGeomLabels_.begin(), iL);
    //*iL << "'.";
    edm::ESHandle<CSCGeometry> cscGeomHandle = iSetup.getHandle(cscGeoTokens_[idx]);
    edm::LogInfo("Test") << "CSCGeometry pointer: " << cscGeomHandle.product();
  }

  edm::LogInfo("Test") << "@SUB=analyze"
                       << "Succesfully accessed " << tkGeomLabels_.size() << " Tracker-, " << dtGeomLabels_.size()
                       << " DT- and " << cscGeomLabels_.size() << " CSC-geometries.";
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestAccessGeom);
