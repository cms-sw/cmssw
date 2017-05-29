// -*- C++ -*-
//
// Package:    MeasurementTrackerTest
// Class:      MeasurementTrackerTest
// 
/**\class MeasurementTrackerTest MeasurementTrackerTest.cc RecoTracker/MeasurementTrackerTest/src/MeasurementTrackerTest.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Fri Mar 16 13:19:20 CDT 2007
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <RecoTracker/MeasurementDet/interface/MeasurementTracker.h>
#include <RecoTracker/Record/interface/CkfComponentsRecord.h>
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"


//class definition
class MeasurementTrackerTest : public edm::EDAnalyzer {
public:
  explicit MeasurementTrackerTest(const edm::ParameterSet&);
  ~MeasurementTrackerTest();


private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  std::string theMeasurementTrackerName;
  std::string theNavigationSchoolName;
};


MeasurementTrackerTest::MeasurementTrackerTest(const edm::ParameterSet& iConfig): 
   theMeasurementTrackerName(iConfig.getParameter<std::string>("measurementTracker"))
  ,theNavigationSchoolName(iConfig.getParameter<std::string>("navigationSchool")){}

MeasurementTrackerTest::~MeasurementTrackerTest() {}

void MeasurementTrackerTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  //get the measurementtracker
  edm::ESHandle<MeasurementTracker> measurementTracker;
  edm::ESHandle<NavigationSchool>   navSchool;

  iSetup.get<CkfComponentsRecord>().get(theMeasurementTrackerName, measurementTracker);
  iSetup.get<NavigationSchoolRecord>().get(theNavigationSchoolName, navSchool);

}


//define this as a plug-in
DEFINE_FWK_MODULE(MeasurementTrackerTest);
