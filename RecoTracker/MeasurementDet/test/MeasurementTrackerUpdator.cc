// -*- C++ -*-
//
// Package:    MeasurementTrackerUpdator
// Class:      MeasurementTrackerUpdator
// 
/**\class MeasurementTrackerUpdator MeasurementTrackerUpdator.cc RecoTracker/MeasurementTrackerUpdator/src/MeasurementTrackerUpdator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Fri Mar 16 13:19:20 CDT 2007
// $Id: MeasurementTrackerUpdator.cc,v 1.2 2009/03/04 13:34:26 vlimant Exp $
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
//#include <RecoTracker/Record/interface/MeasurementTrackerRecord.h>

//class definition
class MeasurementTrackerUpdator : public edm::EDAnalyzer {
public:
  explicit MeasurementTrackerUpdator(const edm::ParameterSet&);
  ~MeasurementTrackerUpdator();


private:
  virtual void beginRun(edm::Run & run, const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  std::string theMeasurementTrackerName;
};


MeasurementTrackerUpdator::MeasurementTrackerUpdator(const edm::ParameterSet& iConfig): 
  theMeasurementTrackerName(iConfig.getParameter<std::string>("measurementTrackerName")){}

MeasurementTrackerUpdator::~MeasurementTrackerUpdator() {}

void MeasurementTrackerUpdator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  //get the measurementtracker
  edm::ESHandle<MeasurementTracker> measurementTracker;

  edm::LogInfo("MeasurementTrackerUpdator")<<"from CkfComponentRecord";
  iSetup.get<CkfComponentsRecord>().get(theMeasurementTrackerName, measurementTracker);

  //update it to trigger the possible unpacking so that it is decoupled from the hosting module
  measurementTracker->update(iEvent);

}

void MeasurementTrackerUpdator::beginRun(edm::Run & run, const edm::EventSetup&) {}
void MeasurementTrackerUpdator::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(MeasurementTrackerUpdator);
