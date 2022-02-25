// -*- C++ -*-
//
// Package:    SiPixelCalibDigiFilter
// Class:      SiPixelCalibDigiFilter
//
/**\class SiPixelCalibDigiFilter SiPixelCalibDigiFilter.cc CalibTracker/SiPixelTools/src/SiPixelCalibDigiFilter.cc

 Description: Filters events that contain no information after the digis are collected into patterns by SiPixelCalibDigiProducer

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Evan Klose Friis
//         Created:  Tue Nov  6 16:59:50 CET 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigi.h"

class SiPixelCalibDigiFilter : public edm::stream::EDFilter<> {
public:
  explicit SiPixelCalibDigiFilter(const edm::ParameterSet&);
  ~SiPixelCalibDigiFilter() override = default;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::DetSetVector<SiPixelCalibDigi>> tPixelCalibDigi;
};

//
// constructors and destructor
//
SiPixelCalibDigiFilter::SiPixelCalibDigiFilter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  tPixelCalibDigi = consumes<edm::DetSetVector<SiPixelCalibDigi>>(edm::InputTag("SiPixelCalibDigiProducer"));
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool SiPixelCalibDigiFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  Handle<DetSetVector<SiPixelCalibDigi>> listOfDetIds;
  iEvent.getByToken(tPixelCalibDigi, listOfDetIds);

  if (listOfDetIds->empty()) {
    return false;
  } else {
    return true;
  }
}

// -- define this as a plug-in
DEFINE_FWK_MODULE(SiPixelCalibDigiFilter);
