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
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigi.h"

//
// class declaration
//

class SiPixelCalibDigiFilter : public edm::EDFilter {
   public:
      explicit SiPixelCalibDigiFilter(const edm::ParameterSet&);
      ~SiPixelCalibDigiFilter() override;

   private:
      void beginJob() override;
      bool filter(edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;
      
      // ----------member data ---------------------------
      edm::EDGetTokenT<edm::DetSetVector<SiPixelCalibDigi>> tPixelCalibDigi;
};
