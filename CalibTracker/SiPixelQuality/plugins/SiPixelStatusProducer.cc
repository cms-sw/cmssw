// C++ standard
#include <string>
// ROOT
#include "TMath.h"
// CMSSW FW
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Transition.h"
// CMSSW DataFormats
#include "DataFormats/Common/interface/ConditionsInEdm.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
// "FED error 25"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
// CMSSW CondFormats
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

// Header file
#include "CalibTracker/SiPixelQuality/plugins/SiPixelStatusProducer.h"



SiPixelStatusProducer::SiPixelStatusProducer(edm::ParameterSet const& iPSet){
      
      //we need to register the fact that we will make the BeamSpot object
      produces<SiPixelDetectorStatus, edm::Transition::EndLuminosityBlock>();
}

//--------------------------------------------------------------------------------------------------
SiPixelStatusProducer::~SiPixelStatusProducer() {}



void SiPixelStatusProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
        fDet_ = SiPixelDetectorStatus();
}

   //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) final {}

void SiPixelStatusProducer::accumulate(edm::Event const& iEvent, edm::EventSetup const& iSetup) {}


void SiPixelStatusProducer::endLuminosityBlockSummary(edm::LuminosityBlock const& iLumi, edm::EventSetup const&, 
                                  std::vector<SiPixelDetectorStatus>* iPoints) const{
      //add the Stream's partial information to the full information
      iPoints->push_back(fDet_);
      
}




DEFINE_FWK_MODULE(SiPixelStatusProducer);
