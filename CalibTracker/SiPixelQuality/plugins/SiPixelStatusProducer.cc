// C++ standard
#include <string>
// ROOT
#include "TMath.h"
// CMSSW FW
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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


SiPixelStatusProducer::SiPixelStatusProducer(const edm::ParameterSet& iConfig,  SiPixelTopoCache const*){

  /* badPixelFEDChannelCollections */
  std::vector<edm::InputTag> badPixelFEDChannelCollectionLabels =
      iConfig.getParameter<edm::ParameterSet>("SiPixelStatusProducerParameters")
          .getParameter<std::vector<edm::InputTag>>("badPixelFEDChannelCollections");
  for (auto& t : badPixelFEDChannelCollectionLabels)
    theBadPixelFEDChannelsTokens_.push_back(consumes<PixelFEDChannelCollection>(t));

  /* pixel clusters */
  fPixelClusterLabel_ = iConfig.getParameter<edm::ParameterSet>("SiPixelStatusProducerParameters")
                            .getUntrackedParameter<edm::InputTag>("pixelClusterLabel");
  fSiPixelClusterToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(fPixelClusterLabel_);

  /* register products */
  produces<SiPixelDetectorStatus, edm::Transition::EndLuminosityBlock>("siPixelStatus");
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


/* helper function */
int SiPixelStatusProducer::indexROC(int irow, int icol, int nROCcolumns) {
  return int(icol + irow * nROCcolumns);
  // generate the folling roc index that is going to map with ROC id as
  // 8  9  10 11 12 13 14 15
  // 0  1  2  3  4  5  6  7
}

DEFINE_FWK_MODULE(SiPixelStatusProducer);
