#ifndef CalibTracker_SiPixelQuality_SiPixelStatusProducer_h
#define CalibTracker_SiPixelQuality_SiPixelStatusProducer_h

/**_________________________________________________________________
 *    class:   SiPixelStatusProducer.h
 *       package: CalibTracker/SiPixelQuality
 *          reference : https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface
 *________________________________________________________________**/


// C++ standard
#include <string>
// // CMS FW
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
//
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
//
// // Pixel data format
#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
// Tracker Geo
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
//
// SiPixelTopoFinder
#include "CalibTracker/SiPixelQuality/interface/SiPixelTopoFinder.h"
// SiPixelDetectorStatus
#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"

class SiPixelStatusProducer : 

      public edm::stream::EDProducer<edm::LuminosityBlockSummaryCache<std::vector<SiPixelDetectorStatus>>, 
                                     edm::EndLuminosityBlockProducer,
                                     edm::Accumulator> {
public:

   SiPixelStatusProducer(edm::ParameterSet const& iPSet);
   ~SiPixelStatusProducer();


   static std::shared_ptr<std::vector<SiPixelDetectorStatus>> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, 
                                                                                  edm::EventSetup const&, 
                                                                                  LuminosityBlockContext const*){

                        return std::make_shared<std::vector<SiPixelDetectorStatus>>();
   }

   void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) final;
   //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) final {}

   void accumulate(edm::Event const& iEvent, edm::EventSetup const& iSetup) final;


   void endLuminosityBlockSummary(edm::LuminosityBlock const& iLumi, edm::EventSetup const&, 
                                  std::vector<SiPixelDetectorStatus>* siPixelDetectorStatusVtr) const override;

   static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                               edm::EventSetup const&,
                                               LuminosityBlockContext const* iContext,
                                               std::vector<SiPixelDetectorStatus>*) {
      //Nothing to do
   }

   static void globalEndLuminosityBlockProduce(edm::LuminosityBlock& iLumi,
                                               edm::EventSetup const&,
                                               LuminosityBlockContext const* iContext,
                                               std::vector<SiPixelDetectorStatus> const* siPixelDetectorStatusVtr) {
      //
   }                           

private:

  // Producer production (output collection)
  SiPixelDetectorStatus fDet_;


};

#endif
