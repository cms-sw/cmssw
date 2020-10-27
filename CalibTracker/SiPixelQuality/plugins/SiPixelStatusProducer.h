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
// SiPixelTopoFinder
#include "CalibTracker/SiPixelQuality/interface/SiPixelTopoFinder.h"
// SiPixelDetectorStatus
#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"

/* Cache to pertain SiPixelTopoFinder */
class SiPixelTopoCache {

public:

  SiPixelTopoCache(edm::ParameterSet const& iPSet);

  std::shared_ptr<SiPixelTopoFinder> getSiPixelTopoFinder(edm::EventSetup const& iEventSetup,
                                                       edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken,
                                                       edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken,
                                                       edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> siPixelFedCablingMapToken) const {

    std::shared_ptr<SiPixelTopoFinder> returnValue;

    m_queue.pushAndWait([&]() {

       if(!this->siPixelFedCablingMapWatcher_.check(iEventSetup)
          && !this->trackerDIGIGeoWatcher_.check(iEventSetup)
          && !this->trackerTopoWatcher_.check(iEventSetup)) {
          /*the condition hasn't changed so we can just use our old value*/
          returnValue = m_mostRecentSiPixelTopoFinder_;
       }
       else {
          /*the condition has changed so we need to update*/
          const TrackerGeometry* trackerGeometry = &iEventSetup.getData(trackerGeometryToken);
          const TrackerTopology* trackerTopology = &iEventSetup.getData(trackerTopologyToken);
          const SiPixelFedCablingMap* cablingMap = &iEventSetup.getData(siPixelFedCablingMapToken);

          /*use a lambda to generate a new PixelTopoFinder only if there isn't an old one available*/
          returnValue = m_holder.makeOrGet( [this]() { return new SiPixelTopoFinder();});

          m_mostRecentSiPixelTopoFinder_->init(trackerGeometry, trackerTopology, cablingMap);
          m_mostRecentSiPixelTopoFinder_ = returnValue;
        }

    });//m_queue

    return returnValue;

  }

private:

  mutable edm::ReusableObjectHolder<SiPixelTopoFinder> m_holder;
  mutable edm::SerialTaskQueue m_queue;

  /*  Condition watchers  */
  /* CablingMaps */
  mutable edm::ESWatcher<SiPixelFedCablingMapRcd> siPixelFedCablingMapWatcher_;
  /* TrackerDIGIGeo */
  mutable edm::ESWatcher<TrackerDigiGeometryRecord> trackerDIGIGeoWatcher_;
  /* TrackerTopology */
  mutable edm::ESWatcher<TrackerTopologyRcd> trackerTopoWatcher_;

  /* SiPixelTopoFinder */
  mutable std::shared_ptr<SiPixelTopoFinder> m_mostRecentSiPixelTopoFinder_;

};

/*|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||*/

class SiPixelStatusProducer : 

      public edm::stream::EDProducer<edm::LuminosityBlockSummaryCache<std::vector<SiPixelDetectorStatus>>, 
                                     edm::EndLuminosityBlockProducer,
                                     edm::Accumulator> {
public:

   SiPixelStatusProducer(edm::ParameterSet const& iPSet);
   ~SiPixelStatusProducer();

   static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

               edm::ParameterSetDescription desc;
               {
                 edm::ParameterSetDescription psd0;
                 psd0.addUntracked<edm::InputTag>("pixelClusterLabel", edm::InputTag("siPixelClusters", "", "RECO"));
                 psd0.add<std::vector<edm::InputTag>>("badPixelFEDChannelCollections",{edm::InputTag("siPixelDigis"),});
                 desc.add<edm::ParameterSetDescription>("SiPixelStatusProducerParameters", psd0);
               }
               descriptions.add("siPixelStatusProducer", desc);
   }


   void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) final;
   //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) final {}

   void accumulate(edm::Event const& iEvent, edm::EventSetup const& iSetup) final;


   void endLuminosityBlockSummary(edm::LuminosityBlock const& iLumi, edm::EventSetup const&, 
                                  std::vector<SiPixelDetectorStatus>* siPixelDetectorStatusVtr) const override;


   static std::shared_ptr<std::vector<SiPixelDetectorStatus>> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                                  edm::EventSetup const&,
                                                                                  LuminosityBlockContext const*){
                        return std::make_shared<std::vector<SiPixelDetectorStatus>>();
   }

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

  virtual int indexROC(int irow, int icol, int nROCcolumns) final;

  /* ParameterSet */
  edm::InputTag fPixelClusterLabel_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> fSiPixelClusterToken_;
  std::vector<edm::EDGetTokenT<PixelFEDChannelCollection>> theBadPixelFEDChannelsTokens_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> siPixelFedCablingMapToken_;

  // Producer production (output collection)
  SiPixelDetectorStatus fDet_;


};

#endif
