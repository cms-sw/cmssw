/**_________________________________________________________________
class:   AlcaPCCEventProducer.cc

authors: Sam Higginbotham (shigginb@cern.ch), Chris Palmer (capalmer@cern.ch), Attila Radl (attila.radl@cern.ch)
________________________________________________________________**/

// C++ standard
#include <string>

// CMS
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Luminosity/interface/PixelClusterCountsInEvent.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "TMath.h"

//The class
class AlcaPCCEventProducer : public edm::global::EDProducer<> {
public:
  explicit AlcaPCCEventProducer(const edm::ParameterSet&);
  ~AlcaPCCEventProducer() override = default;
  void produce(edm::StreamID id, edm::Event& e, edm::EventSetup const& c) const final;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::InputTag pixelClusterLabel_;
  const std::string trigstring_;  //specifies the trigger Rand or ZeroBias
  const bool savePerROCInfo_;     // save per ROC data (important for the special fills)
  const edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelToken_;

  static constexpr int rowsperroc = 52;
  static constexpr int colsperroc = 80;
  static constexpr int nROCcolumns = 8;
};

//--------------------------------------------------------------------------------------------------
AlcaPCCEventProducer::AlcaPCCEventProducer(const edm::ParameterSet& iConfig)
    : pixelClusterLabel_(iConfig.getParameter<edm::InputTag>("pixelClusterLabel")),
      trigstring_(iConfig.getUntrackedParameter<std::string>("trigstring", "alcaPCCEvent")),
      savePerROCInfo_(iConfig.getParameter<bool>("savePerROCInfo")),
      pixelToken_(consumes<edmNew::DetSetVector<SiPixelCluster> >(pixelClusterLabel_)) {
  produces<reco::PixelClusterCountsInEvent, edm::Transition::Event>(trigstring_);
}

//--------------------------------------------------------------------------------------------------
void AlcaPCCEventProducer::produce(edm::StreamID id, edm::Event& iEvent, edm::EventSetup const& iSetup) const {
  std::unique_ptr<reco::PixelClusterCountsInEvent> thePCCob = std::make_unique<reco::PixelClusterCountsInEvent>();
  unsigned int bx = iEvent.bunchCrossing();

  //Looping over the clusters and adding the counts up
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > hClusterColl;
  iEvent.getByToken(pixelToken_, hClusterColl);

  const edmNew::DetSetVector<SiPixelCluster>& clustColl = *(hClusterColl.product());
  // ----------------------------------------------------------------------
  // -- Clusters without tracks
  for (auto const& mod : clustColl) {
    if (mod.empty()) {
      continue;
    }
    DetId detId = mod.id();

    if (savePerROCInfo_) {
      // Iterate over Clusters in module to fill per ROC histogram
      for (auto const& cluster : mod) {
        for (int i = 0; i < cluster.size(); ++i) {
          const auto pix = cluster.pixel(i);
          // TODO: add roc threshold to config if(di.adc > fRocThreshold_) {
          if (pix.adc > 0) {
            int irow = pix.x / rowsperroc; /* constant column direction is along x-axis */
            int icol = pix.y / colsperroc; /* constant row direction is along y-axis */
            /* generate the folling roc index that is going to map with ROC id as
            8  9  10 11 12 13 14 15
            0  1  2  3  4  5  6  7 */
            int key = icol + irow * nROCcolumns;
            thePCCob->incrementRoc(((detId << 7) + key), 1);
          }
        }
      }
    }

    int nCluster = mod.size();
    thePCCob->increment(detId(), nCluster);
    thePCCob->setbxID(bx);
  }

  iEvent.put(std::move(thePCCob), std::string(trigstring_));
}

//--------------------------------------------------------------------------------------------------
void AlcaPCCEventProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription evtParamDesc;
  evtParamDesc.add<edm::InputTag>("pixelClusterLabel", edm::InputTag("siPixelClustersForLumi"));
  evtParamDesc.addUntracked<std::string>("trigstring", "alcaPCCEvent");
  evtParamDesc.add<bool>("savePerROCInfo", true);
  descriptions.add("alcaPCCEventProducer", evtParamDesc);
}

DEFINE_FWK_MODULE(AlcaPCCEventProducer);
