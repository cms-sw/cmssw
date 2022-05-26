#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"

#include "RecoLocalFastTime/Records/interface/MTDCPERecord.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDClusterParameterEstimator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class MTDTrackingRecHitProducer : public edm::global::EDProducer<> {
public:
  explicit MTDTrackingRecHitProducer(const edm::ParameterSet& ps);
  ~MTDTrackingRecHitProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const override;

private:
  const edm::EDGetTokenT<FTLClusterCollection> ftlbClusters_;  // collection of barrel digis
  const edm::EDGetTokenT<FTLClusterCollection> ftleClusters_;  // collection of endcap digis

  const edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  const edm::ESGetToken<MTDClusterParameterEstimator, MTDCPERecord> cpeToken_;
};

MTDTrackingRecHitProducer::MTDTrackingRecHitProducer(const edm::ParameterSet& ps)
    : ftlbClusters_(consumes<FTLClusterCollection>(ps.getParameter<edm::InputTag>("barrelClusters"))),
      ftleClusters_(consumes<FTLClusterCollection>(ps.getParameter<edm::InputTag>("endcapClusters"))),
      mtdgeoToken_(esConsumes<MTDGeometry, MTDDigiGeometryRecord>()),
      cpeToken_(esConsumes<MTDClusterParameterEstimator, MTDCPERecord>(edm::ESInputTag("", "MTDCPEBase"))) {
  produces<MTDTrackingDetSetVector>();
}

// Configuration descriptions
void MTDTrackingRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("barrelClusters", edm::InputTag("mtdClusters:FTLBarrel"));
  desc.add<edm::InputTag>("endcapClusters", edm::InputTag("mtdClusters:FTLEndcap"));
  descriptions.add("mtdTrackingRecHitProducer", desc);
}

void MTDTrackingRecHitProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {
  auto const& geom = es.getData(mtdgeoToken_);

  auto const& cpe = es.getData(cpeToken_);

  edm::Handle<FTLClusterCollection> inputBarrel;
  evt.getByToken(ftlbClusters_, inputBarrel);

  edm::Handle<FTLClusterCollection> inputEndcap;
  evt.getByToken(ftleClusters_, inputEndcap);

  std::array<edm::Handle<FTLClusterCollection>, 2> inputHandle{{inputBarrel, inputEndcap}};

  auto outputhits = std::make_unique<MTDTrackingDetSetVector>();
  auto& theoutputhits = *outputhits;

  //---------------------------------------------------------------------------
  //!  Iterate over DetUnits, then over Clusters and invoke the CPE on each,
  //!  and make a RecHit to store the result.
  //---------------------------------------------------------------------------

  for (auto const& theInput : inputHandle) {
    if (!theInput.isValid()) {
      edm::LogWarning("MTDTrackingRecHitProducer") << "MTDTrackingRecHitProducer: Invalid collection";
      continue;
    }
    const edmNew::DetSetVector<FTLCluster>& input = *theInput;

    LogDebug("MTDTrackingRecHitProducer") << "inputCollection " << input.size();
    for (const auto& DSVit : input) {
      unsigned int detid = DSVit.detId();
      DetId detIdObject(detid);
      const auto genericDet = geom.idToDetUnit(detIdObject);
      if (genericDet == nullptr) {
        throw cms::Exception("MTDTrackingRecHitProducer")
            << "GeographicalID: " << std::hex << detid << " is invalid!" << std::dec << std::endl;
      }

      MTDTrackingDetSetVector::FastFiller recHitsOnDet(theoutputhits, detid);

      LogDebug("MTDTrackingRecHitProducer") << "MTD cluster DetId " << detid << " # cluster " << DSVit.size();

      for (const auto& clustIt : DSVit) {
        LogDebug("MTDTrackingRecHitProducer") << "Cluster: size " << clustIt.size() << " " << clustIt.x() << ","
                                              << clustIt.y() << " " << clustIt.energy() << " " << clustIt.time();
        MTDClusterParameterEstimator::ReturnType tuple = cpe.getParameters(clustIt, *genericDet);
        LocalPoint lp(std::get<0>(tuple));
        LocalError le(std::get<1>(tuple));

        // Create a persistent edm::Ref to the cluster
        edm::Ref<edmNew::DetSetVector<FTLCluster>, FTLCluster> cluster = edmNew::makeRefTo(theInput, &clustIt);
        // Make a RecHit and add it to the DetSet
        MTDTrackingRecHit hit(lp, le, *genericDet, cluster);
        LogDebug("MTDTrackingRecHitProducer")
            << "MTD_TRH: " << hit.localPosition().x() << "," << hit.localPosition().y() << " : "
            << hit.localPositionError().xx() << "," << hit.localPositionError().yy() << " : " << hit.time() << " : "
            << hit.timeError();
        // Now save it =================
        recHitsOnDet.push_back(hit);
      }  //  <-- End loop on Clusters
    }    //    <-- End loop on DetUnits
    LogDebug("MTDTrackingRecHitProducer") << "outputCollection " << theoutputhits.size();
  }

  evt.put(std::move(outputhits));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MTDTrackingRecHitProducer);
