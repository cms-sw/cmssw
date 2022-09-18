/** \file GEMCSCSegmentProducer.cc
 *
 */

#include <RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegmentProducer.h>
#include <RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegmentBuilder.h>

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DataFormats/GEMRecHit/interface/GEMRecHitCollection.h>
#include <DataFormats/GEMRecHit/interface/GEMCSCSegmentCollection.h>
#include <DataFormats/GEMRecHit/interface/GEMCSCSegment.h>

#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>

GEMCSCSegmentProducer::GEMCSCSegmentProducer(const edm::ParameterSet& pas)
    : kCSCGeometryToken_(esConsumes<CSCGeometry, MuonGeometryRecord>()),
      kGEMGeometryToken_(esConsumes<GEMGeometry, MuonGeometryRecord>()),
      kCSCSegmentCollectionToken_(consumes<CSCSegmentCollection>(pas.getParameter<edm::InputTag>("inputObjectsCSC"))),
      kGEMRecHitCollectionToken_(consumes<GEMRecHitCollection>(pas.getParameter<edm::InputTag>("inputObjectsGEM"))),
      iev(0) {
  segmentBuilder_ = new GEMCSCSegmentBuilder(pas);  // pass on the parameterset

  // register what this produces
  produces<GEMCSCSegmentCollection>();
}

GEMCSCSegmentProducer::~GEMCSCSegmentProducer() {
  LogDebug("GEMCSCSegment") << "deleting GEMCSCSegmentBuilder after " << iev << " events w/ gem and csc data.";
  delete segmentBuilder_;
}

void GEMCSCSegmentProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // gemcscSegments
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputObjectsGEM", edm::InputTag("gemRecHits"));
  desc.add<edm::InputTag>("inputObjectsCSC", edm::InputTag("cscSegments"));
  desc.add<bool>("enableME21GE21", false);
  desc.add<std::string>("algo_name", "GEMCSCSegAlgoRR");
  {
    edm::ParameterSetDescription psd0;
    psd0.addUntracked<bool>("GEMCSCDebug", true);
    psd0.add<unsigned int>("minHitsPerSegment", 2);
    psd0.add<bool>("preClustering", true);
    psd0.add<double>("dXclusBoxMax", 1.0);
    psd0.add<double>("dYclusBoxMax", 5.0);
    psd0.add<bool>("preClusteringUseChaining", true);
    psd0.add<double>("dPhiChainBoxMax", 1.0);
    psd0.add<double>("dThetaChainBoxMax", 0.02);
    psd0.add<double>("dRChainBoxMax", 0.5);
    psd0.add<int>("maxRecHitsInCluster", 6);
    desc.add<edm::ParameterSetDescription>("algo_psets", psd0);
  }
  descriptions.add("gemcscSegments", desc);
}

void GEMCSCSegmentProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {
  LogDebug("GEMCSCSegment") << "start producing segments for " << ++iev << "th event w/ gem and csc data";

  // find the geometry (& conditions?) for this event & cache it in the builder
  const auto cscg = setup.getHandle(kCSCGeometryToken_);
  if (not cscg.isValid()) {
    edm::LogError("GEMCSCSegment") << "invalid CSCGeometry";
    return;
  }
  const CSCGeometry* cgeom = &*cscg;

  const auto gemg = setup.getHandle(kGEMGeometryToken_);
  if (not gemg.isValid()) {
    edm::LogError("GEMCSCSegment") << "invalid GEMGeometry";
    return;
  }
  const GEMGeometry* ggeom = &*gemg;

  // cache the geometry in the builder
  segmentBuilder_->setGeometry(ggeom, cgeom);

  // fill the map with matches between GEM and CSC chambers
  segmentBuilder_->LinkGEMRollsToCSCChamberIndex(ggeom, cgeom);

  // get the collection of CSCSegment and GEMRecHits
  const auto cscSegment = ev.getHandle(kCSCSegmentCollectionToken_);
  if (not cscSegment.isValid()) {
    edm::LogError("GEMCSCSegment") << "invalid CSCSegmentCollection";
    return;
  }

  const auto gemRecHits = ev.getHandle(kGEMRecHitCollectionToken_);
  if (not gemRecHits.isValid()) {
    edm::LogError("GEMCSCSegment") << "invalid GEMRecHitCollection";
    return;
  }

  // create empty collection of GEMCSC Segments
  auto oc = std::make_unique<GEMCSCSegmentCollection>();

  // pass the empty collection of GEMCSC Segments and fill it
  segmentBuilder_->build(gemRecHits.product(), cscSegment.product(), *oc);  //@@ FILL oc

  // put the filled collection in event
  ev.put(std::move(oc));
}
