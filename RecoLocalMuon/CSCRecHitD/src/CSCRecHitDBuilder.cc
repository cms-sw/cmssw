// This is CSCRecHitDBuilder.cc

#include <RecoLocalMuon/CSCRecHitD/src/CSCRecHitDBuilder.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCHitFromStripOnly.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCHitFromWireOnly.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCMake2DRecHit.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCWireHitCollection.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHitCollection.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCRangeMapForRecHit.h>

#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <CondFormats/CSCObjects/interface/CSCDBGains.h>
#include <CondFormats/DataRecord/interface/CSCDBGainsRcd.h>
#include <CondFormats/CSCObjects/interface/CSCDBCrosstalk.h>
#include <CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h>
#include <CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h>
#include <CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>
#include <cassert>

CSCRecHitDBuilder::CSCRecHitDBuilder(const edm::ParameterSet& ps) : geom_(nullptr) {
  // Receives ParameterSet percolated down from EDProducer

  useCalib = ps.getParameter<bool>("CSCUseCalibrations");
  stripWireDeltaT = ps.getParameter<int>("CSCstripWireDeltaTime");

  hitsFromStripOnly_ = new CSCHitFromStripOnly(ps);
  hitsFromWireOnly_ = new CSCHitFromWireOnly(ps);
  make2DHits_ = new CSCMake2DRecHit(ps);
}

CSCRecHitDBuilder::~CSCRecHitDBuilder() {
  delete hitsFromStripOnly_;
  delete hitsFromWireOnly_;
  delete make2DHits_;
}

void CSCRecHitDBuilder::build(const CSCStripDigiCollection* stripdc,
                              const CSCWireDigiCollection* wiredc,
                              CSCRecHit2DCollection& oc) {
  LogTrace("CSCRecHitDBuilder") << "[CSCRecHitDBuilder] build entered";

  if (!geom_)
    throw cms::Exception("MissingGeometry") << "[CSCRecHitDBuilder::getLayer] Missing geometry" << std::endl;

  //  create 2-D hits by looking at superposition of strip and wire hit in a layer
  //
  // N.B.  I've sorted the hits from layer 1-6 always, so can test if there are "holes",
  // that is layers without hits for a given chamber.

  int layer_idx = 0;
  CSCDetId old_id;

  for (CSCStripDigiCollection::DigiRangeIterator it = stripdc->begin(); it != stripdc->end(); ++it) {
    const CSCDetId& id = (*it).first;
    const CSCLayer* layer = getLayer(id);
    const CSCStripDigiCollection::Range& rstripd = (*it).second;

    // Skip if no strip digis in this layer
    if (rstripd.second == rstripd.first)
      continue;

    const CSCDetId& sDetId = id;

    // This is used to test for gaps in layers and needs to be initialized here
    if (layer_idx == 0) {
      old_id = sDetId;
    }

    CSCDetId compId = sDetId;
    CSCWireDigiCollection::Range rwired = wiredc->get(sDetId);
    // Skip if no wire digis in this layer
    // But for ME11, real wire digis are labelled as belonging to ME1b, so that's where ME1a must look
    // (We try ME1a - above - anyway, because simulated wire digis are labelled as ME1a.)
    if (rwired.second == rwired.first) {
      if (sDetId.station() != 1 || sDetId.ring() != 4) {
        continue;  // not ME1a, skip to next layer
      }
      // So if ME1a has no wire digis (always the case for data) make the
      // wire digi ID point to ME1b. This is what is compared to the
      // strip digi ID below (and not used anywhere else).
      // Later, rechits use the strip digi ID for construction.

      // It is ME1a but no wire digis there, so try ME1b...
      int endcap = sDetId.endcap();
      int chamber = sDetId.chamber();
      int layer = sDetId.layer();
      CSCDetId idw(endcap, 1, 1, chamber, layer);  // Set idw to same layer in ME1b
      compId = idw;
      rwired = wiredc->get(compId);
    }

    // Fill bad channel bitsets for this layer
    recoConditions_->fillBadChannelWords(id);

    // Build strip hits for this layer
    std::vector<CSCStripHit> const& cscStripHit = hitsFromStripOnly_->runStrip(id, layer, rstripd);

    if (cscStripHit.empty())
      continue;

    // now build collection of wire only hits !
    std::vector<CSCWireHit> const& cscWireHit = hitsFromWireOnly_->runWire(compId, layer, rwired);

    // Build 2D hit for all possible strip-wire pairs
    // overlapping within this layer

    LogTrace("CSCRecHitBuilder") << "[CSCRecHitDBuilder] found " << cscStripHit.size() << " strip and "
                                 << cscWireHit.size() << " wire hits in layer " << sDetId;

    // Vector to store rechit within layer
    std::vector<CSCRecHit2D> hitsInLayer;
    unsigned int hits_in_layer = 0;

    for (auto const& s_hit : cscStripHit) {
      for (auto const& w_hit : cscWireHit) {
        CSCRecHit2D rechit = make2DHits_->hitFromStripAndWire(sDetId, layer, w_hit, s_hit);
        // Store rechit as a Local Point:
        LocalPoint rhitlocal = rechit.localPosition();
        float yreco = rhitlocal.y();
        bool isInFiducial = false;
        //in me1/1 chambers the strip cut region is at local y = 30 cm, +-5 cm area around it proved to be a suitabla region for omiting the check
        if ((sDetId.station() == 1) && (sDetId.ring() == 1 || sDetId.ring() == 4) && (fabs(yreco + 30.) < 5.)) {
          isInFiducial = true;
        } else {
          isInFiducial = make2DHits_->isHitInFiducial(layer, rechit);
        }
        if (isInFiducial) {
          hitsInLayer.push_back(rechit);
          hits_in_layer++;
        }
      }
    }

    LogTrace("CSCRecHitDBuilder") << "[CSCRecHitDBuilder] " << hits_in_layer << " rechits found in layer " << sDetId;

    // output vector of 2D rechits to collection
    if (hits_in_layer > 0) {
      oc.put(sDetId, hitsInLayer.begin(), hitsInLayer.end());
    }
    layer_idx++;
    old_id = sDetId;
  }

  LogTrace("CSCRecHitDBuilder") << "[CSCRecHitDBuilder] " << oc.size() << " 2d rechits created in this event.";
}

const CSCLayer* CSCRecHitDBuilder::getLayer(const CSCDetId& detId) { return geom_->layer(detId); }

void CSCRecHitDBuilder::setConditions(CSCRecoConditions* reco) {
  recoConditions_ = reco;
  hitsFromStripOnly_->setConditions(reco);
  hitsFromWireOnly_->setConditions(reco);
  make2DHits_->setConditions(reco);
}
