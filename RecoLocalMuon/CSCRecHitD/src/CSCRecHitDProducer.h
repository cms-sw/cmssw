#ifndef CSCRecHitD_CSCRecHitDProducer_h
#define CSCRecHitD_CSCRecHitDProducer_h

/** \class CSCRecHitDProducer 
 *
 * Produces a collection of CSCRecHit2D's (2-dim space-point RecHits)
 * in endcap muon CSCs. 
 * It starts from collections of CSC wire and strip digis.
 * The treatment here is differently than from CSCRecHit2Producer 
 * existing in RecoLocalMuon/CSCRecHit as pseudo-segments are built 
 * from wire hits only and strip only hits. 
 *
 * \author Stoyan Stoynev
 *
 */

#include <FWCore/Framework/interface/ConsumesCollector.h>
#include <FWCore/Framework/interface/Frameworkfwd.h>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>
#include <FWCore/Utilities/interface/ESGetToken.h>

#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

class CSCRecHitDBuilder;
class CSCRecoConditions;

class CSCRecHitDProducer : public edm::stream::EDProducer<> {
public:
  explicit CSCRecHitDProducer(const edm::ParameterSet& ps);
  ~CSCRecHitDProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // Counting events processed
  unsigned iRun;
  bool useCalib;
  bool useStaticPedestals;
  bool useTimingCorrections;
  bool useGasGainCorrections;

  CSCRecHitDBuilder* recHitBuilder_;
  CSCRecoConditions* recoConditions_;

  edm::EDGetTokenT<CSCStripDigiCollection> s_token;
  edm::EDGetTokenT<CSCWireDigiCollection> w_token;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeom_token;
};

#endif
