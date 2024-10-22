#ifndef CSCSegment_CSCSegmentProducer_h
#define CSCSegment_CSCSegmentProducer_h

/** \class CSCSegmentProducer 
 * Produces a collection of CSCSegment's in endcap muon CSCs. 
 *
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

class CSCSegmentBuilder;

class CSCSegmentProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit CSCSegmentProducer(const edm::ParameterSet&);
  /// Destructor
  ~CSCSegmentProducer() override;
  /// Produce the CSCSegment collection
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  int iev;  // events through
  CSCSegmentBuilder* segmentBuilder_;
  edm::EDGetTokenT<CSCRecHit2DCollection> m_token;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> m_cscGeometryToken;
};

#endif
