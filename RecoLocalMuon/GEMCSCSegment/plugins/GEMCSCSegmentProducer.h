#ifndef GEMCSCSegment_GEMCSCSegmentProducer_h
#define GEMCSCSegment_GEMCSCSegmentProducer_h

/** \class GEMCSCSegmentProducer 
 * Produces a collection of GEM-CSCSegments
 *
 * $Date: 2014/02/06 12:19:20 $
 *
 * \author Raffaella Radogna
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

class GEMCSCSegmentBuilder;

class GEMCSCSegmentProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit GEMCSCSegmentProducer(const edm::ParameterSet&);
  /// Destructor
  ~GEMCSCSegmentProducer() override;
  /// generate gemcscSegment_cfi
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  /// Produce the GEM-CSCSegment collection
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> kCSCGeometryToken_;
  const edm::ESGetToken<GEMGeometry, MuonGeometryRecord> kGEMGeometryToken_;
  const edm::EDGetTokenT<CSCSegmentCollection> kCSCSegmentCollectionToken_;
  const edm::EDGetTokenT<GEMRecHitCollection> kGEMRecHitCollectionToken_;

  int iev;  // events through
  GEMCSCSegmentBuilder* segmentBuilder_;
};

#endif
