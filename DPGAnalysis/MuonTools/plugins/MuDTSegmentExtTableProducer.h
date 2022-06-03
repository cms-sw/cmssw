#ifndef MuNtuple_MuDTSegmentExtTableProducer_h
#define MuNtuple_MuDTSegmentExtTableProducer_h

/** \class MuDTSegmentExtTableProducer MuDTSegmentExtTableProducer.h DPGAnalysis/MuonTools/src/MuDTSegmentExtTableProducer.h
 *  
 * Helper class : the segment TableProducer for Phase-1 / Phase2 DT segments (the DataFormat is the same)
 *
 * \author C. Battilana (INFN BO)
 *
 *
 */

#include "DPGAnalysis/MuonTools/src/MuBaseFlatTableProducer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

class MuDTSegmentExtTableProducer : public MuBaseFlatTableProducer {
public:
  /// Constructor
  MuDTSegmentExtTableProducer(const edm::ParameterSet &);

  /// Fill descriptors
  static void fillDescriptions(edm::ConfigurationDescriptions &);

protected:
  /// Fill tree branches for a given event
  void fillTable(edm::Event &) final;

  /// Get info from the ES by run
  void getFromES(const edm::Run &, const edm::EventSetup &) final;

  /// Get info from the ES for a given event
  void getFromES(const edm::EventSetup &) final;

private:
  static const int FIRST_LAYER{1};
  static const int LAST_LAYER{4};
  static const int THETA_SL{2};
  /// The segment token
  nano_mu::EDTokenHandle<DTRecSegment4DCollection> m_token;

  /// Fill rec-hit table
  bool m_fillHits;

  /// Fill segment extrapolation  table
  bool m_fillExtr;

  /// DT Geometry
  nano_mu::ESTokenHandle<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun> m_dtGeometry;

  /// Tracking Geometry
  nano_mu::ESTokenHandle<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> m_trackingGeometry;

  /// Handle DT trigger time pedestals
  std::unique_ptr<DTTTrigBaseSync> m_dtSync;
};

#endif
