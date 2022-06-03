#ifndef MuNtuple_MuGEMMuonExtTableProducer_h
#define MuNtuple_MuGEMMuonExtTableProducer_h

#include "DPGAnalysis/MuonTools/src/MuBaseFlatTableProducer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

class MuGEMMuonExtTableProducer : public MuBaseFlatTableProducer {
public:
  /// Constructor
  MuGEMMuonExtTableProducer(const edm::ParameterSet &);

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
  /// The RECO mu token
  nano_mu::EDTokenHandle<reco::MuonCollection> m_token;

  /// Fill matches table
  bool m_fillPropagated;

  /// GEM Geometry
  nano_mu::ESTokenHandle<GEMGeometry, MuonGeometryRecord, edm::Transition::BeginRun> m_gemGeometry;

  /// Transient Track Builder
  nano_mu::ESTokenHandle<TransientTrackBuilder, TransientTrackRecord> m_transientTrackBuilder;

  /// Muon service proxy
  std::unique_ptr<MuonServiceProxy> m_muonSP;
};

#endif
