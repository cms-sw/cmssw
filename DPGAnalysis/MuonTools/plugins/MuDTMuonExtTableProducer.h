#ifndef MuNtuple_MuDTMuonExtTableProducer_h
#define MuNtuple_MuDTMuonExtTableProducer_h

/** \class MuDTMuonExtTableProducer MuDTMuonExtTableProducer.h DPGAnalysis/MuonTools/plugins/MuDTMuonExtTableProducer.h
 *  
 * Helper class : the muon filler
 *
 * \author L. Lunerti (INFN BO)
 *
 *
 */

#include "DPGAnalysis/MuonTools/src/MuBaseFlatTableProducer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/MuonReco/interface/MuonPFIsolation.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class MuDTMuonExtTableProducer : public MuBaseFlatTableProducer {
public:
  /// Constructor
  MuDTMuonExtTableProducer(const edm::ParameterSet &);

  /// Fill descriptors
  static void fillDescriptions(edm::ConfigurationDescriptions &);

protected:
  /// Fill tree branches for a given events
  void fillTable(edm::Event &) final;

  /// Get info from the ES by run
  void getFromES(const edm::Run &, const edm::EventSetup &) final;

private:
  /// Tokens
  nano_mu::EDTokenHandle<reco::MuonCollection> m_muToken;
  nano_mu::EDTokenHandle<DTRecSegment4DCollection> m_dtSegmentToken;

  nano_mu::EDTokenHandle<edm::TriggerResults> m_trigResultsToken;
  nano_mu::EDTokenHandle<trigger::TriggerEvent> m_trigEventToken;

  /// Fill matches table
  bool m_fillMatches;

  /// Name of the triggers used by muon filler for trigger matching
  std::string m_trigName;
  std::string m_isoTrigName;

  /// DT Geometry
  nano_mu::ESTokenHandle<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun> m_dtGeometry;

  /// HLT config provider
  HLTConfigProvider m_hltConfig;

  /// Indices of the triggers used by muon filler for trigger matching
  std::vector<int> m_trigIndices;
  std::vector<int> m_isoTrigIndices;

  bool hasTrigger(std::vector<int> &,
                  const trigger::TriggerObjectCollection &,
                  edm::Handle<trigger::TriggerEvent> &,
                  const reco::Muon &);

  float computeTrkIso(const reco::MuonIsolation &, float);
  float computePFIso(const reco::MuonPFIsolation &, float);
};

#endif
