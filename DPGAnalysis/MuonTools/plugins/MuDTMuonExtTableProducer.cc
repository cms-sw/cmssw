/** \class MuDTMuonExtTableProducer MuDTMuonExtTableProducer.cc DPGAnalysis/MuonTools/plugins/MuDTMuonExtTableProducer.cc
 *  
 * Helper class : the muon filler
 *
 * \author L. Lunerti (INFN BO)
 *
 *
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
#include "DataFormats/MuonReco/interface/MuonSegmentMatch.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "TString.h"
#include "TRegexp.h"

#include <numeric>
#include <vector>

#include "DPGAnalysis/MuonTools/interface/MuBaseFlatTableProducer.h"

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
  MuDTMuonExtTableProducer(const edm::ParameterSet&);

  /// Fill descriptors
  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  /// Fill tree branches for a given events
  void fillTable(edm::Event&) final;

  /// Get info from the ES by run
  void getFromES(const edm::Run&, const edm::EventSetup&) final;

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

  bool hasTrigger(std::vector<int>&,
                  const trigger::TriggerObjectCollection&,
                  edm::Handle<trigger::TriggerEvent>&,
                  const reco::Muon&);
};

MuDTMuonExtTableProducer::MuDTMuonExtTableProducer(const edm::ParameterSet& config)
    : MuBaseFlatTableProducer(config),
      m_muToken{config, consumesCollector(), "src"},
      m_dtSegmentToken{config, consumesCollector(), "dtSegmentSrc"},
      m_trigResultsToken{config, consumesCollector(), "trigResultsSrc"},
      m_trigEventToken{config, consumesCollector(), "trigEventSrc"},
      m_fillMatches{config.getParameter<bool>("fillMatches")},
      m_trigName{config.getParameter<std::string>("trigName")},
      m_isoTrigName{config.getParameter<std::string>("isoTrigName")},
      m_dtGeometry{consumesCollector()} {
  produces<nanoaod::FlatTable>();
  if (m_fillMatches) {
    produces<nanoaod::FlatTable>("matches");
    produces<nanoaod::FlatTable>("staMatches");
  }
}

void MuDTMuonExtTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("name", "muon");
  desc.add<edm::InputTag>("src", edm::InputTag{"muons"});
  desc.add<edm::InputTag>("dtSegmentSrc", edm::InputTag{"dt4DSegments"});
  desc.add<edm::InputTag>("trigEventSrc", edm::InputTag{"hltTriggerSummaryAOD::HLT"});
  desc.add<edm::InputTag>("trigResultsSrc", edm::InputTag{"TriggerResults::HLT"});

  desc.add<bool>("fillMatches", true);

  desc.add<std::string>("trigName", "HLT_Mu55*");
  desc.add<std::string>("isoTrigName", "HLT_IsoMu2*");

  descriptions.addWithDefaultLabel(desc);
}

void MuDTMuonExtTableProducer::getFromES(const edm::Run& run, const edm::EventSetup& environment) {
  m_dtGeometry.getFromES(environment);

  bool changed{true};
  m_hltConfig.init(run, environment, "HLT", changed);

  const bool enableWildcard{true};

  TString tName = TString(m_trigName);
  TRegexp tNamePattern = TRegexp(tName, enableWildcard);

  for (unsigned iPath = 0; iPath < m_hltConfig.size(); ++iPath) {
    TString pathName = TString(m_hltConfig.triggerName(iPath));
    if (pathName.Contains(tNamePattern))
      m_trigIndices.push_back(static_cast<int>(iPath));
  }

  tName = TString(m_isoTrigName);
  tNamePattern = TRegexp(tName, enableWildcard);

  for (unsigned iPath = 0; iPath < m_hltConfig.size(); ++iPath) {
    TString pathName = TString(m_hltConfig.triggerName(iPath));
    if (pathName.Contains(tNamePattern))
      m_isoTrigIndices.push_back(static_cast<int>(iPath));
  }
}

void MuDTMuonExtTableProducer::fillTable(edm::Event& ev) {
  unsigned int nMuons{0};

  std::vector<bool> firesIsoTrig;
  std::vector<bool> firesTrig;

  std::vector<int> nMatches;
  std::vector<int> staMu_nMatchSeg;

  std::vector<uint32_t> matches_begin;
  std::vector<uint32_t> matches_end;

  std::vector<uint32_t> staMatches_begin;
  std::vector<uint32_t> staMatches_end;

  std::vector<int8_t> matches_wheel;
  std::vector<int8_t> matches_sector;
  std::vector<int8_t> matches_station;

  std::vector<float> matches_x;
  std::vector<float> matches_y;

  std::vector<float> matches_phi;
  std::vector<float> matches_eta;
  std::vector<float> matches_edgeX;
  std::vector<float> matches_edgeY;

  std::vector<float> matches_dXdZ;
  std::vector<float> matches_dYdZ;

  std::vector<uint32_t> staMatches_MuSegIdx;

  auto muons = m_muToken.conditionalGet(ev);
  auto segments = m_dtSegmentToken.conditionalGet(ev);

  auto triggerResults = m_trigResultsToken.conditionalGet(ev);
  auto triggerEvent = m_trigEventToken.conditionalGet(ev);

  if (muons.isValid() && segments.isValid()) {
    for (const auto& muon : (*muons)) {
      if (triggerResults.isValid() && triggerEvent.isValid()) {
        const auto& triggerObjects = triggerEvent->getObjects();

        bool hasIsoTrig = hasTrigger(m_isoTrigIndices, triggerObjects, triggerEvent, muon);
        bool hasTrig = hasTrigger(m_trigIndices, triggerObjects, triggerEvent, muon);

        firesIsoTrig.push_back(hasIsoTrig);
        firesTrig.push_back(hasTrig);

      } else {
        firesIsoTrig.push_back(false);
        firesTrig.push_back(false);
      }

      size_t iMatches = 0;
      size_t iSegMatches = 0;

      if (m_fillMatches) {
        matches_begin.push_back(matches_wheel.size());

        if (muon.isMatchesValid()) {
          for (const auto& match : muon.matches()) {
            if (match.id.det() == DetId::Muon && match.id.subdetId() == MuonSubdetId::DT) {
              DTChamberId dtId(match.id.rawId());
              const auto chamb = m_dtGeometry->chamber(static_cast<DTChamberId>(match.id));

              matches_wheel.push_back(dtId.wheel());
              matches_sector.push_back(dtId.sector());
              matches_station.push_back(dtId.station());

              matches_x.push_back(match.x);
              matches_y.push_back(match.y);

              matches_phi.push_back(chamb->toGlobal(LocalPoint(match.x, match.y, 0.)).phi());
              matches_eta.push_back(chamb->toGlobal(LocalPoint(match.x, match.y, 0.)).eta());

              matches_edgeX.push_back(match.edgeX);
              matches_edgeY.push_back(match.edgeY);

              matches_dXdZ.push_back(match.dXdZ);
              matches_dYdZ.push_back(match.dYdZ);

              ++iMatches;
            }
          }
        }

        matches_end.push_back(matches_wheel.size());

        //SEGMENT MATCHING VARIABLES

        staMatches_begin.push_back(staMatches_MuSegIdx.size());

        if (!muon.outerTrack().isNull()) {
          reco::TrackRef outerTrackRef = muon.outerTrack();

          auto recHitIt = outerTrackRef->recHitsBegin();
          auto recHitEnd = outerTrackRef->recHitsEnd();

          for (; recHitIt != recHitEnd; ++recHitIt) {
            DetId detId = (*recHitIt)->geographicalId();

            if (detId.det() == DetId::Muon && detId.subdetId() == MuonSubdetId::DT) {
              const auto dtSegmentSta = dynamic_cast<const DTRecSegment4D*>((*recHitIt));
              int iSeg = 0;

              for (const auto& segment : (*segments)) {
                if (dtSegmentSta && dtSegmentSta->chamberId().station() == segment.chamberId().station() &&
                    dtSegmentSta->chamberId().wheel() == segment.chamberId().wheel() &&
                    dtSegmentSta->chamberId().sector() == segment.chamberId().sector() &&
                    std::abs(dtSegmentSta->localPosition().x() - segment.localPosition().x()) < 0.001 &&
                    std::abs(dtSegmentSta->localPosition().y() - segment.localPosition().y()) < 0.001 &&
                    std::abs(dtSegmentSta->localDirection().x() - segment.localDirection().x()) < 0.001 &&
                    std::abs(dtSegmentSta->localDirection().y() - segment.localDirection().y()) < 0.001) {
                  staMatches_MuSegIdx.push_back(iSeg);
                  ++iSegMatches;
                }

                ++iSeg;
              }  //loop over segments
            }

          }  //loop over recHits
        }

        staMatches_end.push_back(staMatches_MuSegIdx.size());
      }

      nMatches.push_back(iMatches);
      staMu_nMatchSeg.push_back(iSegMatches);

      ++nMuons;
    }
  }

  auto table = std::make_unique<nanoaod::FlatTable>(nMuons, m_name, false, true);

  addColumn(table,
            "firesIsoTrig",
            firesIsoTrig,
            "True if the muon is matched to an isolated trigger"
            "<br />specified in the ntuple config file");

  addColumn(table,
            "firesTrig",
            firesTrig,
            "True if the muon is matched to a  (non isolated)trigger"
            "<br />specified in the ntuple config file");

  addColumn(table, "nMatches", nMatches, "Number of muon chamber matches (DT only)");
  addColumn(table, "staMu_nMatchSeg", staMu_nMatchSeg, "Number of segments used in the standalone track (DT only)");

  addColumn(table,
            "matches_begin",
            matches_begin,
            "begin() of range of quantities for a given muon in the *_matches_* vectors");
  addColumn(
      table, "matches_end", matches_end, "end() of range of quantities for a given muon in the *_matches_* vectors");

  addColumn(table,
            "staMatches_begin",
            staMatches_begin,
            "begin() of range of quantities for a given muon in the matches_staMuSegIdx vector");
  addColumn(table,
            "staMatches_end",
            staMatches_end,
            "end() of range of quantities for a given muon in the matches_staMuSegIdx vector");

  ev.put(std::move(table));

  if (m_fillMatches) {
    auto sum = [](std::vector<int> v) { return std::accumulate(v.begin(), v.end(), 0); };

    auto tabMatches = std::make_unique<nanoaod::FlatTable>(sum(nMatches), m_name + "_matches", false, false);

    tabMatches->setDoc("RECO muon matches_* vectors");

    addColumn(tabMatches, "x", matches_x, "x position of the extrapolated track on the matched DT chamber");
    addColumn(tabMatches, "y", matches_y, "x position of the extrapolated track on the matched DT chamber");

    addColumn(tabMatches, "wheel", matches_wheel, "matched DT chamber wheel");
    addColumn(tabMatches, "sector", matches_sector, "matched DT chamber sector");
    addColumn(tabMatches, "station", matches_station, "matched DT chamber station");

    addColumn(
        tabMatches, "phi", matches_phi, "phi of the (x,y) position on the matched DT chamber (global reference frame)");
    addColumn(tabMatches,
              "eta",
              matches_eta,
              " eta of the (x,y) position on the matched cDT hamber (global reference frame)");

    addColumn(tabMatches, "dXdZ", matches_dXdZ, "dXdZ of the extrapolated track on the matched DT chamber");
    addColumn(tabMatches, "dYdZ", matches_dYdZ, "dYdZ of the extrapolated track on the matched DT chamber");

    ev.put(std::move(tabMatches), "matches");

    auto tabStaMatches =
        std::make_unique<nanoaod::FlatTable>(sum(staMu_nMatchSeg), m_name + "_staMatches", false, false);

    tabStaMatches->setDoc("RECO muon staMatches_* vector");

    addColumn(tabStaMatches,
              "MuSegIdx",
              staMatches_MuSegIdx,
              "Index of DT segments used in the standalone track it corresponds"
              "<br />to the index of a given segment in the ntuple seg_* collection");

    ev.put(std::move(tabStaMatches), "staMatches");
  }
}

bool MuDTMuonExtTableProducer::hasTrigger(std::vector<int>& trigIndices,
                                          const trigger::TriggerObjectCollection& trigObjs,
                                          edm::Handle<trigger::TriggerEvent>& trigEvent,
                                          const reco::Muon& muon) {
  float dRMatch = 999.;
  for (int trigIdx : trigIndices) {
    const std::vector<std::string> trigModuleLabels = m_hltConfig.moduleLabels(trigIdx);

    const unsigned trigModuleIndex =
        std::find(trigModuleLabels.begin(), trigModuleLabels.end(), "hltBoolEnd") - trigModuleLabels.begin() - 1;
    const unsigned hltFilterIndex = trigEvent->filterIndex(edm::InputTag(trigModuleLabels[trigModuleIndex], "", "HLT"));
    if (hltFilterIndex < trigEvent->sizeFilters()) {
      const trigger::Keys keys = trigEvent->filterKeys(hltFilterIndex);
      const trigger::Vids vids = trigEvent->filterIds(hltFilterIndex);
      const unsigned nTriggers = vids.size();

      for (unsigned iTrig = 0; iTrig < nTriggers; ++iTrig) {
        trigger::TriggerObject trigObj = trigObjs[keys[iTrig]];
        float dR = deltaR(muon, trigObj);
        if (dR < dRMatch)
          dRMatch = dR;
      }
    }
  }

  return dRMatch < 0.1;  //CB should become programmable
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuDTMuonExtTableProducer);