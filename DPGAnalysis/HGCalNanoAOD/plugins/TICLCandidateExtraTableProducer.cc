#include <algorithm>
#include <map>
#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoHGCal/TICL/interface/TICLUtils.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"

class TICLCandidateExtraTableProducer : public SimpleFlatTableProducerBase<TICLCandidate, std::vector<TICLCandidate>> {
public:
  using TProd = edm::Ptr<ticl::Trackster>;
  static constexpr float kInvalidBoundaryValue = -999.f;

  TICLCandidateExtraTableProducer(edm::ParameterSet const& params)
      : SimpleFlatTableProducerBase<TICLCandidate, std::vector<TICLCandidate>>(params),
        tracksters_token_(consumes<std::vector<ticl::Trackster>>(params.getParameter<edm::InputTag>("tracksters"))),
        tracks_token_(consumes<std::vector<reco::Track>>(params.getParameter<edm::InputTag>("tracks"))),
        hasLinkedTracksters_(params.existsAs<edm::InputTag>("linkedTracksters")),
        linkedTracksters_token_(
            hasLinkedTracksters_ ? consumes<std::vector<std::vector<unsigned int>>>(
                                        params.getParameter<edm::InputTag>("linkedTracksters"))
                                  : edm::EDGetTokenT<std::vector<std::vector<unsigned int>>>()),
        hasPUInfo_(params.existsAs<edm::InputTag>("caloParticles") &&
                   params.existsAs<edm::InputTag>("caloParticleToSimClustersMap")),
        caloParticles_token_(hasPUInfo_ ? consumes<std::vector<CaloParticle>>(
                                               params.getParameter<edm::InputTag>("caloParticles"))
                                         : edm::EDGetTokenT<std::vector<CaloParticle>>()),
        caloParticleToSimClustersMap_token_(
            hasPUInfo_ ? consumes<std::map<uint, std::vector<uint>>>(
                             params.getParameter<edm::InputTag>("caloParticleToSimClustersMap"))
                       : edm::EDGetTokenT<std::map<uint, std::vector<uint>>>()),
        produceGeneralTrackBoundary_(params.getParameter<bool>("produceGeneralTrackBoundary")),
        detector_(params.getParameter<std::string>("detector")),
        propName_(params.getParameter<std::string>("propagator")),
        geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
        bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
        propagator_token_(esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", propName_))),
        hdc_token_(esConsumes<HGCalDDDConstants, IdealGeometryRecord>(
            edm::ESInputTag("", (detector_ == "HFNose") ? "HGCalHFNoseSensitive" : "HGCalEESensitive"))) {
    // Only declare/emit the linkedTracksters product for instances that actually have it
    // configured (reco TICLCandidates). Otherwise, two module instances of this same class
    // (reco + sim) would both try to produce a main FlatTable literally named
    // "linkedTracksters", which NanoAOD's output module rejects as "multiple main tables".
    if (hasLinkedTracksters_) {
      produces<nanoaod::FlatTable>("linkedTracksters");
    }

    // Full track-index / GSF-track-index lists per candidate (all tracks, not just the first
    // via trackPtr()/gsftrackPtr()). Always produced by every instance; the product name is
    // prefixed with this->name_ ("TICLCandidates"/"SimTICLCandidates") so the two module
    // instances never collide on the same product name.
    produces<nanoaod::FlatTable>(this->name_ + "TrackIdxs");
    produces<nanoaod::FlatTable>(this->name_ + "GsfTrackIdxs");

    // GeneralTrack HGCal-boundary extension: independent of TICLCandidate linking, one row per
    // track in the full tracks collection. Optional enable on exactly one module instance.
    if (produceGeneralTrackBoundary_) {
      produces<nanoaod::FlatTable>("trackBoundary");
    }

    if (params.existsAs<edm::ParameterSet>("collectionVariables")) {
      edm::ParameterSet const& collectionVarsPSet = params.getParameter<edm::ParameterSet>("collectionVariables");
      for (const auto& coltablename : collectionVarsPSet.getParameterNamesForType<edm::ParameterSet>()) {
        const auto& coltablePSet = collectionVarsPSet.getParameter<edm::ParameterSet>(coltablename);

        CollectionVariableTableInfo coltable;
        coltable.name =
            coltablePSet.existsAs<std::string>("name") ? coltablePSet.getParameter<std::string>("name") : coltablename;
        coltable.doc = coltablePSet.getParameter<std::string>("doc");
        coltable.useCount = coltablePSet.getParameter<bool>("useCount");
        coltable.useOffset = coltablePSet.getParameter<bool>("useOffset");

        coltables_.push_back(std::move(coltable));
        produces<nanoaod::FlatTable>(coltables_.back().name + "Table");
      }
    }
  }

  //write empty tables when products are missing
  void writeEmptyTables(edm::Event& iEvent, size_t table_size) const {
    auto out = std::make_unique<nanoaod::FlatTable>(table_size, this->name_, /*singleton*/ false, /*extension*/ true);

    for (const auto& coltable : coltables_) {
      std::vector<uint16_t> emptyCounts(table_size, 0);
      std::vector<uint16_t> emptyOffsets(table_size, 0);

      if (coltable.useCount) {
        out->addColumn<uint16_t>("n" + coltable.name, emptyCounts, "Count for " + coltable.name);
      }
      if (coltable.useOffset) {
        out->addColumn<uint16_t>("o" + coltable.name, emptyOffsets, "Offset for " + coltable.name);
      }

      auto outcoltable = std::make_unique<nanoaod::FlatTable>(0, coltable.name, false, false);
      std::vector<uint32_t> emptyTracksterKeys;
      std::vector<float> emptyBoundaryX, emptyBoundaryY, emptyBoundaryZ;
      std::vector<float> emptyBoundaryEta, emptyBoundaryPhi;
      std::vector<float> emptyBoundaryPx, emptyBoundaryPy, emptyBoundaryPz;
      outcoltable->addColumn<uint32_t>("tracksterIndex", emptyTracksterKeys, "Index of associated Trackster");
      outcoltable->addColumn<float>("track_boundaryX", emptyBoundaryX, "Track X position at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryY", emptyBoundaryY, "Track Y position at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryZ", emptyBoundaryZ, "Track Z position at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryEta", emptyBoundaryEta, "Track eta at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryPhi", emptyBoundaryPhi, "Track phi at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryPx", emptyBoundaryPx, "Track Px at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryPy", emptyBoundaryPy, "Track Py at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryPz", emptyBoundaryPz, "Track Pz at HGCal boundary");
      outcoltable->setDoc(coltable.doc);
      iEvent.put(std::move(outcoltable), coltable.name + "Table");
    }

    // linkedTracksters (pre-merge trackster links)
    if (hasLinkedTracksters_) {
      out->addColumn<uint16_t>("nLinkedTracksters",
                                std::vector<uint16_t>(table_size, 0),
                                "Number of Tracksters linked to candidate before final linking/merging");

      auto linkedTable = std::make_unique<nanoaod::FlatTable>(0, "linkedTracksters", false, false);
      std::vector<uint32_t> emptyLinkedIdx;
      linkedTable->addColumn<uint32_t>("tracksterIndex", emptyLinkedIdx, "Index of linked Trackster");
      linkedTable->setDoc("Tracksters linked to candidate before final linking/merging");
      iEvent.put(std::move(linkedTable), "linkedTracksters");
    }

    // Full track-index / GSF-track-index lists per candidate
    out->addColumn<uint16_t>(
        "nTrackIdxs", std::vector<uint16_t>(table_size, 0), "Number of generalTracks associated with candidate");
    {
      auto trackIdxsTable = std::make_unique<nanoaod::FlatTable>(0, this->name_ + "TrackIdxs", false, false);
      std::vector<uint32_t> emptyIdx;
      trackIdxsTable->addColumn<uint32_t>("trackIndex", emptyIdx, "Index of associated generalTrack");
      trackIdxsTable->setDoc("Full list of generalTrack indices associated with the candidate");
      iEvent.put(std::move(trackIdxsTable), this->name_ + "TrackIdxs");
    }

    out->addColumn<uint16_t>(
        "nGsfTrackIdxs", std::vector<uint16_t>(table_size, 0), "Number of GSFTracks associated with candidate");
    {
      auto gsfTrackIdxsTable = std::make_unique<nanoaod::FlatTable>(0, this->name_ + "GsfTrackIdxs", false, false);
      std::vector<uint32_t> emptyIdx;
      gsfTrackIdxsTable->addColumn<uint32_t>("trackIndex", emptyIdx, "Index of associated GSFTrack");
      gsfTrackIdxsTable->setDoc("Full list of GSFTrack indices associated with the candidate");
      iEvent.put(std::move(gsfTrackIdxsTable), this->name_ + "GsfTrackIdxs");
    }

    // isPU (sim candidates only); 
    if (hasPUInfo_) {
      out->addColumn<int>("isPU",
                           std::vector<int>(table_size, -1),
                           "PU flag of the candidate's parent CaloParticle: 1 = pileup (non-zero event/bx), 0 = "
                           "otherwise, -1 = unresolved");
    }

    if (out->nColumns() > 0) {
      out->setDoc(this->doc_);
      iEvent.put(std::move(out));
    }
  }

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    const auto& prod = iEvent.getHandle(this->src_);

    const auto& hgcons = iSetup.getData(hdc_token_);
    const auto& geom = iSetup.getData(geometry_token_);
    const auto& bfield = iSetup.getData(bfield_token_);
    const auto& propagator = iSetup.getData(propagator_token_);

    const auto firstDisk = ticl::utils::buildHGCalFirstDisks(hgcons, geom);

    // GeneralTrack HGCal-boundary extension: computed independently of TICLCandidate/Trackster
    // validity, for every track in the full tracks collection, so it always matches
    // GeneralTrack's own row count exactly. Handled first and unconditionally so the declared
    // product is always put(), regardless of what happens with candidates below.
    if (produceGeneralTrackBoundary_) {
      const auto& allTracks_h = iEvent.getHandle(tracks_token_);
      const size_t nAllTracks = allTracks_h.isValid() ? allTracks_h->size() : 0;
      auto trackBoundaryTable =
          std::make_unique<nanoaod::FlatTable>(nAllTracks, "GeneralTrack", /*singleton*/ false, /*extension*/ true);
      std::vector<float> hgcal_x(nAllTracks), hgcal_y(nAllTracks), hgcal_z(nAllTracks);
      std::vector<float> hgcal_eta(nAllTracks), hgcal_phi(nAllTracks);
      std::vector<float> hgcal_px(nAllTracks), hgcal_py(nAllTracks), hgcal_pz(nAllTracks);

      if (allTracks_h.isValid()) {
        const auto& allTracks = *allTracks_h;
        for (size_t i = 0; i < allTracks.size(); ++i) {
          const auto& track = allTracks[i];
          int iSide = int(track.eta() > 0);
          const auto& fts = trajectoryStateTransform::outerFreeState(track, &bfield);
          const auto& tsos = propagator.propagate(fts, firstDisk[iSide]->surface());
          if (tsos.isValid()) {
            const auto& globalPos = tsos.globalPosition();
            const auto& globalMom = tsos.globalMomentum();
            hgcal_x[i] = globalPos.x();
            hgcal_y[i] = globalPos.y();
            hgcal_z[i] = globalPos.z();
            hgcal_eta[i] = globalPos.eta();
            hgcal_phi[i] = globalPos.phi();
            hgcal_px[i] = globalMom.x();
            hgcal_py[i] = globalMom.y();
            hgcal_pz[i] = globalMom.z();
          } else {
            hgcal_x[i] = hgcal_y[i] = hgcal_z[i] = kInvalidBoundaryValue;
            hgcal_eta[i] = hgcal_phi[i] = kInvalidBoundaryValue;
            hgcal_px[i] = hgcal_py[i] = hgcal_pz[i] = kInvalidBoundaryValue;
          }
        }
      }

      trackBoundaryTable->addColumn<float>("hgcal_x", hgcal_x, "Track X position at HGCal boundary");
      trackBoundaryTable->addColumn<float>("hgcal_y", hgcal_y, "Track Y position at HGCal boundary");
      trackBoundaryTable->addColumn<float>("hgcal_z", hgcal_z, "Track Z position at HGCal boundary");
      trackBoundaryTable->addColumn<float>("hgcal_eta", hgcal_eta, "Track eta at HGCal boundary");
      trackBoundaryTable->addColumn<float>("hgcal_phi", hgcal_phi, "Track phi at HGCal boundary");
      trackBoundaryTable->addColumn<float>("hgcal_px", hgcal_px, "Track Px at HGCal boundary");
      trackBoundaryTable->addColumn<float>("hgcal_py", hgcal_py, "Track Py at HGCal boundary");
      trackBoundaryTable->addColumn<float>("hgcal_pz", hgcal_pz, "Track Pz at HGCal boundary");
      iEvent.put(std::move(trackBoundaryTable), "trackBoundary");
    }

    if (!prod.isValid() && this->skipNonExistingSrc_) {
      writeEmptyTables(iEvent, 0);
      return;
    }

    const auto& tracksters_h = iEvent.getHandle(tracksters_token_);
    if (!tracksters_h.isValid() && this->skipNonExistingSrc_) {
      writeEmptyTables(iEvent, prod->size());
      return;
    }
    const auto& tracksters = *tracksters_h;

    const auto& tracks_h = iEvent.getHandle(tracks_token_);
    if (!tracks_h.isValid() && this->skipNonExistingSrc_) {
      writeEmptyTables(iEvent, prod->size());
      return;
    }
    const auto& tracks = *tracks_h;

    // linkedTracksters (pre-merge trackster links) is only meaningful for reco TICLCandidates;
    // it's an optional parameter so sim configurations can simply omit it.
    edm::Handle<std::vector<std::vector<unsigned int>>> linkedTracksters_h;
    if (hasLinkedTracksters_) {
      linkedTracksters_h = iEvent.getHandle(linkedTracksters_token_);
      if (!linkedTracksters_h.isValid() && this->skipNonExistingSrc_) {
        writeEmptyTables(iEvent, prod->size());
        return;
      }
    }

    // isPU (sim candidates only): resolved via the seedID/seedIndex of the candidate's first
    // constituent Trackster, same CaloParticle-resolution logic as SimTracksterTableProducer,
    edm::Handle<std::vector<CaloParticle>> caloParticles_h;
    edm::Handle<std::map<uint, std::vector<uint>>> cpToSCMap_h;
    if (hasPUInfo_) {
      caloParticles_h = iEvent.getHandle(caloParticles_token_);
      cpToSCMap_h = iEvent.getHandle(caloParticleToSimClustersMap_token_);
      if ((!caloParticles_h.isValid() || !cpToSCMap_h.isValid()) && this->skipNonExistingSrc_) {
        writeEmptyTables(iEvent, prod->size());
        return;
      }
    }

    const auto& candidates = *prod;
    const size_t table_size = candidates.size();

    auto out = std::make_unique<nanoaod::FlatTable>(table_size, this->name_, /*singleton*/ false, /*extension*/ true);

    unsigned int coltablesize = 0;
    std::vector<unsigned int> counts;
    counts.reserve(table_size);

    std::vector<uint32_t> tracksterKeys;
    std::vector<float> track_boundaryX, track_boundaryY, track_boundaryZ;
    std::vector<float> track_boundaryEta, track_boundaryPhi;
    std::vector<float> track_boundaryPx, track_boundaryPy, track_boundaryPz;

    std::vector<uint16_t> linkedCounts;
    linkedCounts.reserve(table_size);
    std::vector<uint32_t> linkedTracksterIndices;

    std::vector<uint16_t> trackIdxCounts;
    trackIdxCounts.reserve(table_size);
    std::vector<uint32_t> allTrackIndices;

    std::vector<uint16_t> gsfTrackIdxCounts;
    gsfTrackIdxCounts.reserve(table_size);
    std::vector<uint32_t> allGsfTrackIndices;

    std::vector<int> isPU;
    if (hasPUInfo_) {
      isPU.reserve(table_size);
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
      const auto& cand = candidates[i];
      const auto& children = cand.tracksters();
      counts.push_back(children.size());
      coltablesize += children.size();

      for (const auto& t : children) {
        tracksterKeys.push_back(t.key());

        // get the trackster and its track indices
        const auto& trackster = tracksters.at(t.key());
        const auto& trackIdxs = trackster.trackIdxs();

        if (!trackIdxs.empty()) {
          // for now, only one track
          const auto trackIdx = trackIdxs[0];
          const auto& track = tracks.at(trackIdx);

          int iSide = int(track.eta() > 0);
          const auto& fts = trajectoryStateTransform::outerFreeState(track, &bfield);
          const auto& tsos = propagator.propagate(fts, firstDisk[iSide]->surface());

          if (tsos.isValid()) {
            const auto& globalPos = tsos.globalPosition();
            const auto& globalMom = tsos.globalMomentum();
            track_boundaryX.push_back(globalPos.x());
            track_boundaryY.push_back(globalPos.y());
            track_boundaryZ.push_back(globalPos.z());
            track_boundaryEta.push_back(globalPos.eta());
            track_boundaryPhi.push_back(globalPos.phi());
            track_boundaryPx.push_back(globalMom.x());
            track_boundaryPy.push_back(globalMom.y());
            track_boundaryPz.push_back(globalMom.z());
          } else {
            track_boundaryX.push_back(kInvalidBoundaryValue);
            track_boundaryY.push_back(kInvalidBoundaryValue);
            track_boundaryZ.push_back(kInvalidBoundaryValue);
            track_boundaryEta.push_back(kInvalidBoundaryValue);
            track_boundaryPhi.push_back(kInvalidBoundaryValue);
            track_boundaryPx.push_back(kInvalidBoundaryValue);
            track_boundaryPy.push_back(kInvalidBoundaryValue);
            track_boundaryPz.push_back(kInvalidBoundaryValue);
          }
        } else {
          // no tracks associated with this trackster
          track_boundaryX.push_back(kInvalidBoundaryValue);
          track_boundaryY.push_back(kInvalidBoundaryValue);
          track_boundaryZ.push_back(kInvalidBoundaryValue);
          track_boundaryEta.push_back(kInvalidBoundaryValue);
          track_boundaryPhi.push_back(kInvalidBoundaryValue);
          track_boundaryPx.push_back(kInvalidBoundaryValue);
          track_boundaryPy.push_back(kInvalidBoundaryValue);
          track_boundaryPz.push_back(kInvalidBoundaryValue);
        }
      }

      // linked tracksters (pre-merge links), 
      // linkedTracksters[i] holds the trackster indices linked to candidate i
      // before the final linking/merging step. Not available/meaningful for sim candidates.
      if (hasLinkedTracksters_ && linkedTracksters_h.isValid() && i < linkedTracksters_h->size()) {
        const auto& linked = (*linkedTracksters_h)[i];
        linkedCounts.push_back(linked.size());
        for (auto idx : linked) {
          linkedTracksterIndices.push_back(idx);
        }
      } else {
        linkedCounts.push_back(0);
      }

      // Full track-index / GSF-track-index lists (all tracks, not just the first via
      // trackPtr()/gsftrackPtr()). cand.trackPtrs()/gsfTrackPtrs() are Ptrs into whatever
      // collection the candidate was built against -- key() gives the row index directly.
      const auto& candTrackPtrs = cand.trackPtrs();
      trackIdxCounts.push_back(candTrackPtrs.size());
      for (const auto& tp : candTrackPtrs) {
        allTrackIndices.push_back(tp.key());
      }

      const auto& candGsfTrackPtrs = cand.gsfTrackPtrs();
      gsfTrackIdxCounts.push_back(candGsfTrackPtrs.size());
      for (const auto& gp : candGsfTrackPtrs) {
        allGsfTrackIndices.push_back(gp.key());
      }

      // isPU: resolve the parent CaloParticle from the first constituent Trackster's
      // seedID/seedIndex, then read its g4Track eventId/bunchCrossing 
      if (hasPUInfo_) {
        int puFlag = -1;
        if (caloParticles_h.isValid() && cpToSCMap_h.isValid() && !children.empty()) {
          const auto& firstTrackster = tracksters[children[0].key()];
          const auto& caloParticles = *caloParticles_h;
          const auto& cpToSCMap = *cpToSCMap_h;

          const CaloParticle* cp = nullptr;
          if (firstTrackster.seedID() == caloParticles_h.id()) {
            const auto seedIdx = firstTrackster.seedIndex();
            if (seedIdx >= 0 && static_cast<size_t>(seedIdx) < caloParticles.size()) {
              cp = &caloParticles[seedIdx];
            }
          } else {
            const auto seedIdx = firstTrackster.seedIndex();
            for (const auto& [cpIdx, scVec] : cpToSCMap) {
              if (seedIdx >= 0 &&
                  std::find(scVec.begin(), scVec.end(), static_cast<unsigned int>(seedIdx)) != scVec.end()) {
                if (static_cast<size_t>(cpIdx) < caloParticles.size()) {
                  cp = &caloParticles[cpIdx];
                }
                break;
              }
            }
          }

          if (cp && !cp->g4Tracks().empty()) {
            const auto& simTrack = cp->g4Tracks()[0];
            puFlag = (simTrack.eventId().event() != 0 || simTrack.eventId().bunchCrossing() != 0) ? 1 : 0;
          }
        }
        isPU.push_back(puFlag);
      }
    }

    for (const auto& coltable : coltables_) {
      if (coltable.useCount) {
        out->addColumn<uint16_t>("n" + coltable.name, counts, "Count for " + coltable.name);
      }
      if (coltable.useOffset) {
        std::vector<unsigned int> offsets;
        offsets.reserve(counts.size());
        unsigned int offset = 0;
        for (auto c : counts) {
          offsets.push_back(offset);
          offset += c;
        }
        out->addColumn<uint16_t>("o" + coltable.name, offsets, "Offset for " + coltable.name);
      }

      auto outcoltable = std::make_unique<nanoaod::FlatTable>(coltablesize, coltable.name, false, false);

      outcoltable->addColumn<uint32_t>("tracksterIndex", tracksterKeys, "Index of associated Trackster");
      outcoltable->addColumn<float>("track_boundaryX", track_boundaryX, "Track X position at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryY", track_boundaryY, "Track Y position at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryZ", track_boundaryZ, "Track Z position at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryEta", track_boundaryEta, "Track eta at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryPhi", track_boundaryPhi, "Track phi at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryPx", track_boundaryPx, "Track Px at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryPy", track_boundaryPy, "Track Py at HGCal boundary");
      outcoltable->addColumn<float>("track_boundaryPz", track_boundaryPz, "Track Pz at HGCal boundary");

      outcoltable->setDoc(coltable.doc);
      iEvent.put(std::move(outcoltable), coltable.name + "Table");
    }

    // linkedTracksters output: count column on the main table + flattened sub-table.
    // Only emitted for instances configured with linkedTracksters (reco)
    if (hasLinkedTracksters_) {
      out->addColumn<uint16_t>("nLinkedTracksters",
                                linkedCounts,
                                "Number of Tracksters linked to candidate before final linking/merging");

      auto linkedTable =
          std::make_unique<nanoaod::FlatTable>(linkedTracksterIndices.size(), "linkedTracksters", false, false);
      linkedTable->addColumn<uint32_t>("tracksterIndex", linkedTracksterIndices, "Index of linked Trackster");
      linkedTable->setDoc("Tracksters linked to candidate before final linking/merging");
      iEvent.put(std::move(linkedTable), "linkedTracksters");
    }

    // Full track-index / GSF-track-index lists: count columns on the main table + flattened
    // sub-tables, always emitted (see constructor comment). Product names are prefixed with
    // this->name_ so the reco and sim module instances never collide.
    out->addColumn<uint16_t>(
        "nTrackIdxs", trackIdxCounts, "Number of generalTracks associated with candidate");
    {
      auto trackIdxsTable =
          std::make_unique<nanoaod::FlatTable>(allTrackIndices.size(), this->name_ + "TrackIdxs", false, false);
      trackIdxsTable->addColumn<uint32_t>("trackIndex", allTrackIndices, "Index of associated generalTrack");
      trackIdxsTable->setDoc("Full list of generalTrack indices associated with the candidate");
      iEvent.put(std::move(trackIdxsTable), this->name_ + "TrackIdxs");
    }

    out->addColumn<uint16_t>(
        "nGsfTrackIdxs", gsfTrackIdxCounts, "Number of GSFTracks associated with candidate");
    {
      auto gsfTrackIdxsTable =
          std::make_unique<nanoaod::FlatTable>(allGsfTrackIndices.size(), this->name_ + "GsfTrackIdxs", false, false);
      gsfTrackIdxsTable->addColumn<uint32_t>("trackIndex", allGsfTrackIndices, "Index of associated GSFTrack");
      gsfTrackIdxsTable->setDoc("Full list of GSFTrack indices associated with the candidate");
      iEvent.put(std::move(gsfTrackIdxsTable), this->name_ + "GsfTrackIdxs");
    }

    // isPU output: one scalar per candidate, directly on the extension table.
    // Only emitted for instances configured with caloParticles + caloParticleToSimClustersMap (sim).
    if (hasPUInfo_) {
      out->addColumn<int>(
          "isPU", isPU, "PU flag of the candidate's parent CaloParticle: 1 = pileup (non-zero event/bx), 0 = otherwise, -1 = unresolved");
    }

    if (out->nColumns() > 0) {
      out->setDoc(this->doc_);
      iEvent.put(std::move(out));
    }
  }

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event&,
                                                const edm::Handle<std::vector<TICLCandidate>>&) const override {
    return std::make_unique<nanoaod::FlatTable>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc =
        SimpleFlatTableProducerBase<TICLCandidate, std::vector<TICLCandidate>>::baseDescriptions();

    desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersCLUE3DHigh"));
    desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
    desc.addOptional<edm::InputTag>("linkedTracksters")
        ->setComment(
            "Pre-merge/pre-linking Trackster links (reco TICLCandidates only, e.g. ");
    desc.addOptional<edm::InputTag>("caloParticles")
        ->setComment(
            "CaloParticle collection (sim candidates only), used together with "
            "caloParticleToSimClustersMap to compute the isPU column. Omit for reco candidates.");
    desc.addOptional<edm::InputTag>("caloParticleToSimClustersMap")
        ->setComment(
            "CaloParticle-to-SimCluster index map (sim candidates only), e.g. the unlabeled product from "
            "SimTrackstersProducer (same module as the simTracksters src). Required together with "
            "caloParticles to compute the isPU column. Omit for reco candidates.");
    desc.add<std::string>("detector", "HGCAL");
    desc.add<std::string>("propagator", "PropagatorWithMaterial");
    desc.add<bool>("produceGeneralTrackBoundary", false)
        ->setComment(
            "If true, also emit an extension of the GeneralTrack table with HGCal-boundary "
            "propagated position/momentum for every track in the tracks collection (unconditional, "
            "not restricted to candidate-linked tracks), mirroring TICLDumper's track_hgcal_* branches. "
            "Enable on exactly one module instance to avoid duplicate GeneralTrack extensions.");

    edm::ParameterSetDescription coltable;
    coltable.add<std::string>("name", "hltTiclCandidate");
    coltable.add<std::string>("doc", "TICL Candidates");
    coltable.add<bool>("useCount", true);
    coltable.add<bool>("useOffset", false);
    edm::ParameterSetDescription colvariables;  // unused here
    coltable.add<edm::ParameterSetDescription>("variables", colvariables);

    edm::ParameterSetDescription coltables;
    coltables.addOptionalNode(
        edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, true, coltable), false);

    desc.addOptional<edm::ParameterSetDescription>("collectionVariables", coltables);
    descriptions.addWithDefaultLabel(desc);
  }

protected:
  struct CollectionVariableTableInfo {
    std::string name;
    std::string doc;
    bool useCount;
    bool useOffset;
  };
  std::vector<CollectionVariableTableInfo> coltables_;

  const edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksters_token_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  const bool hasLinkedTracksters_;
  const edm::EDGetTokenT<std::vector<std::vector<unsigned int>>> linkedTracksters_token_;
  const bool hasPUInfo_;
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloParticles_token_;
  const edm::EDGetTokenT<std::map<uint, std::vector<uint>>> caloParticleToSimClustersMap_token_;
  const bool produceGeneralTrackBoundary_;
  const std::string detector_;
  const std::string propName_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TICLCandidateExtraTableProducer);
