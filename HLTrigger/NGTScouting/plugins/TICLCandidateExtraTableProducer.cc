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
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

//
// One-to-many: TICLCandidate -> linked Tracksters
// Or SimTICLCandidate --> SimTracksters
//

namespace {
  std::array<std::unique_ptr<GeomDet>, 2> buildHGCalFirstDisks(const HGCalDDDConstants& hgcons,
                                                               const CaloGeometry& geom) {
    hgcal::RecHitTools rhtools;
    rhtools.setGeometry(geom);
    float zVal = hgcons.waferZ(1, true);
    std::pair<float, float> rMinMax = hgcons.rangeR(zVal, true);

    std::array<std::unique_ptr<GeomDet>, 2> firstDisk;
    for (int iSide = 0; iSide < 2; ++iSide) {
      float zSide = (iSide == 0) ? (-1. * zVal) : zVal;
      firstDisk[iSide] = std::make_unique<GeomDet>(
          Disk::build(Disk::PositionType(0, 0, zSide),
                      Disk::RotationType(),
                      SimpleDiskBounds(rMinMax.first, rMinMax.second, zSide - 0.5, zSide + 0.5))
              .get());
    }
    return firstDisk;
  }
}  // namespace

class TICLCandidateExtraTableProducer : public SimpleFlatTableProducerBase<TICLCandidate, std::vector<TICLCandidate>> {
public:
  using TProd = edm::Ptr<ticl::Trackster>;

  TICLCandidateExtraTableProducer(edm::ParameterSet const& params)
      : SimpleFlatTableProducerBase<TICLCandidate, std::vector<TICLCandidate>>(params),
        tracksters_token_(consumes<std::vector<ticl::Trackster>>(params.getParameter<edm::InputTag>("tracksters"))),
        tracks_token_(consumes<std::vector<reco::Track>>(params.getParameter<edm::InputTag>("tracks"))),
        detector_(params.getParameter<std::string>("detector")),
        propName_(params.getParameter<std::string>("propagator")),
        geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
        bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
        propagator_token_(esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", propName_))),
        hdc_token_(esConsumes<HGCalDDDConstants, IdealGeometryRecord>(
            edm::ESInputTag("", (detector_ == "HFNose") ? "HGCalHFNoseSensitive" : "HGCalEESensitive"))) {
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

        this->coltables_.push_back(std::move(coltable));
        produces<nanoaod::FlatTable>(coltables_.back().name + "Table");
      }
    }
  }

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    const auto& prod = iEvent.getHandle(this->src_);

    const auto& hgcons = iSetup.getData(hdc_token_);
    const auto& geom = iSetup.getData(geometry_token_);
    const auto& bfield = iSetup.getData(bfield_token_);
    const auto& propagator = iSetup.getData(propagator_token_);

    const auto firstDisk = buildHGCalFirstDisks(hgcons, geom);

    if (!prod.isValid() && this->skipNonExistingSrc_) {
      auto out = std::make_unique<nanoaod::FlatTable>(0, this->name_, /*singleton*/ false, /*extension*/ true);

      for (const auto& coltable : this->coltables_) {
        std::vector<uint16_t> emptyCounts;
        std::vector<uint16_t> emptyOffsets;

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

      if (out->nColumns() > 0) {
        out->setDoc(this->doc_);
        iEvent.put(std::move(out));
      }
      return;
    }

    const auto& tracksters_h = iEvent.getHandle(tracksters_token_);
    if (!tracksters_h.isValid() && this->skipNonExistingSrc_) {
      const auto& candidates = *prod;
      const size_t table_size = candidates.size();
      auto out = std::make_unique<nanoaod::FlatTable>(table_size, this->name_, /*singleton*/ false, /*extension*/ true);

      for (const auto& coltable : this->coltables_) {
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

      if (out->nColumns() > 0) {
        out->setDoc(this->doc_);
        iEvent.put(std::move(out));
      }
      return;
    }
    const auto& tracksters = *tracksters_h;

    const auto& tracks_h = iEvent.getHandle(tracks_token_);
    if (!tracks_h.isValid() && this->skipNonExistingSrc_) {
      const auto& candidates = *prod;
      const size_t table_size = candidates.size();
      auto out = std::make_unique<nanoaod::FlatTable>(table_size, this->name_, /*singleton*/ false, /*extension*/ true);

      for (const auto& coltable : this->coltables_) {
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

      if (out->nColumns() > 0) {
        out->setDoc(this->doc_);
        iEvent.put(std::move(out));
      }
      return;
    }
    const auto& tracks = *tracks_h;

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

    for (const auto& cand : candidates) {
      const auto& children = cand.tracksters();
      counts.push_back(children.size());
      coltablesize += children.size();

      for (const auto& t : children) {
        tracksterKeys.push_back(t.key());

        // get the trackster and its track indices
        const auto& trackster = tracksters[t.key()];
        const auto& trackIdxs = trackster.trackIdxs();

        if (!trackIdxs.empty()) {
          // for now, only one track
          const auto trackIdx = trackIdxs[0];
          const auto& track = tracks[trackIdx];

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
            track_boundaryX.push_back(-999);
            track_boundaryY.push_back(-999);
            track_boundaryZ.push_back(-999);
            track_boundaryEta.push_back(-999);
            track_boundaryPhi.push_back(-999);
            track_boundaryPx.push_back(-999);
            track_boundaryPy.push_back(-999);
            track_boundaryPz.push_back(-999);
          }
        } else {
          // no tracks associated with this trackster
          track_boundaryX.push_back(-999);
          track_boundaryY.push_back(-999);
          track_boundaryZ.push_back(-999);
          track_boundaryEta.push_back(-999);
          track_boundaryPhi.push_back(-999);
          track_boundaryPx.push_back(-999);
          track_boundaryPy.push_back(-999);
          track_boundaryPz.push_back(-999);
        }
      }
    }

    for (const auto& coltable : this->coltables_) {
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
    desc.add<std::string>("detector", "HGCAL");
    desc.add<std::string>("propagator", "PropagatorWithMaterial");

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
  const std::string detector_;
  const std::string propName_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TICLCandidateExtraTableProducer);
