#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class PixelVertexProducerFromSoA : public edm::global::EDProducer<> {
public:
  using IndToEdm = std::vector<uint16_t>;

  explicit PixelVertexProducerFromSoA(const edm::ParameterSet &iConfig);
  ~PixelVertexProducerFromSoA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const override;

  edm::EDGetTokenT<ZVertexHeterogeneous> tokenVertex_;
  edm::EDGetTokenT<reco::BeamSpot> tokenBeamSpot_;
  edm::EDGetTokenT<reco::TrackCollection> tokenTracks_;
  edm::EDGetTokenT<IndToEdm> tokenIndToEdm_;
};

PixelVertexProducerFromSoA::PixelVertexProducerFromSoA(const edm::ParameterSet &conf)
    : tokenVertex_(consumes<ZVertexHeterogeneous>(conf.getParameter<edm::InputTag>("src"))),
      tokenBeamSpot_(consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpot"))),
      tokenTracks_(consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("TrackCollection"))),
      tokenIndToEdm_(consumes<IndToEdm>(conf.getParameter<edm::InputTag>("TrackCollection"))) {
  produces<reco::VertexCollection>();
}

void PixelVertexProducerFromSoA::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("TrackCollection", edm::InputTag("pixelTracks"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("src", edm::InputTag("pixelVertexSoA"));

  descriptions.add("pixelVertexFromSoA", desc);
}

void PixelVertexProducerFromSoA::produce(edm::StreamID streamID, edm::Event &iEvent, const edm::EventSetup &) const {
  auto vertexes = std::make_unique<reco::VertexCollection>();

  edm::Handle<reco::TrackCollection> trackCollection;
  iEvent.getByToken(tokenTracks_, trackCollection);
  auto const &tracks = *(trackCollection.product());
  edm::Handle<IndToEdm> indToEdmH;
  iEvent.getByToken(tokenIndToEdm_, indToEdmH);
  auto const &indToEdm = *indToEdmH;

  edm::Handle<reco::BeamSpot> bsHandle;
  iEvent.getByToken(tokenBeamSpot_, bsHandle);

  float x0 = 0, y0 = 0, z0 = 0, dxdz = 0, dydz = 0;
  std::vector<int32_t> itrk;
  if (!bsHandle.isValid()) {
    edm::LogWarning("PixelVertexProducer") << "No beamspot found. returning vertexes with (0,0,Z) ";
  } else {
    const reco::BeamSpot &bs = *bsHandle;
    x0 = bs.x0();
    y0 = bs.y0();
    z0 = bs.z0();
    dxdz = bs.dxdz();
    dydz = bs.dydz();
  }

  auto const &soa = *(iEvent.get(tokenVertex_).get());

  int nv = soa.nvFinal;

  // std::cout << "converting " << nv << " vertices " << " from " << indToEdm.size() << " tracks" << std::endl;

  std::set<uint16_t> uind;  // fort verifing index consistency
  for (int j = nv - 1; j >= 0; --j) {
    auto i = soa.sortInd[j];  // on gpu sorted in ascending order....
    assert(i < nv);
    uind.insert(i);
    assert(itrk.empty());
    auto z = soa.zv[i];
    auto x = x0 + dxdz * z;
    auto y = y0 + dydz * z;
    z += z0;
    reco::Vertex::Error err;
    err(2, 2) = 1.f / soa.wv[i];
    err(2, 2) *= 2.;  // artifically inflate error
    //Copy also the tracks (no intention to be efficient....)
    for (auto k = 0U; k < indToEdm.size(); ++k) {
      if (soa.idv[k] == int16_t(i))
        itrk.push_back(k);
    }
    auto nt = itrk.size();
    if (nt == 0) {
      std::cout << "vertex " << i << " with no tracks..." << std::endl;
      continue;
    }
    if (nt < 2) {
      itrk.clear();
      continue;
    }  // remove outliers
    (*vertexes).emplace_back(reco::Vertex::Point(x, y, z), err, soa.chi2[i], soa.ndof[i], nt);
    auto &v = (*vertexes).back();
    for (auto it : itrk) {
      assert(it < int(indToEdm.size()));
      auto k = indToEdm[it];
      if (k > tracks.size()) {
        edm::LogWarning("PixelVertexProducer") << "oops track " << it << " does not exists on CPU " << k;
        continue;
      }
      auto tk = reco::TrackRef(trackCollection, k);
      v.add(reco::TrackBaseRef(tk));
    }
    itrk.clear();
  }

  LogDebug("PixelVertexProducer") << ": Found " << vertexes->size() << " vertexes\n";
  for (unsigned int i = 0; i < vertexes->size(); ++i) {
    LogDebug("PixelVertexProducer") << "Vertex number " << i << " has " << (*vertexes)[i].tracksSize()
                                    << " tracks with a position of " << (*vertexes)[i].z() << " +- "
                                    << std::sqrt((*vertexes)[i].covariance(2, 2));
  }

  // legacy logic....
  if (vertexes->empty() && bsHandle.isValid()) {
    const reco::BeamSpot &bs = *bsHandle;

    GlobalError bse(bs.rotatedCovariance3D());
    if ((bse.cxx() <= 0.) || (bse.cyy() <= 0.) || (bse.czz() <= 0.)) {
      AlgebraicSymMatrix33 we;
      we(0, 0) = 10000;
      we(1, 1) = 10000;
      we(2, 2) = 10000;
      vertexes->push_back(reco::Vertex(bs.position(), we, 0., 0., 0));

      edm::LogInfo("PixelVertexProducer") << "No vertices found. Beamspot with invalid errors " << bse.matrix()
                                          << "\nWill put Vertex derived from dummy-fake BeamSpot into Event.\n"
                                          << (*vertexes)[0].x() << "\n"
                                          << (*vertexes)[0].y() << "\n"
                                          << (*vertexes)[0].z() << "\n";
    } else {
      vertexes->push_back(reco::Vertex(bs.position(), bs.rotatedCovariance3D(), 0., 0., 0));

      edm::LogInfo("PixelVertexProducer") << "No vertices found. Will put Vertex derived from BeamSpot into Event:\n"
                                          << (*vertexes)[0].x() << "\n"
                                          << (*vertexes)[0].y() << "\n"
                                          << (*vertexes)[0].z() << "\n";
    }
  } else if (vertexes->empty() && !bsHandle.isValid()) {
    edm::LogWarning("PixelVertexProducer") << "No beamspot and no vertex found. No vertex returned.";
  }

  iEvent.put(std::move(vertexes));
}

DEFINE_FWK_MODULE(PixelVertexProducerFromSoA);
