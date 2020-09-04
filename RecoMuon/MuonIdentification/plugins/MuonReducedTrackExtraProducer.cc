#include "RecoMuon/MuonIdentification/plugins/MuonReducedTrackExtraProducer.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

MuonReducedTrackExtraProducer::MuonReducedTrackExtraProducer(const edm::ParameterSet& pset)
    : muonToken_(consumes<edm::View<reco::Muon>>(pset.getParameter<edm::InputTag>("muonTag"))),
      outputClusters_(pset.getParameter<bool>("outputClusters")),
      selector_(pset.getParameter<std::string>("cut")),
      trackExtraOutToken_(produces<reco::TrackExtraCollection>()),
      trackingRecHitsOutToken_(produces<TrackingRecHitCollection>()),
      associationOutToken_(produces<edm::Association<reco::TrackExtraCollection>>()) {
  std::vector<edm::InputTag> trackExtraTags = pset.getParameter<std::vector<edm::InputTag>>("trackExtraTags");
  for (edm::InputTag const& tag : trackExtraTags) {
    trackExtraTokens_.push_back(consumes<reco::TrackExtraCollection>(tag));
  }

  std::vector<edm::InputTag> trackExtraAssocTags = pset.getParameter<std::vector<edm::InputTag>>("trackExtraAssocs");
  for (edm::InputTag const& tag : trackExtraAssocTags) {
    trackExtraAssocs_.push_back(consumes<edm::Association<reco::TrackExtraCollection>>(tag));
  }

  if (outputClusters_) {
    pixelClusterToken_ =
        consumes<edmNew::DetSetVector<SiPixelCluster>>(pset.getParameter<edm::InputTag>("pixelClusterTag"));
    stripClusterToken_ =
        consumes<edmNew::DetSetVector<SiStripCluster>>(pset.getParameter<edm::InputTag>("stripClusterTag"));

    pixelClusterOutToken_ = produces<edmNew::DetSetVector<SiPixelCluster>>();
    stripClusterOutToken_ = produces<edmNew::DetSetVector<SiStripCluster>>();
  }
}

void MuonReducedTrackExtraProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment(
      "Produces reduced set of TrackExtras and corresponding TrackingRecHits and (optionally) Pixe/Strip clusters "
      "associated to a muon track.");
  desc.add<edm::InputTag>("muonTag", edm::InputTag("muons"));
  desc.add<std::vector<edm::InputTag>>("trackExtraTags",
                                       {edm::InputTag("generalTracks"),
                                        edm::InputTag("globalMuons"),
                                        edm::InputTag("tevMuons", "firstHit"),
                                        edm::InputTag("tevMuons", "picky"),
                                        edm::InputTag("tevMuons", "dyt")});
  desc.add<std::vector<edm::InputTag>>("trackExtraAssocs", {});
  desc.add<edm::InputTag>("pixelClusterTag", edm::InputTag("siPixelClusters"));
  desc.add<edm::InputTag>("stripClusterTag", edm::InputTag("siStripClusters"));
  desc.add<std::string>("cut", "pt > 3.0");
  desc.add<bool>("outputClusters", true);
  descriptions.add("muonReducedTrackExtras", desc);
}

void MuonReducedTrackExtraProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  auto muons = event.getHandle(muonToken_);

  std::vector<edm::Handle<reco::TrackExtraCollection>> trackExtrasV(trackExtraTokens_.size());
  for (unsigned int i = 0; i < trackExtraTokens_.size(); ++i) {
    event.getByToken(trackExtraTokens_[i], trackExtrasV[i]);
  }

  std::vector<edm::Handle<edm::Association<reco::TrackExtraCollection>>> trackExtraAssocs(trackExtraAssocs_.size());
  for (unsigned int i = 0; i < trackExtraAssocs_.size(); ++i) {
    event.getByToken(trackExtraAssocs_[i], trackExtraAssocs[i]);
  }

  std::map<edm::ProductID, std::vector<bool>> idxstokeep;
  for (auto const& trackExtras : trackExtrasV) {
    idxstokeep[trackExtras.id()].resize(trackExtras->size(), false);
  }

  //loop over muons and mark track extras to keep
  for (auto const& muon : *muons) {
    if (!selector_(muon)) {
      continue;
    }
    reco::TrackExtraRef trackExtra = muon.bestTrack()->extra();
    // check recursively through association maps if present
    for (auto const& assoc : trackExtraAssocs) {
      if (!assoc->contains(trackExtra.id())) {
        continue;
      }
      reco::TrackExtraRef const& trackExtraOut = (*assoc)[trackExtra];
      if (trackExtraOut.isNonnull()) {
        trackExtra = trackExtraOut;
      }
    }
    auto idxs = idxstokeep.find(trackExtra.id());
    if (idxs != idxstokeep.end()) {
      idxs->second[trackExtra.key()] = true;
    }
  }

  //output collections for TrackExtras and TrackingRecHits and the association map
  reco::TrackExtraCollection trackExtrasOut;
  TrackingRecHitCollection trackingRecHitsOut;
  edm::Association<reco::TrackExtraCollection> association;
  edm::Association<reco::TrackExtraCollection>::Filler assocfiller(association);

  //refprod for the output TrackExtraCollection
  edm::RefProd<reco::TrackExtraCollection> trackExtraRefProd = event.getRefBeforePut(trackExtraOutToken_);
  //refprod for the output TrackingRecHitCollection
  edm::RefProd<TrackingRecHitCollection> hitRefProd = event.getRefBeforePut(trackingRecHitsOutToken_);

  association.setRef(trackExtraRefProd);

  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> pixelClusters;
  edm::Handle<edmNew::DetSetVector<SiStripCluster>> stripClusters;

  //indexes of pixel/strip clusters to keep
  std::vector<bool> pixelClustersToKeep;
  std::vector<bool> stripClustersToKeep;

  if (outputClusters_) {
    event.getByToken(pixelClusterToken_, pixelClusters);
    event.getByToken(stripClusterToken_, stripClusters);

    pixelClustersToKeep.resize(pixelClusters->dataSize(), false);
    stripClustersToKeep.resize(stripClusters->dataSize(), false);
  }

  using SiPixelClusterRef = edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster>;
  using SiStripClusterRef = edm::Ref<edmNew::DetSetVector<SiStripCluster>, SiStripCluster>;

  //loop over track extras and fill output together with TrackingRechits
  //as well as marking pixel and strip clusters to keep
  for (auto const& trackExtras : trackExtrasV) {
    const std::vector<bool>& idxs = idxstokeep.at(trackExtras.id());
    //indices for association (-1 for null association)
    std::vector<int> associdxs(trackExtras->size(), -1);
    for (unsigned int i = 0; i < trackExtras->size(); ++i) {
      if (!idxs[i]) {
        continue;
      }
      const reco::TrackExtra& trackExtra = (*trackExtras)[i];

      //fill association idx
      associdxs[i] = trackExtrasOut.size();

      //fill TrackExtra
      trackExtrasOut.emplace_back(trackExtra.outerPosition(),
                                  trackExtra.outerMomentum(),
                                  trackExtra.outerOk(),
                                  trackExtra.innerPosition(),
                                  trackExtra.innerMomentum(),
                                  trackExtra.innerOk(),
                                  trackExtra.outerStateCovariance(),
                                  trackExtra.outerDetId(),
                                  trackExtra.innerStateCovariance(),
                                  trackExtra.innerDetId(),
                                  trackExtra.seedDirection(),
                                  trackExtra.seedRef());

      //rekey refs to TrackingRecHits
      reco::TrackExtra& trackExtraOut = trackExtrasOut.back();
      trackExtraOut.setHits(hitRefProd, trackingRecHitsOut.size(), trackExtra.recHitsSize());
      for (auto const& hit : trackExtra.recHits()) {
        if (outputClusters_) {
          //mark clusters to keep
          TrackerSingleRecHit const* singleHit = dynamic_cast<TrackerSingleRecHit const*>(&*hit);
          if (singleHit != nullptr) {
            SiPixelClusterRef const& pixelRef = singleHit->cluster_pixel();
            if (pixelRef.isNonnull() && pixelRef.id() == pixelClusters.id()) {
              pixelClustersToKeep[pixelRef.key()] = true;
            }
            SiStripClusterRef const& stripRef = singleHit->cluster_strip();
            if (stripRef.isNonnull() && stripRef.id() == stripClusters.id()) {
              stripClustersToKeep[stripRef.key()] = true;
            }
            SiStripMatchedRecHit2D const* matched2DHit = dynamic_cast<SiStripMatchedRecHit2D const*>(singleHit);
            if (matched2DHit != nullptr) {
              SiStripClusterRef const& monoRef = matched2DHit->monoClusterRef().cluster_strip();
              SiStripClusterRef const& stereoRef = matched2DHit->stereoClusterRef().cluster_strip();
              if (monoRef.isNonnull() && monoRef.id() == stripClusters.id()) {
                stripClustersToKeep[monoRef.key()] = true;
              }
              if (stereoRef.isNonnull() && stereoRef.id() == stripClusters.id()) {
                stripClustersToKeep[stereoRef.key()] = true;
              }
            }
          }
        }
        //fill output hit
        trackingRecHitsOut.push_back(hit->clone());
      }
    }
    assocfiller.insert(trackExtras, associdxs.begin(), associdxs.end());
  }

  assocfiller.fill();

  if (outputClusters_) {
    //output collections for clusters
    edmNew::DetSetVector<SiPixelCluster> pixelClustersOut;
    edmNew::DetSetVector<SiStripCluster> stripClustersOut;
    //mapping of indices from input to output collections
    std::unordered_map<unsigned int, unsigned int> pixelClusterIdxMap;
    std::unordered_map<unsigned int, unsigned int> stripClusterIdxMap;

    //fill output clusters
    //this indexes the internal data array of the DetSetVector
    unsigned int iIndex = 0;
    //loop over edmNew:::DetSet<T>
    for (auto setIter = pixelClusters->begin(), setIterEnd = pixelClusters->end(); setIter != setIterEnd; ++setIter) {
      //fill items from this DetSet
      typename edmNew::DetSetVector<SiPixelCluster>::FastFiller ff(pixelClustersOut, setIter->detId());
      for (auto iter = setIter->begin(), iterEnd = setIter->end(); iter != iterEnd; ++iter, ++iIndex) {
        if (pixelClustersToKeep[iIndex]) {
          ff.push_back(*iter);
          const unsigned int oIndex = pixelClusterIdxMap.size();
          pixelClusterIdxMap[iIndex] = oIndex;
        }
      }
    }

    iIndex = 0;
    for (auto setIter = stripClusters->begin(), setIterEnd = stripClusters->end(); setIter != setIterEnd; ++setIter) {
      //fill items from this DetSet
      typename edmNew::DetSetVector<SiStripCluster>::FastFiller ff(stripClustersOut, setIter->detId());
      for (auto iter = setIter->begin(), iterEnd = setIter->end(); iter != iterEnd; ++iter, ++iIndex) {
        if (stripClustersToKeep[iIndex]) {
          ff.push_back(*iter);
          const unsigned int oIndex = stripClusterIdxMap.size();
          stripClusterIdxMap[iIndex] = oIndex;
        }
      }
    }

    edm::OrphanHandle<edmNew::DetSetVector<SiPixelCluster>> pixelClustersOutH =
        event.emplace(pixelClusterOutToken_, std::move(pixelClustersOut));
    edm::OrphanHandle<edmNew::DetSetVector<SiStripCluster>> stripClustersOutH =
        event.emplace(stripClusterOutToken_, std::move(stripClustersOut));

    //rekey cluster references in output hit collection
    for (auto& hit : trackingRecHitsOut) {
      TrackerSingleRecHit* singleHit = dynamic_cast<TrackerSingleRecHit*>(&hit);
      if (singleHit != nullptr) {
        SiPixelClusterRef const& pixelRef = singleHit->cluster_pixel();
        if (pixelRef.isNonnull() && pixelRef.id() == pixelClusters.id()) {
          SiPixelClusterRef const pixelRefOut(pixelClustersOutH, pixelClusterIdxMap.at(pixelRef.key()));
          singleHit->setClusterPixelRef(pixelRefOut);
        }
        SiStripClusterRef const& stripRef = singleHit->cluster_strip();
        if (stripRef.isNonnull() && stripRef.id() == stripClusters.id()) {
          SiStripClusterRef const stripRefOut(stripClustersOutH, stripClusterIdxMap.at(stripRef.key()));
          singleHit->setClusterStripRef(stripRefOut);
        }
        SiStripMatchedRecHit2D* matched2DHit = dynamic_cast<SiStripMatchedRecHit2D*>(singleHit);
        if (matched2DHit != nullptr) {
          SiStripClusterRef const& monoRef = matched2DHit->monoClusterRef().cluster_strip();
          SiStripClusterRef const& stereoRef = matched2DHit->stereoClusterRef().cluster_strip();
          if (monoRef.isNonnull() && monoRef.id() == stripClusters.id()) {
            SiStripClusterRef const monoRefOut(stripClustersOutH, stripClusterIdxMap.at(monoRef.key()));
            matched2DHit->monoClusterRef() = OmniClusterRef(monoRefOut);
          }
          if (stereoRef.isNonnull() && stereoRef.id() == stripClusters.id()) {
            SiStripClusterRef const stereoRefOut(stripClustersOutH, stripClusterIdxMap.at(stereoRef.key()));
            matched2DHit->stereoClusterRef() = OmniClusterRef(stereoRefOut);
          }
        }
      }
    }
  }

  event.emplace(trackExtraOutToken_, std::move(trackExtrasOut));
  event.emplace(trackingRecHitsOutToken_, std::move(trackingRecHitsOut));
  event.emplace(associationOutToken_, std::move(association));
}
