#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "Utilities/General/interface/ClassName.h"

#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"

class TkAlCaOverlapTagger : public edm::stream::EDProducer<> {
public:
  TkAlCaOverlapTagger(const edm::ParameterSet& iConfig);
  ~TkAlCaOverlapTagger() override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> trajTrackToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> siPixelClustersToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> siStripClustersToken_;
  edm::InputTag src_;
  edm::InputTag srcClust_;
  bool rejectBadMods_;
  std::vector<unsigned int> BadModsList_;

  int layerFromId(const DetId& id, const TrackerTopology* tTopo) const;
};

TkAlCaOverlapTagger::TkAlCaOverlapTagger(const edm::ParameterSet& iConfig)
    : topoToken_(esConsumes()),
      trajTrackToken_(consumes((iConfig.getParameter<edm::InputTag>("src")))),
      siPixelClustersToken_(consumes((iConfig.getParameter<edm::InputTag>("Clustersrc")))),
      siStripClustersToken_(consumes((iConfig.getParameter<edm::InputTag>("Clustersrc")))),
      rejectBadMods_(iConfig.getParameter<bool>("rejectBadMods")),
      BadModsList_(iConfig.getParameter<std::vector<uint32_t>>("BadMods")) {
  produces<AliClusterValueMap>();  //produces the ValueMap (VM) to be stored in the Event at the end
}

TkAlCaOverlapTagger::~TkAlCaOverlapTagger() = default;

void TkAlCaOverlapTagger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Retrieve tracker topology from geometry
  const TrackerTopology* tTopo = &iSetup.getData(topoToken_);

  edm::Handle<TrajTrackAssociationCollection> assoMap;
  iEvent.getByToken(trajTrackToken_, assoMap);
  LogDebug("TkAlCaOverlapTagger") << "\n\n############################\n###  Starting a new TkAlCaOverlapTagger - Ev "
                                  << iEvent.id().run() << ", " << iEvent.id().event();

  AlignmentClusterFlag iniflag;
  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> pixelclusters;
  iEvent.getByToken(siPixelClustersToken_, pixelclusters);  //same label as tracks
  std::vector<AlignmentClusterFlag> pixelvalues(pixelclusters->dataSize(),
                                                iniflag);  //vector where to store value to be fileld in the VM

  edm::Handle<edmNew::DetSetVector<SiStripCluster>> stripclusters;
  iEvent.getByToken(siStripClustersToken_, stripclusters);  //same label as tracks
  std::vector<AlignmentClusterFlag> stripvalues(stripclusters->dataSize(),
                                                iniflag);  //vector where to store value to be fileld in the VM

  //start doing the thing!

  //loop over trajectories
  for (TrajTrackAssociationCollection::const_iterator itass = assoMap->begin(); itass != assoMap->end(); ++itass) {
    int nOverlaps = 0;
    const edm::Ref<std::vector<Trajectory>> traj = itass->key;  //trajectory in the collection
    const Trajectory* myTrajectory = &(*traj);
    std::vector<TrajectoryMeasurement> tmColl = myTrajectory->measurements();

    const reco::TrackRef tkref = itass->val;  //associated track track in the collection
    // const Track * trk = &(*tkref);
    int hitcnt = -1;

    //loop over traj meas
    const TrajectoryMeasurement* previousTM(nullptr);
    DetId previousId(0);

    for (std::vector<TrajectoryMeasurement>::const_iterator itTrajMeas = tmColl.begin(); itTrajMeas != tmColl.end();
         ++itTrajMeas) {
      hitcnt++;

      if (previousTM != nullptr) {
        LogDebug("TkAlCaOverlapTagger") << "Checking TrajMeas (" << hitcnt + 1 << "):";
        if (!previousTM->recHit()->isValid()) {
          LogDebug("TkAlCaOverlapTagger") << "Previous RecHit invalid !";
          continue;
        } else {
          LogDebug("TkAlCaOverlapTagger") << "\nDetId: " << std::flush << previousTM->recHit()->geographicalId().rawId()
                                          << "\t Local x of hit: " << previousTM->recHit()->localPosition().x();
        }
      } else {
        LogDebug("TkAlCaOverlapTagger") << "This is the first Traj Meas of the Trajectory! The Trajectory contains "
                                        << tmColl.size() << " TrajMeas";
      }

      TrackingRecHit::ConstRecHitPointer hitpointer = itTrajMeas->recHit();
      const TrackingRecHit* hit = &(*hitpointer);
      if (!hit->isValid())
        continue;

      DetId detid = hit->geographicalId();
      int layer(layerFromId(detid, tTopo));  //layer 1-4=TIB, layer 5-10=TOB
      int subDet = detid.subdetId();

      if ((previousTM != nullptr) && (layer != -1)) {
        for (std::vector<TrajectoryMeasurement>::const_iterator itmCompare = itTrajMeas - 1;
             itmCompare >= tmColl.begin() && itmCompare > itTrajMeas - 4;
             --itmCompare) {
          DetId compareId = itmCompare->recHit()->geographicalId();
          if (subDet != compareId.subdetId() || layer != layerFromId(compareId, tTopo))
            break;
          if (!itmCompare->recHit()->isValid())
            continue;
          if ((subDet <= 2) ||
              (subDet > 2 && SiStripDetId(detid).stereo() ==
                                 SiStripDetId(compareId).stereo())) {  //if either pixel or strip stereo module

            //
            //WOW, we have an overlap!!!!!!
            //
            AlignmentClusterFlag hitflag(hit->geographicalId());
            hitflag.SetOverlapFlag();
            LogDebug("TkAlCaOverlapTagger") << "Overlap found in SubDet " << subDet << "!!!" << std::flush;

            bool hitInStrip = (subDet == SiStripDetId::TIB) || (subDet == SiStripDetId::TID) ||
                              (subDet == SiStripDetId::TOB) || (subDet == SiStripDetId::TEC);
            if (hitInStrip) {
              LogDebug("TkAlCaOverlapTagger") << "  TypeId of the RecHit: " << className(*hit);
              // const std::type_info &type = typeid(*hit);
              const SiStripRecHit2D* transstriphit2D = dynamic_cast<const SiStripRecHit2D*>(hit);
              const SiStripRecHit1D* transstriphit1D = dynamic_cast<const SiStripRecHit1D*>(hit);

              //   if (type == typeid(SiStripRecHit1D)) {
              if (transstriphit1D != nullptr) {
                //	const SiStripRecHit1D* striphit=dynamic_cast<const  SiStripRecHit1D*>(hit);
                const SiStripRecHit1D* striphit = transstriphit1D;
                if (striphit != nullptr) {
                  SiStripRecHit1D::ClusterRef stripclust(striphit->cluster());

                  if (stripclust.id() ==
                      stripclusters
                          .id()) {  //ensure that the stripclust is really present in the original cluster collection!!!
                    stripvalues[stripclust.key()] = hitflag;
                  } else {
                    edm::LogError("TkAlCaOverlapTagger")
                        << "ERROR in <TkAlCaOverlapTagger::produce>: ProdId of Strip clusters mismatched: "
                        << stripclust.id() << " vs " << stripclusters.id();
                  }
                } else {
                  edm::LogError("TkAlCaOverlapTagger") << "ERROR in <TkAlCaOverlapTagger::produce>: Dynamic cast of "
                                                          "Strip RecHit failed!   TypeId of the RecHit: "
                                                       << className(*hit);
                }
              }  //end if sistriprechit1D
              else if (transstriphit2D != nullptr) {
                //else if (type == typeid(SiStripRecHit2D)) {
                //		const SiStripRecHit2D* striphit=dynamic_cast<const  SiStripRecHit2D*>(hit);
                const SiStripRecHit2D* striphit = transstriphit2D;
                if (striphit != nullptr) {
                  SiStripRecHit2D::ClusterRef stripclust(striphit->cluster());

                  if (stripclust.id() ==
                      stripclusters
                          .id()) {  //ensure that the stripclust is really present in the original cluster collection!!!
                    stripvalues[stripclust.key()] = hitflag;

                    LogDebug("TkAlCaOverlapTagger")
                        << ">>> Storing in the ValueMap a StripClusterRef with Cluster.Key: " << stripclust.key()
                        << " (" << striphit->cluster().key() << "), Cluster.Id: " << stripclust.id() << "  (DetId is "
                        << hit->geographicalId().rawId() << ")";
                  } else {
                    edm::LogError("TkAlCaOverlapTagger")
                        << "ERROR in <TkAlCaOverlapTagger::produce>: ProdId of Strip clusters mismatched: "
                        << stripclust.id() << " vs " << stripclusters.id();
                  }

                  LogDebug("TkAlCaOverlapTagger") << "Cluster baricentre: " << stripclust->barycenter();
                } else {
                  edm::LogError("TkAlCaOverlapTagger") << "ERROR in <TkAlCaOverlapTagger::produce>: Dynamic cast of "
                                                          "Strip RecHit failed!   TypeId of the RecHit: "
                                                       << className(*hit);
                }
              }  //end if Sistriprechit2D
              else {
                edm::LogError("TkAlCaOverlapTagger") << "ERROR in <TkAlCaOverlapTagger::produce>: Impossible to "
                                                        "determine the type of SiStripRecHit.  TypeId of the RecHit: "
                                                     << className(*hit);
              }

            }       //end if hit in Strips
            else {  //pixel hit
              const SiPixelRecHit* transpixelhit = dynamic_cast<const SiPixelRecHit*>(hit);
              if (transpixelhit != nullptr) {
                const SiPixelRecHit* pixelhit = transpixelhit;
                SiPixelClusterRefNew pixclust(pixelhit->cluster());

                if (pixclust.id() == pixelclusters.id()) {
                  pixelvalues[pixclust.key()] = hitflag;
                  LogDebug("TkAlCaOverlapTagger")
                      << ">>> Storing in the ValueMap a PixelClusterRef with ProdID: " << pixclust.id()
                      << "  (DetId is " << hit->geographicalId().rawId() << ")";  //"  and  a Val with ID: "<<flag.id();
                } else {
                  edm::LogError("TkAlCaOverlapTagger")
                      << "ERROR in <TkAlCaOverlapTagger::produce>: ProdId of Pixel clusters mismatched: "
                      << pixclust.id() << " vs " << pixelclusters.id();
                }
              } else {
                edm::LogError("TkAlCaOverlapTagger") << "ERROR in <TkAlCaOverlapTagger::produce>: Dynamic cast of "
                                                        "Pixel RecHit failed!   TypeId of the RecHit: "
                                                     << className(*hit);
              }
            }  //end 'else' it is a pixel hit

            nOverlaps++;
            break;
          }
        }  //end second loop on TM
      }    //end if a previous TM exists

      previousTM = &(*itTrajMeas);
      previousId = detid;
    }  //end loop over traj meas
    LogDebug("TkAlCaOverlapTagger") << "Found " << nOverlaps << " overlaps in this trajectory";
  }  //end loop over trajectories

  // prepare output
  auto hitvalmap = std::make_unique<AliClusterValueMap>();
  AliClusterValueMap::Filler mapfiller(*hitvalmap);

  edm::TestHandle<std::vector<AlignmentClusterFlag>> fakePixelHandle(&pixelvalues, pixelclusters.id());
  mapfiller.insert(fakePixelHandle, pixelvalues.begin(), pixelvalues.end());

  edm::TestHandle<std::vector<AlignmentClusterFlag>> fakeStripHandle(&stripvalues, stripclusters.id());
  mapfiller.insert(fakeStripHandle, stripvalues.begin(), stripvalues.end());
  mapfiller.fill();

  // iEvent.put(std::move(stripmap));
  iEvent.put(std::move(hitvalmap));
}  //end  TkAlCaOverlapTagger::produce
int TkAlCaOverlapTagger::layerFromId(const DetId& id, const TrackerTopology* tTopo) const {
  if (uint32_t(id.subdetId()) == PixelSubdetector::PixelBarrel) {
    return tTopo->pxbLayer(id);
  } else if (uint32_t(id.subdetId()) == PixelSubdetector::PixelEndcap) {
    return tTopo->pxfDisk(id) + (3 * (tTopo->pxfSide(id) - 1));
  } else if (id.subdetId() == StripSubdetector::TIB) {
    return tTopo->tibLayer(id);
  } else if (id.subdetId() == StripSubdetector::TOB) {
    return tTopo->tobLayer(id);
  } else if (id.subdetId() == StripSubdetector::TEC) {
    return tTopo->tecWheel(id) + (9 * (tTopo->tecSide(id) - 1));
  } else if (id.subdetId() == StripSubdetector::TID) {
    return tTopo->tidWheel(id) + (3 * (tTopo->tidSide(id) - 1));
  }
  return -1;

}  //end layerfromId

void TkAlCaOverlapTagger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Tagger of overlaps for tracker alignment");
  desc.add<edm::InputTag>("src", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("Clustersrc", edm::InputTag("ALCARECOTkAlCosmicsCTF0T"));
  desc.add<bool>("rejectBadMods", false);
  desc.add<std::vector<uint32_t>>("BadMods", {});
  descriptions.addWithDefaultLabel(desc);
}

// ========= MODULE DEF ==============
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TkAlCaOverlapTagger);
