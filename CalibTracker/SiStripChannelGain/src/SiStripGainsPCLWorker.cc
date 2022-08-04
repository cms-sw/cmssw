#include "CalibTracker/SiStripChannelGain/interface/SiStripGainsPCLWorker.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <sstream>

//********************************************************************************//
SiStripGainsPCLWorker::SiStripGainsPCLWorker(const edm::ParameterSet& iConfig) {
  MinTrackMomentum = iConfig.getUntrackedParameter<double>("minTrackMomentum", 3.0);
  MaxTrackMomentum = iConfig.getUntrackedParameter<double>("maxTrackMomentum", 99999.0);
  MinTrackEta = iConfig.getUntrackedParameter<double>("minTrackEta", -5.0);
  MaxTrackEta = iConfig.getUntrackedParameter<double>("maxTrackEta", 5.0);
  MaxNrStrips = iConfig.getUntrackedParameter<unsigned>("maxNrStrips", 2);
  MinTrackHits = iConfig.getUntrackedParameter<unsigned>("MinTrackHits", 8);
  MaxTrackChiOverNdf = iConfig.getUntrackedParameter<double>("MaxTrackChiOverNdf", 3);
  MaxTrackingIteration = iConfig.getUntrackedParameter<int>("MaxTrackingIteration", 7);
  AllowSaturation = iConfig.getUntrackedParameter<bool>("AllowSaturation", false);
  FirstSetOfConstants = iConfig.getUntrackedParameter<bool>("FirstSetOfConstants", true);
  Validation = iConfig.getUntrackedParameter<bool>("Validation", false);
  OldGainRemoving = iConfig.getUntrackedParameter<bool>("OldGainRemoving", false);
  useCalibration = iConfig.getUntrackedParameter<bool>("UseCalibration", false);
  doChargeMonitorPerPlane = iConfig.getUntrackedParameter<bool>("doChargeMonitorPerPlane", false);
  m_DQMdir = iConfig.getUntrackedParameter<std::string>("DQMdir", "AlCaReco/SiStripGains");
  m_calibrationMode = iConfig.getUntrackedParameter<std::string>("calibrationMode", "StdBunch");
  VChargeHisto = iConfig.getUntrackedParameter<std::vector<std::string>>("ChargeHisto");

  // fill in the mapping between the histogram indices and the (id,side,plane) tuple
  std::vector<std::pair<std::string, std::string>> hnames =
      APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    int id = APVGain::subdetectorId((hnames[i]).first);
    int side = APVGain::subdetectorSide((hnames[i]).first);
    int plane = APVGain::subdetectorPlane((hnames[i]).first);
    int thick = APVGain::thickness((hnames[i]).first);
    std::string s = hnames[i].first;

    auto loc = APVloc(thick, id, side, plane, s);
    theTopologyMap.insert(std::make_pair(i, loc));
  }

  //Set the monitoring element tag and store
  dqm_tag_.reserve(7);
  dqm_tag_.clear();
  dqm_tag_.push_back("StdBunch");    // statistic collection from Standard Collision Bunch @ 3.8 T
  dqm_tag_.push_back("StdBunch0T");  // statistic collection from Standard Collision Bunch @ 0 T
  dqm_tag_.push_back("AagBunch");    // statistic collection from First Collision After Abort Gap @ 3.8 T
  dqm_tag_.push_back("AagBunch0T");  // statistic collection from First Collision After Abort Gap @ 0 T
  dqm_tag_.push_back("IsoMuon");     // statistic collection from Isolated Muon @ 3.8 T
  dqm_tag_.push_back("IsoMuon0T");   // statistic collection from Isolated Muon @ 0 T
  dqm_tag_.push_back("Harvest");     // statistic collection: Harvest

  m_tracks_token = consumes<edm::View<reco::Track>>(iConfig.getParameter<edm::InputTag>("tracks"));
  m_association_token = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("tracks"));

  tTopoToken_ = esConsumes();
  tTopoTokenBR_ = esConsumes<edm::Transition::BeginRun>();
  tkGeomTokenBR_ = esConsumes<edm::Transition::BeginRun>();
  tkGeomToken_ = esConsumes<>();
  gainToken_ = esConsumes<edm::Transition::BeginRun>();
  qualityToken_ = esConsumes<edm::Transition::BeginRun>();
}

//********************************************************************************//
void SiStripGainsPCLWorker::dqmBeginRun(edm::Run const& run,
                                        edm::EventSetup const& iSetup,
                                        APVGain::APVGainHistograms& histograms) const {
  using namespace edm;
  static constexpr float defaultGainTick = 690. / 640.;

  // fills the APV collections at each begin run
  const TrackerGeometry* bareTkGeomPtr = &iSetup.getData(tkGeomTokenBR_);
  const TrackerTopology* bareTkTopoPtr = &iSetup.getData(tTopoTokenBR_);
  checkBookAPVColls(bareTkGeomPtr, bareTkTopoPtr, histograms);

  const auto gainHandle = iSetup.getHandle(gainToken_);
  if (!gainHandle.isValid()) {
    edm::LogError("SiStripGainPCLWorker") << "gainHandle is not valid\n";
    exit(0);
  }

  const auto& siStripQuality = iSetup.getData(qualityToken_);

  for (unsigned int a = 0; a < histograms.APVsCollOrdered.size(); a++) {
    std::shared_ptr<stAPVGain> APV = histograms.APVsCollOrdered[a];

    if (APV->SubDet == PixelSubdetector::PixelBarrel || APV->SubDet == PixelSubdetector::PixelEndcap)
      continue;

    APV->isMasked = siStripQuality.IsApvBad(APV->DetId, APV->APVId);

    if (gainHandle->getNumberOfTags() != 2) {
      edm::LogError("SiStripGainPCLWorker") << "NUMBER OF GAIN TAG IS EXPECTED TO BE 2\n";
      fflush(stdout);
      exit(0);
    };
    float newPreviousGain = gainHandle->getApvGain(APV->APVId, gainHandle->getRange(APV->DetId, 1), 1);
    if (APV->PreviousGain != 1 and newPreviousGain != APV->PreviousGain)
      edm::LogWarning("SiStripGainPCLWorker") << "WARNING: ParticleGain in the global tag changed\n";
    APV->PreviousGain = newPreviousGain;

    float newPreviousGainTick =
        APV->isMasked ? defaultGainTick : gainHandle->getApvGain(APV->APVId, gainHandle->getRange(APV->DetId, 0), 0);
    if (APV->PreviousGainTick != 1 and newPreviousGainTick != APV->PreviousGainTick) {
      edm::LogWarning("SiStripGainPCLWorker")
          << "WARNING: TickMarkGain in the global tag changed\n"
          << std::endl
          << " APV->SubDet: " << APV->SubDet << " APV->APVId:" << APV->APVId << std::endl
          << " APV->PreviousGainTick: " << APV->PreviousGainTick << " newPreviousGainTick: " << newPreviousGainTick
          << std::endl;
    }
    APV->PreviousGainTick = newPreviousGainTick;
  }
}

namespace {
  struct HitCluster {
    uint32_t det;
    const SiStripCluster* strip;
    const SiPixelCluster* pixel;
    HitCluster(uint32_t detId, const SiStripCluster* strip, const SiPixelCluster* pixel)
        : det(detId), strip(strip), pixel(pixel) {}
  };
  std::vector<HitCluster> getClusters(const TrackingRecHit* hit) {
    const auto simple1d = dynamic_cast<const SiStripRecHit1D*>(hit);
    const auto simple = dynamic_cast<const SiStripRecHit2D*>(hit);
    const auto matched = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
    const auto pixel = dynamic_cast<const SiPixelRecHit*>(hit);
    std::vector<HitCluster> clusters;
    if (matched) {
      clusters.emplace_back(matched->monoId(), &matched->monoCluster(), nullptr);
      clusters.emplace_back(matched->stereoId(), &matched->stereoCluster(), nullptr);
    } else if (simple) {
      clusters.emplace_back(simple->geographicalId().rawId(), simple->cluster().get(), nullptr);
    } else if (simple1d) {
      clusters.emplace_back(simple1d->geographicalId().rawId(), simple1d->cluster().get(), nullptr);
    } else if (pixel) {
      clusters.emplace_back(pixel->geographicalId().rawId(), nullptr, pixel->cluster().get());
    }
    return clusters;
  }

  bool isFarFromBorder(const TrajectoryStateOnSurface& trajState, uint32_t detId, const TrackerGeometry* tGeom) {
    const auto gdu = tGeom->idToDetUnit(detId);
    if ((!dynamic_cast<const StripGeomDetUnit*>(gdu)) && (!dynamic_cast<const PixelGeomDetUnit*>(gdu))) {
      edm::LogWarning("SiStripGainCalibTableProducer")
          << "DetId " << detId << " does not seem to belong to the tracker";
      return false;
    }
    const auto plane = gdu->surface();
    const auto trapBounds = dynamic_cast<const TrapezoidalPlaneBounds*>(&plane.bounds());
    const auto rectBounds = dynamic_cast<const RectangularPlaneBounds*>(&plane.bounds());

    static constexpr double distFromBorder = 1.0;
    double halfLength = 0.;
    if (trapBounds) {
      halfLength = trapBounds->parameters()[3];
    } else if (rectBounds) {
      halfLength = .5 * gdu->surface().bounds().length();
    } else {
      return false;
    }

    const auto pos = trajState.localPosition();
    const auto posError = trajState.localError().positionError();
    if (std::abs(pos.y()) + posError.yy() >= (halfLength - distFromBorder))
      return false;

    return true;
  }
}  // namespace

//********************************************************************************//
// ------------ method called for each event  ------------
void SiStripGainsPCLWorker::dqmAnalyze(edm::Event const& iEvent,
                                       edm::EventSetup const& iSetup,
                                       APVGain::APVGainHistograms const& histograms) const {
  using namespace edm;

  unsigned int eventnumber = iEvent.id().event();
  unsigned int runnumber = iEvent.id().run();

  edm::LogInfo("SiStripGainsPCLWorker") << "Processing run " << runnumber << " and event " << eventnumber << std::endl;

  const TrackerTopology* topo = &iSetup.getData(tTopoToken_);
  const TrackerGeometry* tGeom = &iSetup.getData(tkGeomToken_);

  // Event data handles
  edm::Handle<edm::View<reco::Track>> tracks;
  iEvent.getByToken(m_tracks_token, tracks);
  edm::Handle<TrajTrackAssociationCollection> trajTrackAssociations;
  iEvent.getByToken(m_association_token, trajTrackAssociations);

  for (const auto& elem : theTopologyMap) {
    LogDebug("SiStripGainsPCLWorker") << elem.first << " - " << elem.second.m_string << " "
                                      << elem.second.m_subdetectorId << " " << elem.second.m_subdetectorSide << " "
                                      << elem.second.m_subdetectorPlane << std::endl;
  }

  LogDebug("SiStripGainsPCLWorker") << "for mode" << m_calibrationMode << std::endl;

  int elepos = statCollectionFromMode(m_calibrationMode.c_str());

  std::size_t nStoredClusters{0};
  for (const auto& assoc : *trajTrackAssociations) {
    const auto traj = assoc.key.get();
    const auto track = assoc.val.get();

    if ((track->eta() < MinTrackEta) || (track->eta() > MaxTrackEta) || (track->p() < MinTrackMomentum) ||
        (track->p() > MaxTrackMomentum) || (track->numberOfValidHits() < MinTrackHits) ||
        ((track->chi2() / track->ndof()) > MaxTrackChiOverNdf) || (track->algo() > MaxTrackingIteration))
      continue;

    int iCluster{-1};
    for (const auto& meas : traj->measurements()) {
      const auto& trajState = meas.updatedState();
      if (!trajState.isValid())
        continue;

      // there can be 2 (stereo module), 1 (no stereo module), or 0 (no pixel or strip hit) clusters
      auto clusters = getClusters(meas.recHit()->hit());
      for (const auto hitCluster : clusters) {
        ++iCluster;
        bool saturation = false;
        bool overlapping = false;
        unsigned int charge = 0;
        int firstStrip = 0;
        unsigned int nStrips = 0;
        if (hitCluster.strip) {
          const auto& ampls = hitCluster.strip->amplitudes();
          firstStrip = hitCluster.strip->firstStrip();
          nStrips = ampls.size();
          charge = hitCluster.strip->charge();
          saturation = std::any_of(ampls.begin(), ampls.end(), [](uint8_t amp) { return amp >= 254; });

          overlapping = (((firstStrip % 128) == 0) || ((firstStrip / 128) != ((firstStrip + int(nStrips)) / 128)));
        } else if (hitCluster.pixel) {
          const auto& ampls = hitCluster.pixel->pixelADC();
          const int firstRow = hitCluster.pixel->minPixelRow();
          const int firstCol = hitCluster.pixel->minPixelCol();
          firstStrip = ((firstRow / 80) << 3 | (firstCol / 52)) * 128;  //Hack to save the APVId
          nStrips = 0;
          for (const auto amp : ampls) {
            charge += amp;
            if (amp >= 254)
              saturation = true;
          }
        }
        // works for both strip and pixel thanks to firstStrip encoding for pixel above, as in the calibTree
        std::shared_ptr<stAPVGain> APV = histograms.APVsColl.at((hitCluster.det << 4) | (firstStrip / 128));

        const auto farFromEdge = (hitCluster.strip ? isFarFromBorder(trajState, hitCluster.det, tGeom) : true);
        if ((APV->SubDet > 2) &&
            ((!farFromEdge) || overlapping || (saturation && !AllowSaturation) || (nStrips > MaxNrStrips)))
          continue;

        int clusterCharge = 0;
        if (APV->SubDet > 2) {  // strip
          if (useCalibration || !FirstSetOfConstants) {
            saturation = false;
            for (const auto origCharge : hitCluster.strip->amplitudes()) {
              int stripCharge;
              if (useCalibration) {
                if (FirstSetOfConstants) {
                  stripCharge = int(origCharge / APV->CalibGain);
                } else {
                  stripCharge = int(origCharge * (APV->PreviousGain / APV->CalibGain));
                }
              } else {
                if (FirstSetOfConstants) {
                  stripCharge = origCharge;
                } else {
                  stripCharge = int(origCharge * APV->PreviousGain);
                }
              }
              if (stripCharge > 1024) {
                stripCharge = 255;
                saturation = true;
              } else if (stripCharge > 254) {
                stripCharge = 254;
                saturation = true;
              }
              clusterCharge += stripCharge;
            }
            if (saturation && !AllowSaturation)
              continue;
          } else {
            clusterCharge = charge;
          }
        } else {                           // pixel
          clusterCharge = charge / 265.0;  //expected scale factor between pixel and strip charge
        }

        const auto trackDir = trajState.localDirection();
        const auto path = (10. * APV->Thickness) / std::abs(trackDir.z() / trackDir.mag());
        double ClusterChargeOverPath = ((double)clusterCharge) / path;
        if (APV->SubDet > 2) {
          if (Validation) {
            ClusterChargeOverPath /= APV->PreviousGain;
          }
          if (OldGainRemoving) {
            ClusterChargeOverPath *= APV->PreviousGain;
          }
        } else {
          // keep processing of pixel cluster charge until here
          continue;
        }
        ++nStoredClusters;

        // real histogram for calibration
        histograms.Charge_Vs_Index[elepos]->Fill(APV->Index, ClusterChargeOverPath);
        LogDebug("SiStripGainsPCLWorker")
            << " for mode " << m_calibrationMode << "\n"
            << " i " << iCluster << " useCalibration " << useCalibration << " FirstSetOfConstants "
            << FirstSetOfConstants << " APV->PreviousGain " << APV->PreviousGain << " APV->CalibGain " << APV->CalibGain
            << " APV->DetId " << APV->DetId << " APV->Index " << APV->Index << " Charge " << clusterCharge << " Path "
            << path << " ClusterChargeOverPath " << ClusterChargeOverPath << std::endl;

        // Fill monitoring histograms
        int mCharge1 = 0;
        for (const auto sCharge : hitCluster.strip->amplitudes()) {
          if (sCharge > 254) {
            mCharge1 += 254;
          } else {
            mCharge1 += sCharge;
          }
        }
        // Revome gains for monitoring
        int mCharge2 = mCharge1 * APV->PreviousGain;                          // remove G2
        int mCharge3 = mCharge1 * APV->PreviousGainTick;                      // remove G1
        int mCharge4 = mCharge1 * APV->PreviousGain * APV->PreviousGainTick;  // remove G1 and G2

        LogDebug("SiStripGainsPCLWorker") << " full charge " << mCharge1 << " remove G2 " << mCharge2 << " remove G1 "
                                          << mCharge3 << " remove G1*G2 " << mCharge4 << std::endl;

        auto indices = APVGain::FetchIndices(theTopologyMap, hitCluster.det, topo);

        for (auto m : indices)
          histograms.Charge_1[elepos][m]->Fill(((double)mCharge1) / path);
        for (auto m : indices)
          histograms.Charge_2[elepos][m]->Fill(((double)mCharge2) / path);
        for (auto m : indices)
          histograms.Charge_3[elepos][m]->Fill(((double)mCharge3) / path);
        for (auto m : indices)
          histograms.Charge_4[elepos][m]->Fill(((double)mCharge4) / path);

        if (APV->SubDet == StripSubdetector::TIB) {
          histograms.Charge_Vs_PathlengthTIB[elepos]->Fill(path, clusterCharge);  // TIB

        } else if (APV->SubDet == StripSubdetector::TOB) {
          histograms.Charge_Vs_PathlengthTOB[elepos]->Fill(path, clusterCharge);  // TOB

        } else if (APV->SubDet == StripSubdetector::TID) {
          if (APV->Eta < 0) {
            histograms.Charge_Vs_PathlengthTIDM[elepos]->Fill(path, clusterCharge);
          }  // TID minus
          else if (APV->Eta > 0) {
            histograms.Charge_Vs_PathlengthTIDP[elepos]->Fill(path, clusterCharge);
          }  // TID plus

        } else if (APV->SubDet == StripSubdetector::TEC) {
          if (APV->Eta < 0) {
            if (APV->Thickness < 0.04) {
              histograms.Charge_Vs_PathlengthTECM1[elepos]->Fill(path, clusterCharge);
            }  // TEC minus, type 1
            else if (APV->Thickness > 0.04) {
              histograms.Charge_Vs_PathlengthTECM2[elepos]->Fill(path, clusterCharge);
            }  // TEC minus, type 2
          } else if (APV->Eta > 0) {
            if (APV->Thickness < 0.04) {
              histograms.Charge_Vs_PathlengthTECP1[elepos]->Fill(path, clusterCharge);
            }  // TEC plus, type 1
            else if (APV->Thickness > 0.04) {
              histograms.Charge_Vs_PathlengthTECP2[elepos]->Fill(path, clusterCharge);
            }  // TEC plus, type 2
          }
        }
      }
    }
  }

  histograms.EventStats->Fill(0., 0., 1);
  histograms.EventStats->Fill(1., 0., tracks->size());
  histograms.EventStats->Fill(2., 0., nStoredClusters);

  //LogDebug("SiStripGainsPCLWorker")<<" for mode"<< m_calibrationMode
  //				   <<" entries in histogram:"<< histograms.Charge_Vs_Index[elepos].getEntries()
  //				   <<std::endl;
}

//********************************************************************************//
// ------------ method called once each job just before starting event loop  ------------
void SiStripGainsPCLWorker::checkBookAPVColls(const TrackerGeometry* bareTkGeomPtr,
                                              const TrackerTopology* bareTkTopoPtr,
                                              APVGain::APVGainHistograms& histograms) const {
  if (bareTkGeomPtr) {  // pointer not yet set: called the first time => fill the APVColls
    auto const& Det = bareTkGeomPtr->dets();

    edm::LogInfo("SiStripGainsPCLWorker") << " Resetting APV struct" << std::endl;

    unsigned int Index = 0;

    for (unsigned int i = 0; i < Det.size(); i++) {
      DetId Detid = Det[i]->geographicalId();
      int SubDet = Detid.subdetId();

      if (SubDet == StripSubdetector::TIB || SubDet == StripSubdetector::TID || SubDet == StripSubdetector::TOB ||
          SubDet == StripSubdetector::TEC) {
        auto DetUnit = dynamic_cast<const StripGeomDetUnit*>(Det[i]);
        if (!DetUnit)
          continue;

        const StripTopology& Topo = DetUnit->specificTopology();
        unsigned int NAPV = Topo.nstrips() / 128;

        for (unsigned int j = 0; j < NAPV; j++) {
          auto APV = std::make_shared<stAPVGain>();
          APV->Index = Index;
          APV->Bin = -1;
          APV->DetId = Detid.rawId();
          APV->Side = 0;

          if (SubDet == StripSubdetector::TID) {
            APV->Side = bareTkTopoPtr->tidSide(Detid);
          } else if (SubDet == StripSubdetector::TEC) {
            APV->Side = bareTkTopoPtr->tecSide(Detid);
          }

          APV->APVId = j;
          APV->SubDet = SubDet;
          APV->FitMPV = -1;
          APV->FitMPVErr = -1;
          APV->FitWidth = -1;
          APV->FitWidthErr = -1;
          APV->FitChi2 = -1;
          APV->FitNorm = -1;
          APV->Gain = -1;
          APV->PreviousGain = 1;
          APV->PreviousGainTick = 1;
          APV->x = DetUnit->position().basicVector().x();
          APV->y = DetUnit->position().basicVector().y();
          APV->z = DetUnit->position().basicVector().z();
          APV->Eta = DetUnit->position().basicVector().eta();
          APV->Phi = DetUnit->position().basicVector().phi();
          APV->R = DetUnit->position().basicVector().transverse();
          APV->Thickness = DetUnit->surface().bounds().thickness();
          APV->NEntries = 0;
          APV->isMasked = false;

          histograms.APVsCollOrdered.push_back(APV);
          histograms.APVsColl[(APV->DetId << 4) | APV->APVId] = APV;
          Index++;
          histograms.NStripAPVs++;
        }  // loop on APVs
      }    // if is Strips
    }      // loop on dets

    for (unsigned int i = 0; i < Det.size();
         i++) {  //Make two loop such that the Pixel information is added at the end --> make transition simpler
      DetId Detid = Det[i]->geographicalId();
      int SubDet = Detid.subdetId();
      if (SubDet == PixelSubdetector::PixelBarrel || SubDet == PixelSubdetector::PixelEndcap) {
        auto DetUnit = dynamic_cast<const PixelGeomDetUnit*>(Det[i]);
        if (!DetUnit)
          continue;

        const PixelTopology& Topo = DetUnit->specificTopology();
        unsigned int NROCRow = Topo.nrows() / (80.);
        unsigned int NROCCol = Topo.ncolumns() / (52.);

        for (unsigned int j = 0; j < NROCRow; j++) {
          for (unsigned int i = 0; i < NROCCol; i++) {
            auto APV = std::make_shared<stAPVGain>();
            APV->Index = Index;
            APV->Bin = -1;
            APV->DetId = Detid.rawId();
            APV->Side = 0;
            APV->APVId = (j << 3 | i);
            APV->SubDet = SubDet;
            APV->FitMPV = -1;
            APV->FitMPVErr = -1;
            APV->FitWidth = -1;
            APV->FitWidthErr = -1;
            APV->FitChi2 = -1;
            APV->Gain = -1;
            APV->PreviousGain = 1;
            APV->PreviousGainTick = 1;
            APV->x = DetUnit->position().basicVector().x();
            APV->y = DetUnit->position().basicVector().y();
            APV->z = DetUnit->position().basicVector().z();
            APV->Eta = DetUnit->position().basicVector().eta();
            APV->Phi = DetUnit->position().basicVector().phi();
            APV->R = DetUnit->position().basicVector().transverse();
            APV->Thickness = DetUnit->surface().bounds().thickness();
            APV->isMasked = false;  //SiPixelQuality_->IsModuleBad(Detid.rawId());
            APV->NEntries = 0;

            histograms.APVsCollOrdered.push_back(APV);
            histograms.APVsColl[(APV->DetId << 4) | APV->APVId] = APV;
            Index++;
            histograms.NPixelDets++;

          }  // loop on ROC cols
        }    // loop on ROC rows
      }      // if Pixel
    }        // loop on Dets
  }          //if (!bareTkGeomPtr_) ...
}

//********************************************************************************//
void SiStripGainsPCLWorker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//********************************************************************************//
void SiStripGainsPCLWorker::bookHistograms(DQMStore::IBooker& ibooker,
                                           edm::Run const& run,
                                           edm::EventSetup const& setup,
                                           APVGain::APVGainHistograms& histograms) const {
  ibooker.cd();
  std::string dqm_dir = m_DQMdir;
  const char* tag = dqm_tag_[statCollectionFromMode(m_calibrationMode.c_str())].c_str();

  edm::LogInfo("SiStripGainsPCLWorker") << "Setting " << dqm_dir << " in DQM and booking histograms for tag " << tag
                                        << std::endl;

  ibooker.setCurrentFolder(dqm_dir);

  // this MonitorElement is created to log the number of events / tracks and clusters used
  // by the calibration algorithm

  histograms.EventStats = ibooker.book2I("EventStats", "Statistics", 3, -0.5, 2.5, 1, 0, 1);
  histograms.EventStats->setBinLabel(1, "events count", 1);
  histograms.EventStats->setBinLabel(2, "tracks count", 1);
  histograms.EventStats->setBinLabel(3, "clusters count", 1);

  std::string stag(tag);
  if (!stag.empty() && stag[0] != '_')
    stag.insert(0, 1, '_');

  std::string cvi = std::string("Charge_Vs_Index") + stag;
  std::string cvpTIB = std::string("Charge_Vs_PathlengthTIB") + stag;
  std::string cvpTOB = std::string("Charge_Vs_PathlengthTOB") + stag;
  std::string cvpTIDP = std::string("Charge_Vs_PathlengthTIDP") + stag;
  std::string cvpTIDM = std::string("Charge_Vs_PathlengthTIDM") + stag;
  std::string cvpTECP1 = std::string("Charge_Vs_PathlengthTECP1") + stag;
  std::string cvpTECP2 = std::string("Charge_Vs_PathlengthTECP2") + stag;
  std::string cvpTECM1 = std::string("Charge_Vs_PathlengthTECM1") + stag;
  std::string cvpTECM2 = std::string("Charge_Vs_PathlengthTECM2") + stag;

  int elepos = statCollectionFromMode(tag);

  histograms.Charge_Vs_Index.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTIB.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTOB.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTIDP.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTIDM.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTECP1.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTECP2.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTECM1.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTECM2.reserve(dqm_tag_.size());

  // The cluster charge is stored by exploiting a non uniform binning in order
  // reduce the histogram memory size. The bin width is relaxed with a falling
  // exponential function and the bin boundaries are stored in the binYarray.
  // The binXarray is used to provide as many bins as the APVs.
  //
  // More details about this implementations are here:
  // https://indico.cern.ch/event/649344/contributions/2672267/attachments/1498323/2332518/OptimizeChHisto.pdf

  std::vector<float> binXarray;
  binXarray.reserve(histograms.NStripAPVs + 1);
  for (unsigned int a = 0; a <= histograms.NStripAPVs; a++) {
    binXarray.push_back((float)a);
  }

  std::array<float, 688> binYarray;
  double p0 = 5.445;
  double p1 = 0.002113;
  double p2 = 69.01576;
  double y = 0.;
  for (int b = 0; b < 687; b++) {
    binYarray[b] = y;
    if (y <= 902.)
      y = y + 2.;
    else
      y = (p0 - log(exp(p0 - p1 * y) - p2 * p1)) / p1;
  }
  binYarray[687] = 4000.;

  histograms.Charge_1[elepos].clear();
  histograms.Charge_2[elepos].clear();
  histograms.Charge_3[elepos].clear();
  histograms.Charge_4[elepos].clear();

  auto it = histograms.Charge_Vs_Index.begin();
  histograms.Charge_Vs_Index.insert(
      it + elepos,
      ibooker.book2S(cvi.c_str(), cvi.c_str(), histograms.NStripAPVs, &binXarray[0], 687, binYarray.data()));

  it = histograms.Charge_Vs_PathlengthTIB.begin();
  histograms.Charge_Vs_PathlengthTIB.insert(it + elepos,
                                            ibooker.book2S(cvpTIB.c_str(), cvpTIB.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTOB.begin();
  histograms.Charge_Vs_PathlengthTOB.insert(it + elepos,
                                            ibooker.book2S(cvpTOB.c_str(), cvpTOB.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTIDP.begin();
  histograms.Charge_Vs_PathlengthTIDP.insert(
      it + elepos, ibooker.book2S(cvpTIDP.c_str(), cvpTIDP.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTIDM.begin();
  histograms.Charge_Vs_PathlengthTIDM.insert(
      it + elepos, ibooker.book2S(cvpTIDM.c_str(), cvpTIDM.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTECP1.begin();
  histograms.Charge_Vs_PathlengthTECP1.insert(
      it + elepos, ibooker.book2S(cvpTECP1.c_str(), cvpTECP1.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTECP2.begin();
  histograms.Charge_Vs_PathlengthTECP2.insert(
      it + elepos, ibooker.book2S(cvpTECP2.c_str(), cvpTECP2.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTECM1.begin();
  histograms.Charge_Vs_PathlengthTECM1.insert(
      it + elepos, ibooker.book2S(cvpTECM1.c_str(), cvpTECM1.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTECM2.begin();
  histograms.Charge_Vs_PathlengthTECM2.insert(
      it + elepos, ibooker.book2S(cvpTECM2.c_str(), cvpTECM2.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  std::vector<std::pair<std::string, std::string>> hnames =
      APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    std::string htag = (hnames[i]).first + stag;
    histograms.Charge_1[elepos].push_back(ibooker.book1DD(htag.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.));
  }

  hnames = APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "woG2");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    std::string htag = (hnames[i]).first + stag;
    histograms.Charge_2[elepos].push_back(ibooker.book1DD(htag.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.));
  }

  hnames = APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "woG1");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    std::string htag = (hnames[i]).first + stag;
    histograms.Charge_3[elepos].push_back(ibooker.book1DD(htag.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.));
  }

  hnames = APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "woG1G2");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    std::string htag = (hnames[i]).first + stag;
    histograms.Charge_4[elepos].push_back(ibooker.book1DD(htag.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.));
  }
}
