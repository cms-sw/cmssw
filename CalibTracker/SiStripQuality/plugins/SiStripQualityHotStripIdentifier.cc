#include "CalibTracker/SiStripQuality/plugins/SiStripQualityHotStripIdentifier.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

//Insert here the include to the algos
#include "CalibTracker/SiStripQuality/interface/SiStripHotStripAlgorithmFromClusterOccupancy.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

SiStripQualityHotStripIdentifier::SiStripQualityHotStripIdentifier(const edm::ParameterSet& iConfig)
    : ConditionDBWriter<SiStripBadStrip>(iConfig),
      dataLabel_(iConfig.getUntrackedParameter<std::string>("dataLabel", "")),
      conf_(iConfig),
      fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",
                                                         edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))),
      Cluster_src_(iConfig.getParameter<edm::InputTag>("Cluster_src")),
      Track_src_(iConfig.getUntrackedParameter<edm::InputTag>("Track_src")),
      tracksCollection_in_EventTree(iConfig.getUntrackedParameter<bool>("RemoveTrackClusters", false)),
      tTopoToken_(esConsumes<edm::Transition::BeginRun>()),
      stripQualityToken_(esConsumes<edm::Transition::BeginRun>()) {
  edm::ParameterSet pset = iConfig.getUntrackedParameter<edm::ParameterSet>("ClusterSelection", edm::ParameterSet());
  MinClusterWidth_ = pset.getUntrackedParameter<uint32_t>("minWidth", 1);
  MaxClusterWidth_ = pset.getUntrackedParameter<uint32_t>("maxWidth", 1000);

  bookHistos();
}

SiStripQualityHotStripIdentifier::~SiStripQualityHotStripIdentifier() {}

std::unique_ptr<SiStripBadStrip> SiStripQualityHotStripIdentifier::getNewObject() {
  auto obj = std::make_unique<SiStripBadStrip>();

  edm::ParameterSet parameters = conf_.getParameter<edm::ParameterSet>("AlgoParameters");
  std::string AlgoName = parameters.getParameter<std::string>("AlgoName");
  if (AlgoName == "SiStripHotStripAlgorithmFromClusterOccupancy") {
    edm::LogInfo("SiStripQualityHotStripIdentifier")
        << " [SiStripQualityHotStripIdentifier::getNewObject] call to SiStripHotStripAlgorithmFromClusterOccupancy"
        << std::endl;

    SiStripHotStripAlgorithmFromClusterOccupancy theIdentifier(conf_, tTopo);
    theIdentifier.setProbabilityThreshold(parameters.getUntrackedParameter<double>("ProbabilityThreshold", 1.E-7));
    theIdentifier.setMinNumEntries(parameters.getUntrackedParameter<uint32_t>("MinNumEntries", 100));
    theIdentifier.setMinNumEntriesPerStrip(parameters.getUntrackedParameter<uint32_t>("MinNumEntriesPerStrip", 5));

    const auto detInfo =
        SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());
    SiStripQuality* qobj = new SiStripQuality(detInfo);
    theIdentifier.extractBadStrips(qobj, ClusterPositionHistoMap, stripQuality_);

    edm::LogInfo("SiStripQualityHotStripIdentifier")
        << " [SiStripQualityHotStripIdentifier::getNewObject] copy SiStripObject in SiStripBadStrip" << std::endl;

    std::stringstream ss;

    SiStripBadStrip::RegistryIterator rIter = qobj->getRegistryVectorBegin();
    SiStripBadStrip::RegistryIterator rIterEnd = qobj->getRegistryVectorEnd();
    for (; rIter != rIterEnd; ++rIter) {
      SiStripBadStrip::Range range(qobj->getDataVectorBegin() + rIter->ibegin,
                                   qobj->getDataVectorBegin() + rIter->iend);
      if (!obj->put(rIter->detid, range))
        edm::LogError("SiStripQualityHotStripIdentifier")
            << "[SiStripQualityHotStripIdentifier::getNewObject] detid already exists" << std::endl;
    }
    edm::LogInfo("SiStripQualityHotStripIdentifier")
        << " [SiStripQualityHotStripIdentifier::getNewObject] " << ss.str() << std::endl;

  } else {
    edm::LogError("SiStripQualityHotStripIdentifier")
        << " [SiStripQualityHotStripIdentifier::getNewObject] call for a unknow HotStrip identification algoritm"
        << std::endl;

    std::vector<uint32_t> a;
    SiStripBadStrip::Range range(a.begin(), a.end());
    if (!obj->put(0xFFFFFFFF, range))
      edm::LogError("SiStripQualityHotStripIdentifier")
          << "[SiStripQualityHotStripIdentifier::getNewObject] detid already exists" << std::endl;
  }

  return obj;
}

void SiStripQualityHotStripIdentifier::algoBeginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  tTopo = &iSetup.getData(tTopoToken_);

  resetHistos();

  if (stripQualityWatcher_.check(iSetup)) {
    stripQuality_ = &iSetup.getData(stripQualityToken_);
  }
}

void SiStripQualityHotStripIdentifier::algoEndJob() {
  //Clear map
  ClusterPositionHistoMap.clear();
}

void SiStripQualityHotStripIdentifier::resetHistos() {
  edm::LogInfo("SiStripQualityHotStripIdentifier") << " [SiStripQualityHotStripIdentifier::resetHistos] " << std::endl;
  for (const auto& it : ClusterPositionHistoMap) {
    it.second->Reset();
  }
}

void SiStripQualityHotStripIdentifier::bookHistos() {
  edm::LogInfo("SiStripQualityHotStripIdentifier") << " [SiStripQualityHotStripIdentifier::bookHistos] " << std::endl;
  char hname[1024];
  for (const auto& it : SiStripDetInfoFileReader::read(fp_.fullPath()).getAllData()) {
    sprintf(hname, "h_%d", it.first);
    auto ref = ClusterPositionHistoMap.find(it.first);
    if (ref == ClusterPositionHistoMap.end()) {
      ClusterPositionHistoMap[it.first] =
          std::make_shared<TH1F>(hname, hname, it.second.nApvs * 128, -0.5, it.second.nApvs * 128 - 0.5);
    } else
      edm::LogError("SiStripQualityHotStripIdentifier")
          << " [SiStripQualityHotStripIdentifier::bookHistos] DetId " << it.first
          << " already found in map. Ignoring new data" << std::endl;
  }
}

void SiStripQualityHotStripIdentifier::fillHisto(uint32_t detid, float value) {
  auto ref = ClusterPositionHistoMap.find(detid);
  if (ref != ClusterPositionHistoMap.end())
    ref->second->Fill(value);
  else
    edm::LogError("SiStripQualityHotStripIdentifier")
        << " [SiStripQualityHotStripIdentifier::fillHisto] Histogram not found in the list for DetId " << detid
        << " Ignoring data value " << value << std::endl;
}

void SiStripQualityHotStripIdentifier::algoAnalyze(const edm::Event& e, const edm::EventSetup& eSetup) {
  edm::Handle<edm::DetSetVector<SiStripCluster> > dsv_SiStripCluster;
  e.getByLabel(Cluster_src_, dsv_SiStripCluster);

  edm::Handle<reco::TrackCollection> trackCollection;
  if (tracksCollection_in_EventTree) {
    e.getByLabel(Track_src_, trackCollection);
    if (!trackCollection.isValid()) {
      edm::LogError("SiStripQualityHotStripIdentifier")
          << " [SiStripQualityHotStripIdentifier::algoAnalyze] missing trackCollection with label " << Track_src_
          << std::endl;
    }
  }

  std::set<const void*> vPSiStripCluster;
  //Perform track study
  if (tracksCollection_in_EventTree) {
    int i = 0;
    for (const auto& track : *(trackCollection.product())) {
      LogTrace("SiStripQualityHotStripIdentifier")
          << "Track number " << i + 1 << "\n\tmomentum: " << track.momentum() << "\n\tPT: " << track.pt()
          << "\n\tvertex: " << track.vertex() << "\n\timpact parameter: " << track.d0()
          << "\n\tcharge: " << track.charge() << "\n\tnormalizedChi2: " << track.normalizedChi2() << "\n\tFrom EXTRA : "
          << "\n\t\touter PT " << track.outerPt() << std::endl;

      //Loop on rechits
      for (auto const& recHit : track.recHits()) {
        if (!recHit->isValid()) {
          LogTrace("SiStripQualityHotStripIdentifier") << "\t\t Invalid Hit " << std::endl;
          continue;
        }

        const SiStripRecHit2D* singleHit = dynamic_cast<const SiStripRecHit2D*>(recHit);
        const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D*>(recHit);
        const ProjectedSiStripRecHit2D* projectedHit = dynamic_cast<const ProjectedSiStripRecHit2D*>(recHit);

        if (matchedHit) {
          vPSiStripCluster.insert((void*)&(matchedHit->monoCluster()));
          vPSiStripCluster.insert((void*)&(matchedHit->stereoCluster()));
        } else if (projectedHit) {
          vPSiStripCluster.insert((void*)&*(projectedHit->originalHit().cluster()));
        } else if (singleHit) {
          vPSiStripCluster.insert((void*)&*(singleHit->cluster()));
        } else {
          LogTrace("SiStripQualityHotStripIdentifier") << "NULL hit" << std::endl;
        }
      }
    }
  }

  std::stringstream ss;
  //Loop on Det Clusters
  for (const auto& dSet : *dsv_SiStripCluster) {
    for (const auto& clus : dSet.data) {
      if (MinClusterWidth_ <= clus.amplitudes().size() && clus.amplitudes().size() <= MaxClusterWidth_) {
        if (std::find(vPSiStripCluster.begin(), vPSiStripCluster.end(), (void*)&clus) == vPSiStripCluster.end()) {
          if (edm::isDebugEnabled())
            ss << " adding cluster to histo for detid " << dSet.id << " with barycenter " << clus.barycenter()
               << std::endl;
          fillHisto(dSet.id, clus.barycenter());
        }
      }
    }
  }
  LogTrace("SiStripQualityHotStripIdentifier") << ss.str();
}
