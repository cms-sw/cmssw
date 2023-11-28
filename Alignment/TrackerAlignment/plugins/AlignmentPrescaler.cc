#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"
#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "Utilities/General/interface/ClassName.h"

#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"
#include "TH1F.h"
#include <string>

class TrackerTopology;

class AlignmentPrescaler : public edm::one::EDProducer<> {
public:
  AlignmentPrescaler(const edm::ParameterSet& iConfig);
  ~AlignmentPrescaler() override;
  void beginJob() override;
  void endJob() override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  // tokens
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;   //tracks in input
  edm::EDGetTokenT<AliClusterValueMap> aliClusterToken_;  //Hit-quality association map

  std::string prescfilename_;  //name of the file containing the TTree with the prescaling factors
  std::string presctreename_;  //name of the  TTree with the prescaling factors

  TFile* fpresc_;
  TTree* tpresc_;
  TRandom3* myrand_;

  int layerFromId(const DetId& id, const TrackerTopology* tTopo) const;

  unsigned int detid_;
  float hitPrescFactor_, overlapPrescFactor_;
  int totnhitspxl_;
};

using namespace std;

AlignmentPrescaler::AlignmentPrescaler(const edm::ParameterSet& iConfig)
    : tracksToken_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
      aliClusterToken_(consumes(iConfig.getParameter<edm::InputTag>("assomap"))),
      prescfilename_(iConfig.getParameter<std::string>("PrescFileName")),
      presctreename_(iConfig.getParameter<std::string>("PrescTreeName")) {
  // issue the produce<>
  produces<AliClusterValueMap>();
  produces<AliTrackTakenClusterValueMap>();
}

AlignmentPrescaler::~AlignmentPrescaler() = default;

void AlignmentPrescaler::beginJob() {
  //
  edm::LogPrint("AlignmentPrescaler") << "in AlignmentPrescaler::beginJob" << std::flush;
  fpresc_ = new TFile(prescfilename_.c_str(), "READ");
  tpresc_ = (TTree*)fpresc_->Get(presctreename_.c_str());
  tpresc_->BuildIndex("DetId");
  tpresc_->SetBranchStatus("*", false);
  tpresc_->SetBranchStatus("DetId", true);
  tpresc_->SetBranchStatus("PrescaleFactor", true);
  tpresc_->SetBranchStatus("PrescaleFactorOverlap", true);
  edm::LogPrint("AlignmentPrescaler") << " Branches activated " << std::flush;
  detid_ = 0;
  hitPrescFactor_ = 99.0;
  overlapPrescFactor_ = 88.0;

  tpresc_->SetBranchAddress("DetId", &detid_);
  tpresc_->SetBranchAddress("PrescaleFactor", &hitPrescFactor_);
  tpresc_->SetBranchAddress("PrescaleFactorOverlap", &overlapPrescFactor_);
  edm::LogPrint("AlignmentPrescaler") << " addressed " << std::flush;
  myrand_ = new TRandom3();
  //   myrand_->SetSeed();
  edm::LogPrint("AlignmentPrescaler") << " ok ";
}

void AlignmentPrescaler::endJob() {
  delete tpresc_;
  fpresc_->Close();
  delete fpresc_;
  delete myrand_;
}

void AlignmentPrescaler::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LogDebug("AlignmentPrescaler") << "\n\n#################\n### Starting the AlignmentPrescaler::produce ; Event: "
                                 << iEvent.id().run() << ", " << iEvent.id().event() << std::endl;

  edm::Handle<reco::TrackCollection> Tracks;
  iEvent.getByToken(tracksToken_, Tracks);

  //take  HitAssomap
  AliClusterValueMap InValMap = iEvent.get(aliClusterToken_);

  //prepare the output of the ValueMap flagging tracks
  std::vector<int> trackflags(Tracks->size(), 0);

  //loop on tracks
  for (std::vector<reco::Track>::const_iterator ittrk = Tracks->begin(), edtrk = Tracks->end(); ittrk != edtrk;
       ++ittrk) {
    //loop on tracking rechits
    LogDebug("AlignmentPrescaler") << "Loop on hits of track #" << (ittrk - Tracks->begin()) << std::endl;
    int ntakenhits = 0;
    bool firstTakenHit = false;

    for (auto const& hit : ittrk->recHits()) {
      if (!hit->isValid()) {
        continue;
      }
      uint32_t tmpdetid = hit->geographicalId().rawId();
      tpresc_->GetEntryWithIndex(tmpdetid);

      //-------------
      //decide whether to take this hit or not
      bool takeit = false;
      int subdetId = hit->geographicalId().subdetId();

      //check first if the cluster is also in the overlap asso map
      bool isOverlapHit = false;
      //  bool first=true;
      //ugly...
      const SiPixelRecHit* pixelhit = dynamic_cast<const SiPixelRecHit*>(hit);
      const SiStripRecHit1D* stripHit1D = dynamic_cast<const SiStripRecHit1D*>(hit);
      const SiStripRecHit2D* stripHit2D = dynamic_cast<const SiStripRecHit2D*>(hit);

      AlignmentClusterFlag tmpflag(hit->geographicalId());
      int stripType = 0;
      if (subdetId > 2) {  // SST case
        const std::type_info& type = typeid(*hit);
        if (type == typeid(SiStripRecHit1D))
          stripType = 1;
        else if (type == typeid(SiStripRecHit2D))
          stripType = 2;
        else
          stripType = 3;

        if (stripType == 1) {
          //	  const SiStripRecHit1D* stripHit1D = dynamic_cast<const SiStripRecHit1D*>(hit);

          if (stripHit1D != nullptr) {
            SiStripRecHit1D::ClusterRef stripclust(stripHit1D->cluster());
            tmpflag = InValMap[stripclust];
            tmpflag.SetDetId(hit->geographicalId());
            if (tmpflag.isOverlap())
              isOverlapHit = true;
            LogDebug("AlignmentPrescaler")
                << "~*~*~* Prescale (1D) for module " << tmpflag.detId().rawId() << "("
                << InValMap[stripclust].detId().rawId() << ") is " << hitPrescFactor_ << std::flush;
            if (tmpflag.isOverlap())
              LogDebug("AlignmentPrescaler") << " (it is Overlap)";
          }  //end if striphit1D!=0
        } else if (stripType == 2) {
          //const SiStripRecHit2D* stripHit2D = dynamic_cast<const SiStripRecHit2D*>(hit);
          if (stripHit2D != nullptr) {
            SiStripRecHit2D::ClusterRef stripclust(stripHit2D->cluster());
            tmpflag = InValMap[stripclust];
            tmpflag.SetDetId(hit->geographicalId());
            if (tmpflag.isOverlap())
              isOverlapHit = true;
            LogDebug("AlignmentPrescaler")
                << "~*~*~* Prescale (2D) for module " << tmpflag.detId().rawId() << "("
                << InValMap[stripclust].detId().rawId() << ") is " << hitPrescFactor_ << std::flush;
            if (tmpflag.isOverlap())
              LogDebug("AlignmentPrescaler") << " (it is Overlap)" << endl;
          }  //end if striphit2D!=0
        }
      }  //end if is a strip hit
      else {
        //	const SiPixelRecHit*   pixelhit= dynamic_cast<const SiPixelRecHit*>(hit);
        if (pixelhit != nullptr) {
          SiPixelClusterRefNew pixclust(pixelhit->cluster());
          tmpflag = InValMap[pixclust];
          tmpflag.SetDetId(hit->geographicalId());
          if (tmpflag.isOverlap())
            isOverlapHit = true;
        }
      }  //end else is a pixel hit
      //      tmpflag.SetDetId(hit->geographicalId());

      if (isOverlapHit) {
        LogDebug("AlignmentPrescaler") << "  DetId=" << tmpdetid << " is Overlap! " << flush;
        takeit = (float(myrand_->Rndm()) <= overlapPrescFactor_);
      }
      if (!takeit) {
        float rr = float(myrand_->Rndm());
        takeit = (rr <= hitPrescFactor_);
      }
      if (takeit) {  //HIT TAKEN !
        LogDebug("AlignmentPrescaler") << "  DetId=" << tmpdetid << " taken!" << flush;
        tmpflag.SetTakenFlag();

        if (subdetId > 2) {
          if (stripType == 1) {
            SiStripRecHit1D::ClusterRef stripclust(stripHit1D->cluster());
            InValMap[stripclust] = tmpflag;  //.SetTakenFlag();
          } else if (stripType == 2) {
            SiStripRecHit1D::ClusterRef stripclust(stripHit2D->cluster());
            InValMap[stripclust] = tmpflag;  //.SetTakenFlag();
          } else
            std::cout << "Unknown type of strip hit" << std::endl;
        } else {
          SiPixelClusterRefNew pixclust(pixelhit->cluster());
          InValMap[pixclust] = tmpflag;  //.SetTakenFlag();
        }

        if (!firstTakenHit) {
          firstTakenHit = true;
          LogDebug("AlignmentPrescaler") << "Index of the track iterator is " << ittrk - Tracks->begin();
        }
        ntakenhits++;
      }  //end if take this hit
    }    //end loop on RecHits
    trackflags[ittrk - Tracks->begin()] = ntakenhits;
  }  //end loop on tracks

  //save the asso map, tracks...
  // prepare output
  auto OutVM = std::make_unique<AliClusterValueMap>();
  *OutVM = InValMap;

  iEvent.put(std::move(OutVM));

  auto trkVM = std::make_unique<AliTrackTakenClusterValueMap>();
  AliTrackTakenClusterValueMap::Filler trkmapfiller(*trkVM);
  trkmapfiller.insert(Tracks, trackflags.begin(), trackflags.end());
  trkmapfiller.fill();
  iEvent.put(std::move(trkVM));

}  //end produce

int AlignmentPrescaler::layerFromId(const DetId& id, const TrackerTopology* tTopo) const {
  if (uint32_t(id.subdetId()) == PixelSubdetector::PixelBarrel) {
    return tTopo->pxbLayer(id);
  } else if (uint32_t(id.subdetId()) == PixelSubdetector::PixelEndcap) {
    return tTopo->pxfDisk(id) + (3 * (tTopo->pxfSide(id) - 1));
  } else if (id.subdetId() == StripSubdetector::TIB) {
    return tTopo->tibLayer(id);
  } else if (id.subdetId() == StripSubdetector::TOB) {
    return tTopo->tobLayer(id);
  } else if (id.subdetId() == StripSubdetector::TEC) {
    return tTopo->tecWheel(id) + (9 * (tTopo->pxfSide(id) - 1));
  } else if (id.subdetId() == StripSubdetector::TID) {
    return tTopo->tidWheel(id) + (3 * (tTopo->tidSide(id) - 1));
  }
  return -1;
}  //end layerfromId

void AlignmentPrescaler::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Prescale Tracker Alignment hits for alignment procedures");
  desc.add<edm::InputTag>("src", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("assomap", edm::InputTag("OverlapAssoMap"));
  desc.add<std::string>("PrescFileName", "PrescaleFactors.root");
  desc.add<std::string>("PrescTreeName", "AlignmentHitMap");
  descriptions.addWithDefaultLabel(desc);
}

// ========= MODULE DEF ==============
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlignmentPrescaler);
