// -*- C++ -*-
//
// Package:    PFTracking
// Class:      PFElecTkProducer
//
// Original Author:  Michele Pioppi
//         Created:  Tue Jan 23 15:26:39 CET 2007

/// \brief Abstract
/*!
\author Michele Pioppi, Daniele Benedetti
\date January 2007

 PFElecTkProducer reads the merged GsfTracks collection
 built with the TrackerDriven and EcalDriven seeds
 and transform them in PFGsfRecTracks.
*/

#include <memory>

#include "CommonTools/Utils/interface/KinematicTables.h"
#include "CommonTools/Utils/interface/LazyConstructed.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedTrackerVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoParticleFlow/PFClusterTools/interface/ClusterClusterMapping.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "RecoParticleFlow/PFTracking/interface/ConvBremHeavyObjectCache.h"
#include "RecoParticleFlow/PFTracking/interface/ConvBremPFTrackFinder.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackAlgoTools.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

namespace {

  constexpr float square(float x) { return x * x; };

}  // namespace

class PFElecTkProducer final : public edm::stream::EDProducer<edm::GlobalCache<convbremhelpers::HeavyObjectCache> > {
public:
  ///Constructor
  explicit PFElecTkProducer(const edm::ParameterSet&, const convbremhelpers::HeavyObjectCache*);

  static std::unique_ptr<convbremhelpers::HeavyObjectCache> initializeGlobalCache(const edm::ParameterSet& conf) {
    return std::make_unique<convbremhelpers::HeavyObjectCache>(conf);
  }

  static void globalEndJob(convbremhelpers::HeavyObjectCache const*) {}

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;

  ///Produce the PFRecTrack collection
  void produce(edm::Event&, const edm::EventSetup&) override;

  int findPfRef(const reco::PFRecTrackCollection& pfRTkColl,
                const reco::GsfTrack&,
                edm::soa::EtaPhiTableView trackEtaPhiTable);

  bool applySelection(const reco::GsfTrack&);

  bool resolveGsfTracks(const std::vector<reco::GsfPFRecTrack>& GsfPFVec,
                        unsigned int ngsf,
                        std::vector<unsigned int>& secondaries,
                        const reco::PFClusterCollection& theEClus);

  float minTangDist(const reco::GsfPFRecTrack& primGsf, const reco::GsfPFRecTrack& secGsf);

  bool isSameEgSC(const reco::ElectronSeed& nSeed,
                  const reco::ElectronSeed& iSeed,
                  bool& bothGsfEcalDriven,
                  float& SCEnergy);

  bool isSharingEcalEnergyWithEgSC(const reco::GsfPFRecTrack& nGsfPFRecTrack,
                                   const reco::GsfPFRecTrack& iGsfPFRecTrack,
                                   const reco::ElectronSeed& nSeed,
                                   const reco::ElectronSeed& iSeed,
                                   const reco::PFClusterCollection& theEClus,
                                   bool& bothGsfTrackerDriven,
                                   bool& nEcalDriven,
                                   bool& iEcalDriven,
                                   float& nEnergy,
                                   float& iEnergy);

  bool isInnerMost(const reco::GsfTrackRef& nGsfTrack, const reco::GsfTrackRef& iGsfTrack, bool& sameLayer);

  bool isInnerMostWithLostHits(const reco::GsfTrackRef& nGsfTrack, const reco::GsfTrackRef& iGsfTrack, bool& sameLayer);

  void createGsfPFRecTrackRef(const edm::OrphanHandle<reco::GsfPFRecTrackCollection>& gsfPfHandle,
                              std::vector<reco::GsfPFRecTrack>& gsfPFRecTrackPrimary,
                              const std::map<unsigned int, std::vector<reco::GsfPFRecTrack> >& MapPrimSec);

  // ----------member data ---------------------------
  reco::GsfPFRecTrack pftrack_;
  reco::GsfPFRecTrack secpftrack_;
  edm::ParameterSet conf_;
  edm::EDGetTokenT<reco::GsfTrackCollection> gsfTrackLabel_;
  edm::EDGetTokenT<reco::PFRecTrackCollection> pfTrackLabel_;
  edm::EDGetTokenT<reco::VertexCollection> primVtxLabel_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfEcalClusters_;
  edm::EDGetTokenT<reco::PFDisplacedTrackerVertexCollection> pfNuclear_;
  edm::EDGetTokenT<reco::PFConversionCollection> pfConv_;
  edm::EDGetTokenT<reco::PFV0Collection> pfV0_;
  bool useNuclear_;
  bool useConversions_;
  bool useV0_;
  bool applyAngularGsfClean_;
  double detaCutGsfClean_;
  double dphiCutGsfClean_;

  ///PFTrackTransformer
  std::unique_ptr<PFTrackTransformer> pfTransformer_;
  MultiTrajectoryStateTransform mtsTransform_;
  std::unique_ptr<ConvBremPFTrackFinder> convBremFinder_;

  ///Trajectory of GSfTracks in the event?
  bool trajinev_;
  bool modemomentum_;
  bool applySel_;
  bool applyGsfClean_;
  bool useFifthStepForEcalDriven_;
  bool useFifthStepForTrackDriven_;
  //   bool useFifthStepSec_;
  bool debugGsfCleaning_;
  double SCEne_;
  double detaGsfSC_;
  double dphiGsfSC_;
  double maxPtConvReco_;

  /// Conv Brem Finder
  bool useConvBremFinder_;

  double mvaConvBremFinderIDBarrelLowPt_;
  double mvaConvBremFinderIDBarrelHighPt_;
  double mvaConvBremFinderIDEndcapsLowPt_;
  double mvaConvBremFinderIDEndcapsHighPt_;
  std::string path_mvaWeightFileConvBremBarrelLowPt_;
  std::string path_mvaWeightFileConvBremBarrelHighPt_;
  std::string path_mvaWeightFileConvBremEndcapsLowPt_;
  std::string path_mvaWeightFileConvBremEndcapsHighPt_;

  // cache for multitrajectory states
  std::vector<double> gsfInnerMomentumCache_;
};

using namespace std;
using namespace edm;
using namespace reco;

PFElecTkProducer::PFElecTkProducer(const ParameterSet& iConfig, const convbremhelpers::HeavyObjectCache*)
    : conf_(iConfig) {
  gsfTrackLabel_ = consumes<reco::GsfTrackCollection>(iConfig.getParameter<InputTag>("GsfTrackModuleLabel"));

  pfTrackLabel_ = consumes<reco::PFRecTrackCollection>(iConfig.getParameter<InputTag>("PFRecTrackLabel"));

  primVtxLabel_ = consumes<reco::VertexCollection>(iConfig.getParameter<InputTag>("PrimaryVertexLabel"));

  pfEcalClusters_ = consumes<reco::PFClusterCollection>(iConfig.getParameter<InputTag>("PFEcalClusters"));

  pfNuclear_ = consumes<reco::PFDisplacedTrackerVertexCollection>(iConfig.getParameter<InputTag>("PFNuclear"));

  pfConv_ = consumes<reco::PFConversionCollection>(iConfig.getParameter<InputTag>("PFConversions"));

  pfV0_ = consumes<reco::PFV0Collection>(iConfig.getParameter<InputTag>("PFV0"));

  useNuclear_ = iConfig.getParameter<bool>("useNuclear");
  useConversions_ = iConfig.getParameter<bool>("useConversions");
  useV0_ = iConfig.getParameter<bool>("useV0");
  debugGsfCleaning_ = iConfig.getParameter<bool>("debugGsfCleaning");

  produces<GsfPFRecTrackCollection>();
  produces<GsfPFRecTrackCollection>("Secondary").setBranchAlias("secondary");

  trajinev_ = iConfig.getParameter<bool>("TrajInEvents");
  modemomentum_ = iConfig.getParameter<bool>("ModeMomentum");
  applySel_ = iConfig.getParameter<bool>("applyEGSelection");
  applyGsfClean_ = iConfig.getParameter<bool>("applyGsfTrackCleaning");
  applyAngularGsfClean_ = iConfig.getParameter<bool>("applyAlsoGsfAngularCleaning");
  detaCutGsfClean_ = iConfig.getParameter<double>("maxDEtaGsfAngularCleaning");
  dphiCutGsfClean_ = iConfig.getParameter<double>("maxDPhiBremTangGsfAngularCleaning");
  useFifthStepForTrackDriven_ = iConfig.getParameter<bool>("useFifthStepForTrackerDrivenGsf");
  useFifthStepForEcalDriven_ = iConfig.getParameter<bool>("useFifthStepForEcalDrivenGsf");
  maxPtConvReco_ = iConfig.getParameter<double>("MaxConvBremRecoPT");
  detaGsfSC_ = iConfig.getParameter<double>("MinDEtaGsfSC");
  dphiGsfSC_ = iConfig.getParameter<double>("MinDPhiGsfSC");
  SCEne_ = iConfig.getParameter<double>("MinSCEnergy");

  // set parameter for convBremFinder
  useConvBremFinder_ = iConfig.getParameter<bool>("useConvBremFinder");

  mvaConvBremFinderIDBarrelLowPt_ = iConfig.getParameter<double>("pf_convBremFinderID_mvaCutBarrelLowPt");
  mvaConvBremFinderIDBarrelHighPt_ = iConfig.getParameter<double>("pf_convBremFinderID_mvaCutBarrelHighPt");
  mvaConvBremFinderIDEndcapsLowPt_ = iConfig.getParameter<double>("pf_convBremFinderID_mvaCutEndcapsLowPt");
  mvaConvBremFinderIDEndcapsHighPt_ = iConfig.getParameter<double>("pf_convBremFinderID_mvaCutEndcapsHighPt");
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void PFElecTkProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  //create the empty collections
  auto gsfPFRecTrackCollection = std::make_unique<GsfPFRecTrackCollection>();

  auto gsfPFRecTrackCollectionSecondary = std::make_unique<GsfPFRecTrackCollection>();

  //read collections of tracks
  Handle<GsfTrackCollection> gsftrackscoll;
  iEvent.getByToken(gsfTrackLabel_, gsftrackscoll);

  //read collections of trajectories
  Handle<vector<Trajectory> > TrajectoryCollection;

  //read pfrectrack collection
  Handle<PFRecTrackCollection> thePfRecTrackCollection;
  iEvent.getByToken(pfTrackLabel_, thePfRecTrackCollection);

  // SoA structure for frequently used track information.
  // LazyConstructed so it is only filled when needed, i.e., when there is an electron in the event.
  auto trackEtaPhiTable = makeLazy<edm::soa::EtaPhiTable>(*thePfRecTrackCollection);

  // PFClusters
  Handle<PFClusterCollection> theECPfClustCollection;
  iEvent.getByToken(pfEcalClusters_, theECPfClustCollection);
  const PFClusterCollection& theEcalClusters = *(theECPfClustCollection.product());

  //Primary Vertexes
  Handle<reco::VertexCollection> thePrimaryVertexColl;
  iEvent.getByToken(primVtxLabel_, thePrimaryVertexColl);

  // Displaced Vertex
  Handle<reco::PFDisplacedTrackerVertexCollection> pfNuclears;
  if (useNuclear_)
    iEvent.getByToken(pfNuclear_, pfNuclears);

  // Conversions
  Handle<reco::PFConversionCollection> pfConversions;
  if (useConversions_)
    iEvent.getByToken(pfConv_, pfConversions);

  // V0
  Handle<reco::PFV0Collection> pfV0;
  if (useV0_)
    iEvent.getByToken(pfV0_, pfV0);

  GsfTrackCollection gsftracks = *(gsftrackscoll.product());
  vector<Trajectory> tjvec(0);
  if (trajinev_) {
    iEvent.getByToken(gsfTrackLabel_, TrajectoryCollection);

    tjvec = *(TrajectoryCollection.product());
  }

  vector<reco::GsfPFRecTrack> selGsfPFRecTracks;
  vector<reco::GsfPFRecTrack> primaryGsfPFRecTracks;
  std::map<unsigned int, std::vector<reco::GsfPFRecTrack> > GsfPFMap;

  for (unsigned igsf = 0; igsf < gsftracks.size(); igsf++) {
    GsfTrackRef trackRef(gsftrackscoll, igsf);
    gsfInnerMomentumCache_.push_back(trackRef->pMode());
    TrajectoryStateOnSurface i_inTSOS = mtsTransform_.innerStateOnSurface((*trackRef));
    GlobalVector i_innMom;
    if (i_inTSOS.isValid()) {
      multiTrajectoryStateMode::momentumFromModeCartesian(i_inTSOS, i_innMom);
      gsfInnerMomentumCache_.back() = i_innMom.mag();
    }
  }

  for (unsigned int igsf = 0; igsf < gsftracks.size(); igsf++) {
    GsfTrackRef trackRef(gsftrackscoll, igsf);

    int kf_ind = findPfRef(*thePfRecTrackCollection, gsftracks[igsf], trackEtaPhiTable.value());

    if (kf_ind >= 0) {
      PFRecTrackRef kf_ref(thePfRecTrackCollection, kf_ind);

      // remove fifth step tracks
      if (useFifthStepForEcalDriven_ == false || useFifthStepForTrackDriven_ == false) {
        bool isFifthStepTrack = PFTrackAlgoTools::isFifthStep(kf_ref->trackRef()->algo());
        bool isEcalDriven = true;
        bool isTrackerDriven = true;

        if (trackRef->seedRef().get() == nullptr) {
          isEcalDriven = false;
          isTrackerDriven = false;
        } else {
          auto const& SeedFromRef = static_cast<ElectronSeed const&>(*(trackRef->extra()->seedRef()));
          if (SeedFromRef.caloCluster().isNull())
            isEcalDriven = false;
          if (SeedFromRef.ctfTrack().isNull())
            isTrackerDriven = false;
        }
        //note: the same track could be both ecalDriven and trackerDriven
        if (isFifthStepTrack && isEcalDriven && isTrackerDriven == false && useFifthStepForEcalDriven_ == false) {
          continue;
        }

        if (isFifthStepTrack && isTrackerDriven && isEcalDriven == false && useFifthStepForTrackDriven_ == false) {
          continue;
        }

        if (isFifthStepTrack && isTrackerDriven && isEcalDriven && useFifthStepForTrackDriven_ == false &&
            useFifthStepForEcalDriven_ == false) {
          continue;
        }
      }

      pftrack_ = GsfPFRecTrack(gsftracks[igsf].charge(), reco::PFRecTrack::GSF, igsf, trackRef, kf_ref);
    } else {
      PFRecTrackRef dummyRef;
      pftrack_ = GsfPFRecTrack(gsftracks[igsf].charge(), reco::PFRecTrack::GSF, igsf, trackRef, dummyRef);
    }

    bool validgsfbrem = false;
    if (trajinev_) {
      validgsfbrem = pfTransformer_->addPointsAndBrems(pftrack_, gsftracks[igsf], tjvec[igsf], modemomentum_);
    } else {
      validgsfbrem = pfTransformer_->addPointsAndBrems(pftrack_, gsftracks[igsf], mtsTransform_);
    }

    bool passSel = true;
    if (applySel_)
      passSel = applySelection(gsftracks[igsf]);

    if (validgsfbrem && passSel)
      selGsfPFRecTracks.push_back(pftrack_);
  }

  unsigned int count_primary = 0;
  if (!selGsfPFRecTracks.empty()) {
    for (unsigned int ipfgsf = 0; ipfgsf < selGsfPFRecTracks.size(); ipfgsf++) {
      vector<unsigned int> secondaries(0);
      secondaries.clear();
      bool keepGsf = true;

      if (applyGsfClean_) {
        keepGsf = resolveGsfTracks(selGsfPFRecTracks, ipfgsf, secondaries, theEcalClusters);
      }

      //is primary?
      if (keepGsf == true) {
        // Find kf tracks from converted brem photons
        if (convBremFinder_->foundConvBremPFRecTrack(thePfRecTrackCollection,
                                                     thePrimaryVertexColl,
                                                     pfNuclears,
                                                     pfConversions,
                                                     pfV0,
                                                     globalCache(),
                                                     useNuclear_,
                                                     useConversions_,
                                                     useV0_,
                                                     theEcalClusters,
                                                     selGsfPFRecTracks[ipfgsf])) {
          const vector<PFRecTrackRef>& convBremPFRecTracks(convBremFinder_->getConvBremPFRecTracks());
          for (unsigned int ii = 0; ii < convBremPFRecTracks.size(); ii++) {
            selGsfPFRecTracks[ipfgsf].addConvBremPFRecTrackRef(convBremPFRecTracks[ii]);
          }
        }

        // save primaries gsf tracks
        //	gsfPFRecTrackCollection->push_back(selGsfPFRecTracks[ipfgsf]);
        primaryGsfPFRecTracks.push_back(selGsfPFRecTracks[ipfgsf]);

        // NOTE:: THE TRACKID IS USED TO LINK THE PRIMARY GSF TRACK. THIS NEEDS
        // TO BE CHANGED AS SOON AS IT IS POSSIBLE TO CHANGE DATAFORMATS
        // A MODIFICATION HERE IMPLIES A MODIFICATION IN PFBLOCKALGO.CC/H
        unsigned int primGsfIndex = selGsfPFRecTracks[ipfgsf].trackId();
        vector<reco::GsfPFRecTrack> trueGsfPFRecTracks;
        if (!secondaries.empty()) {
          // loop on secondaries gsf tracks (from converted brems)
          for (unsigned int isecpfgsf = 0; isecpfgsf < secondaries.size(); isecpfgsf++) {
            PFRecTrackRef refsecKF = selGsfPFRecTracks[(secondaries[isecpfgsf])].kfPFRecTrackRef();

            unsigned int secGsfIndex = selGsfPFRecTracks[(secondaries[isecpfgsf])].trackId();
            GsfTrackRef secGsfRef = selGsfPFRecTracks[(secondaries[isecpfgsf])].gsfTrackRef();

            if (refsecKF.isNonnull()) {
              // NOTE::IT SAVED THE TRACKID OF THE PRIMARY!!! THIS IS USED IN PFBLOCKALGO.CC/H
              secpftrack_ = GsfPFRecTrack(
                  gsftracks[secGsfIndex].charge(), reco::PFRecTrack::GSF, primGsfIndex, secGsfRef, refsecKF);
            } else {
              PFRecTrackRef dummyRef;
              // NOTE::IT SAVED THE TRACKID OF THE PRIMARY!!! THIS IS USED IN PFBLOCKALGO.CC/H
              secpftrack_ = GsfPFRecTrack(
                  gsftracks[secGsfIndex].charge(), reco::PFRecTrack::GSF, primGsfIndex, secGsfRef, dummyRef);
            }

            bool validgsfbrem = false;
            if (trajinev_) {
              validgsfbrem = pfTransformer_->addPointsAndBrems(
                  secpftrack_, gsftracks[secGsfIndex], tjvec[secGsfIndex], modemomentum_);
            } else {
              validgsfbrem = pfTransformer_->addPointsAndBrems(secpftrack_, gsftracks[secGsfIndex], mtsTransform_);
            }

            if (validgsfbrem) {
              gsfPFRecTrackCollectionSecondary->push_back(secpftrack_);
              trueGsfPFRecTracks.push_back(secpftrack_);
            }
          }
        }
        GsfPFMap.insert(pair<unsigned int, std::vector<reco::GsfPFRecTrack> >(count_primary, trueGsfPFRecTracks));
        trueGsfPFRecTracks.clear();
        count_primary++;
      }
    }
  }

  const edm::OrphanHandle<GsfPFRecTrackCollection> gsfPfRefProd =
      iEvent.put(std::move(gsfPFRecTrackCollectionSecondary), "Secondary");

  //now the secondary GsfPFRecTracks are in the event, the Ref can be created
  createGsfPFRecTrackRef(gsfPfRefProd, primaryGsfPFRecTracks, GsfPFMap);

  for (unsigned int iGSF = 0; iGSF < primaryGsfPFRecTracks.size(); iGSF++) {
    gsfPFRecTrackCollection->push_back(primaryGsfPFRecTracks[iGSF]);
  }
  iEvent.put(std::move(gsfPFRecTrackCollection));

  selGsfPFRecTracks.clear();
  GsfPFMap.clear();
  primaryGsfPFRecTracks.clear();

  std::vector<double>().swap(gsfInnerMomentumCache_);
}

// create the secondary GsfPFRecTracks Ref
void PFElecTkProducer::createGsfPFRecTrackRef(
    const edm::OrphanHandle<reco::GsfPFRecTrackCollection>& gsfPfHandle,
    std::vector<reco::GsfPFRecTrack>& gsfPFRecTrackPrimary,
    const std::map<unsigned int, std::vector<reco::GsfPFRecTrack> >& MapPrimSec) {
  unsigned int cgsf = 0;
  unsigned int csecgsf = 0;
  for (std::map<unsigned int, std::vector<reco::GsfPFRecTrack> >::const_iterator igsf = MapPrimSec.begin();
       igsf != MapPrimSec.end();
       igsf++, cgsf++) {
    vector<reco::GsfPFRecTrack> SecGsfPF = igsf->second;
    for (unsigned int iSecGsf = 0; iSecGsf < SecGsfPF.size(); iSecGsf++) {
      edm::Ref<reco::GsfPFRecTrackCollection> refgprt(gsfPfHandle, csecgsf);
      gsfPFRecTrackPrimary[cgsf].addConvBremGsfPFRecTrackRef(refgprt);
      ++csecgsf;
    }
  }

  return;
}
// ------------- method for find the corresponding kf pfrectrack ---------------------
int PFElecTkProducer::findPfRef(const reco::PFRecTrackCollection& pfRTkColl,
                                const reco::GsfTrack& gsftk,
                                edm::soa::EtaPhiTableView trackEtaPhiTable) {
  if (gsftk.seedRef().get() == nullptr) {
    return -1;
  }

  // maximum delta_r2 for matching
  constexpr float maxDR2 = square(0.05f);

  // precompute expensive trigonometry
  float gsftkEta = gsftk.eta();
  float gsftkPhi = gsftk.phi();
  auto const& gsftkHits = gsftk.seedRef()->recHits();

  auto const& electronSeedFromRef = static_cast<ElectronSeed const&>(*(gsftk.extra()->seedRef()));
  //CASE 1 ELECTRONSEED DOES NOT HAVE A REF TO THE CKFTRACK
  if (electronSeedFromRef.ctfTrack().isNull()) {
    unsigned int i_pf = 0;
    int ibest = -1;
    unsigned int ish_max = 0;
    float dr2_min = square(1000.f);

    //SEARCH THE PFRECTRACK THAT SHARES HITS WITH THE ELECTRON SEED
    // Here the cpu time has been be improved.
    for (auto const& pft : trackEtaPhiTable) {
      unsigned int ish = 0;

      using namespace edm::soa::col;
      const float dr2 = reco::deltaR2(pft.get<Eta>(), pft.get<Phi>(), gsftkEta, gsftkPhi);

      if (dr2 <= maxDR2) {
        for (auto const& hhit : pfRTkColl[i_pf].trackRef()->recHits()) {
          if (!hhit->isValid())
            continue;
          for (auto const& hit : gsftkHits) {
            if (hit.isValid() && hhit->sharesInput(&hit, TrackingRecHit::all))
              ish++;
          }
        }

        if ((ish > ish_max) || ((ish == ish_max) && (dr2 < dr2_min))) {
          ish_max = ish;
          dr2_min = dr2;
          ibest = i_pf;
        }
      }

      i_pf++;
    }

    return ((ish_max == 0) || (dr2_min > maxDR2)) ? -1 : ibest;
  } else {
    //ELECTRON SEED HAS A REFERENCE

    unsigned int i_pf = 0;

    for (auto const& pft : pfRTkColl) {
      //REF COMPARISON
      if (pft.trackRef() == electronSeedFromRef.ctfTrack()) {
        return i_pf;
      }
      i_pf++;
    }
  }
  return -1;
}

// -- method to apply gsf electron selection to EcalDriven seeds
bool PFElecTkProducer::applySelection(const reco::GsfTrack& gsftk) {
  if (gsftk.seedRef().get() == nullptr)
    return false;
  auto const& ElSeedFromRef = static_cast<ElectronSeed const&>(*(gsftk.extra()->seedRef()));

  bool passCut = false;
  if (ElSeedFromRef.ctfTrack().isNull()) {
    if (ElSeedFromRef.caloCluster().isNull())
      return passCut;
    auto const* scRef = static_cast<SuperCluster const*>(ElSeedFromRef.caloCluster().get());
    //do this just to know if exist a SC?
    if (scRef) {
      float caloEne = scRef->energy();
      float feta = fabs(scRef->eta() - gsftk.etaMode());
      float fphi = fabs(scRef->phi() - gsftk.phiMode());
      if (fphi > TMath::Pi())
        fphi -= TMath::TwoPi();
      if (caloEne > SCEne_ && feta < detaGsfSC_ && fabs(fphi) < dphiGsfSC_)
        passCut = true;
    }
  } else {
    // get all the gsf found by tracker driven
    passCut = true;
  }
  return passCut;
}
bool PFElecTkProducer::resolveGsfTracks(const vector<reco::GsfPFRecTrack>& GsfPFVec,
                                        unsigned int ngsf,
                                        vector<unsigned int>& secondaries,
                                        const reco::PFClusterCollection& theEClus) {
  bool debugCleaning = debugGsfCleaning_;
  bool n_keepGsf = true;

  const reco::GsfTrackRef& nGsfTrack = GsfPFVec[ngsf].gsfTrackRef();

  if (nGsfTrack->seedRef().get() == nullptr)
    return false;
  auto const& nElSeedFromRef = static_cast<ElectronSeed const&>(*(nGsfTrack->extra()->seedRef()));

  /* // now gotten from cache below
  TrajectoryStateOnSurface inTSOS = mtsTransform_.innerStateOnSurface((*nGsfTrack));
  GlobalVector ninnMom;
  float nPin =  nGsfTrack->pMode();
  if(inTSOS.isValid()){
    multiTrajectoryStateMode::momentumFromModeCartesian(inTSOS,ninnMom);
    nPin = ninnMom.mag();
  }
  */
  float nPin = gsfInnerMomentumCache_[nGsfTrack.key()];

  float neta = nGsfTrack->innerMomentum().eta();
  float nphi = nGsfTrack->innerMomentum().phi();

  if (debugCleaning)
    cout << " PFElecTkProducer:: considering track " << nGsfTrack->pt() << " eta,phi " << nGsfTrack->eta() << ", "
         << nGsfTrack->phi() << endl;

  for (unsigned int igsf = 0; igsf < GsfPFVec.size(); igsf++) {
    if (igsf != ngsf) {
      reco::GsfTrackRef iGsfTrack = GsfPFVec[igsf].gsfTrackRef();

      if (debugCleaning)
        cout << " PFElecTkProducer:: and  comparing with track " << iGsfTrack->pt() << " eta,phi " << iGsfTrack->eta()
             << ", " << iGsfTrack->phi() << endl;

      float ieta = iGsfTrack->innerMomentum().eta();
      float iphi = iGsfTrack->innerMomentum().phi();
      float feta = fabs(neta - ieta);
      float fphi = fabs(nphi - iphi);
      if (fphi > TMath::Pi())
        fphi -= TMath::TwoPi();

      // apply a superloose preselection only to avoid un-useful cpu time: hard-coded for this reason
      if (feta < 0.5 && fabs(fphi) < 1.0) {
        if (debugCleaning)
          cout << " Entering angular superloose preselection " << endl;

        /* //now taken from cache below
        TrajectoryStateOnSurface i_inTSOS = mtsTransform_.innerStateOnSurface((*iGsfTrack));
        GlobalVector i_innMom;
        float iPin = iGsfTrack->pMode();
        if(i_inTSOS.isValid()){
          multiTrajectoryStateMode::momentumFromModeCartesian(i_inTSOS,i_innMom);
          iPin = i_innMom.mag();
        }
        */
        float iPin = gsfInnerMomentumCache_[iGsfTrack.key()];

        if (iGsfTrack->seedRef().get() == nullptr)
          continue;
        auto const& iElSeedFromRef = static_cast<ElectronSeed const&>(*(iGsfTrack->extra()->seedRef()));

        float SCEnergy = -1.;
        // Check if two tracks match the same SC
        bool areBothGsfEcalDriven = false;
        ;
        bool isSameSC = isSameEgSC(nElSeedFromRef, iElSeedFromRef, areBothGsfEcalDriven, SCEnergy);

        // CASE1 both GsfTracks ecalDriven and match the same SC
        if (areBothGsfEcalDriven) {
          if (isSameSC) {
            float nEP = SCEnergy / nPin;
            float iEP = SCEnergy / iPin;
            if (debugCleaning)
              cout << " Entering SAME supercluster case "
                   << " nEP " << nEP << " iEP " << iEP << endl;

            // if same SC take the closest or if same
            // distance the best E/p

            // Innermost using LostHits technology
            bool isSameLayer = false;
            bool iGsfInnermostWithLostHits = isInnerMostWithLostHits(nGsfTrack, iGsfTrack, isSameLayer);

            if (debugCleaning)
              cout << " iGsf is InnerMostWithLostHits " << iGsfInnermostWithLostHits << " isSameLayer " << isSameLayer
                   << endl;

            if (iGsfInnermostWithLostHits) {
              n_keepGsf = false;
              return n_keepGsf;
            } else if (isSameLayer) {
              if (fabs(iEP - 1) < fabs(nEP - 1)) {
                n_keepGsf = false;
                return n_keepGsf;
              } else {
                secondaries.push_back(igsf);
              }
            } else {
              // save secondaries gsf track (put selection)
              secondaries.push_back(igsf);
            }
          }  // end same SC case
        } else {
          // enter in the condition where at least one track is trackerDriven
          float minBremDphi = minTangDist(GsfPFVec[ngsf], GsfPFVec[igsf]);
          float nETot = 0.;
          float iETot = 0.;
          bool isBothGsfTrackerDriven = false;
          bool nEcalDriven = false;
          bool iEcalDriven = false;
          bool isSameScEgPf = isSharingEcalEnergyWithEgSC(GsfPFVec[ngsf],
                                                          GsfPFVec[igsf],
                                                          nElSeedFromRef,
                                                          iElSeedFromRef,
                                                          theEClus,
                                                          isBothGsfTrackerDriven,
                                                          nEcalDriven,
                                                          iEcalDriven,
                                                          nETot,
                                                          iETot);

          // check if the first hit of iGsfTrack < nGsfTrack
          bool isSameLayer = false;
          bool iGsfInnermostWithLostHits = isInnerMostWithLostHits(nGsfTrack, iGsfTrack, isSameLayer);

          if (isSameScEgPf) {
            // CASE 2 : One Gsf has reference to a SC and the other one not or both not

            if (debugCleaning) {
              cout << " Sharing ECAL energy passed "
                   << " nEtot " << nETot << " iEtot " << iETot << endl;
              if (isBothGsfTrackerDriven)
                cout << " Both Track are trackerDriven " << endl;
            }

            // Innermost using LostHits technology
            if (iGsfInnermostWithLostHits) {
              n_keepGsf = false;
              return n_keepGsf;
            } else if (isSameLayer) {
              // Thirt Case:  One Gsf has reference to a SC and the other one not or both not
              // gsf tracks starts from the same layer
              // check number of sharing modules (at least 50%)
              // check number of sharing hits (at least 2)
              // check charge flip inner/outer

              // they share energy
              if (isBothGsfTrackerDriven == false) {
                // if at least one Gsf track is EcalDriven choose that one.
                if (iEcalDriven) {
                  n_keepGsf = false;
                  return n_keepGsf;
                } else {
                  secondaries.push_back(igsf);
                }
              } else {
                // if both tracks are tracker driven choose that one with the best E/p
                // with ETot = max(En,Ei)

                float ETot = -1;
                if (nETot != iETot) {
                  if (nETot > iETot)
                    ETot = nETot;
                  else
                    ETot = iETot;
                } else {
                  ETot = nETot;
                }
                float nEP = ETot / nPin;
                float iEP = ETot / iPin;

                if (debugCleaning)
                  cout << " nETot " << nETot << " iETot " << iETot << " ETot " << ETot << endl
                       << " nPin " << nPin << " iPin " << iPin << " nEP " << nEP << " iEP " << iEP << endl;

                if (fabs(iEP - 1) < fabs(nEP - 1)) {
                  n_keepGsf = false;
                  return n_keepGsf;
                } else {
                  secondaries.push_back(igsf);
                }
              }
            } else {
              secondaries.push_back(igsf);
            }
          } else if (feta < detaCutGsfClean_ && minBremDphi < dphiCutGsfClean_) {
            // very close tracks
            bool secPushedBack = false;
            if (nEcalDriven == false && nETot == 0.) {
              n_keepGsf = false;
              return n_keepGsf;
            } else if (iEcalDriven == false && iETot == 0.) {
              secondaries.push_back(igsf);
              secPushedBack = true;
            }
            if (debugCleaning)
              cout << " Close Tracks "
                   << " feta " << feta << " fabs(fphi) " << fabs(fphi) << " minBremDphi " << minBremDphi << " nETot "
                   << nETot << " iETot " << iETot << " nLostHits " << nGsfTrack->missingInnerHits() << " iLostHits "
                   << iGsfTrack->missingInnerHits() << endl;

            // apply selection only if one track has lost hits
            if (applyAngularGsfClean_) {
              if (iGsfInnermostWithLostHits) {
                n_keepGsf = false;
                return n_keepGsf;
              } else if (isSameLayer == false) {
                if (secPushedBack == false)
                  secondaries.push_back(igsf);
              }
            }
          } else if (feta < 0.1 && minBremDphi < 0.2) {
            // failed all the conditions, discard only tracker driven tracks
            // with no PFClusters linked.
            if (debugCleaning)
              cout << " Close Tracks and failed all the conditions "
                   << " feta " << feta << " fabs(fphi) " << fabs(fphi) << " minBremDphi " << minBremDphi << " nETot "
                   << nETot << " iETot " << iETot << " nLostHits " << nGsfTrack->missingInnerHits() << " iLostHits "
                   << iGsfTrack->missingInnerHits() << endl;

            if (nEcalDriven == false && nETot == 0.) {
              n_keepGsf = false;
              return n_keepGsf;
            }
            // Here I do not push back the secondary because considered fakes...
          }
        }
      }
    }
  }

  return n_keepGsf;
}
float PFElecTkProducer::minTangDist(const reco::GsfPFRecTrack& primGsf, const reco::GsfPFRecTrack& secGsf) {
  float minDphi = 1000.;

  std::vector<reco::PFBrem> primPFBrem = primGsf.PFRecBrem();
  std::vector<reco::PFBrem> secPFBrem = secGsf.PFRecBrem();

  unsigned int cbrem = 0;
  for (unsigned isbrem = 0; isbrem < secPFBrem.size(); isbrem++) {
    if (secPFBrem[isbrem].indTrajPoint() == 99)
      continue;
    const reco::PFTrajectoryPoint& atSecECAL =
        secPFBrem[isbrem].extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance);
    if (!atSecECAL.isValid())
      continue;
    float secPhi = atSecECAL.positionREP().Phi();

    unsigned int sbrem = 0;
    for (unsigned ipbrem = 0; ipbrem < primPFBrem.size(); ipbrem++) {
      if (primPFBrem[ipbrem].indTrajPoint() == 99)
        continue;
      const reco::PFTrajectoryPoint& atPrimECAL =
          primPFBrem[ipbrem].extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance);
      if (!atPrimECAL.isValid())
        continue;
      sbrem++;
      if (sbrem <= 3) {
        float primPhi = atPrimECAL.positionREP().Phi();

        float dphi = fabs(primPhi - secPhi);
        if (dphi > TMath::Pi())
          dphi -= TMath::TwoPi();
        if (fabs(dphi) < minDphi) {
          minDphi = fabs(dphi);
        }
      }
    }

    cbrem++;
    if (cbrem == 3)
      break;
  }
  return minDphi;
}
bool PFElecTkProducer::isSameEgSC(const reco::ElectronSeed& nSeed,
                                  const reco::ElectronSeed& iSeed,
                                  bool& bothGsfEcalDriven,
                                  float& SCEnergy) {
  bool isSameSC = false;

  if (nSeed.caloCluster().isNonnull() && iSeed.caloCluster().isNonnull()) {
    auto const* nscRef = static_cast<SuperCluster const*>(nSeed.caloCluster().get());
    auto const* iscRef = static_cast<SuperCluster const*>(iSeed.caloCluster().get());

    if (nscRef && iscRef) {
      bothGsfEcalDriven = true;
      if (nscRef == iscRef) {
        isSameSC = true;
        // retrieve the supercluster energy
        SCEnergy = nscRef->energy();
      }
    }
  }
  return isSameSC;
}
bool PFElecTkProducer::isSharingEcalEnergyWithEgSC(const reco::GsfPFRecTrack& nGsfPFRecTrack,
                                                   const reco::GsfPFRecTrack& iGsfPFRecTrack,
                                                   const reco::ElectronSeed& nSeed,
                                                   const reco::ElectronSeed& iSeed,
                                                   const reco::PFClusterCollection& theEClus,
                                                   bool& bothGsfTrackerDriven,
                                                   bool& nEcalDriven,
                                                   bool& iEcalDriven,
                                                   float& nEnergy,
                                                   float& iEnergy) {
  bool isSharingEnergy = false;

  //which is EcalDriven?
  bool oneEcalDriven = true;
  SuperCluster const* scRef = nullptr;
  GsfPFRecTrack gsfPfTrack;

  if (nSeed.caloCluster().isNonnull()) {
    scRef = static_cast<SuperCluster const*>(nSeed.caloCluster().get());
    assert(scRef);
    nEnergy = scRef->energy();
    nEcalDriven = true;
    gsfPfTrack = iGsfPFRecTrack;
  } else if (iSeed.caloCluster().isNonnull()) {
    scRef = static_cast<SuperCluster const*>(iSeed.caloCluster().get());
    assert(scRef);
    iEnergy = scRef->energy();
    iEcalDriven = true;
    gsfPfTrack = nGsfPFRecTrack;
  } else {
    oneEcalDriven = false;
  }

  if (oneEcalDriven) {
    //run a basic reconstruction for the particle flow

    vector<PFCluster> vecPFClusters;
    vecPFClusters.clear();

    for (PFClusterCollection::const_iterator clus = theEClus.begin(); clus != theEClus.end(); clus++) {
      PFCluster clust = *clus;
      clust.calculatePositionREP();

      float deta = fabs(scRef->position().eta() - clust.position().eta());
      float dphi = fabs(scRef->position().phi() - clust.position().phi());
      if (dphi > TMath::Pi())
        dphi -= TMath::TwoPi();

      // Angle preselection between the supercluster and pfclusters
      // this is needed just to save some cpu-time for this is hard-coded
      if (deta < 0.5 && fabs(dphi) < 1.0) {
        double distGsf = gsfPfTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALShowerMax).isValid()
                             ? LinkByRecHit::testTrackAndClusterByRecHit(gsfPfTrack, clust)
                             : -1.;
        // check if it touch the GsfTrack
        if (distGsf > 0.) {
          if (nEcalDriven)
            iEnergy += clust.energy();
          else
            nEnergy += clust.energy();
          vecPFClusters.push_back(clust);
        }
        // check if it touch the Brem-tangents
        else {
          vector<PFBrem> primPFBrem = gsfPfTrack.PFRecBrem();
          for (unsigned ipbrem = 0; ipbrem < primPFBrem.size(); ipbrem++) {
            if (primPFBrem[ipbrem].indTrajPoint() == 99)
              continue;
            const reco::PFRecTrack& pfBremTrack = primPFBrem[ipbrem];
            double dist = pfBremTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALShowerMax).isValid()
                              ? LinkByRecHit::testTrackAndClusterByRecHit(pfBremTrack, clust, true)
                              : -1.;
            if (dist > 0.) {
              if (nEcalDriven)
                iEnergy += clust.energy();
              else
                nEnergy += clust.energy();
              vecPFClusters.push_back(clust);
            }
          }
        }
      }  // END if angle preselection
    }    // PFClusters Loop
    if (!vecPFClusters.empty()) {
      for (unsigned int pf = 0; pf < vecPFClusters.size(); pf++) {
        bool isCommon = ClusterClusterMapping::overlap(vecPFClusters[pf], *scRef);
        if (isCommon) {
          isSharingEnergy = true;
        }
        break;
      }
    }
  } else {
    // both tracks are trackerDriven, try ECAL energy matching also in this case.

    bothGsfTrackerDriven = true;
    vector<PFCluster> nPFCluster;
    vector<PFCluster> iPFCluster;

    nPFCluster.clear();
    iPFCluster.clear();

    for (PFClusterCollection::const_iterator clus = theEClus.begin(); clus != theEClus.end(); clus++) {
      PFCluster clust = *clus;
      clust.calculatePositionREP();

      float ndeta = fabs(nGsfPFRecTrack.gsfTrackRef()->eta() - clust.position().eta());
      float ndphi = fabs(nGsfPFRecTrack.gsfTrackRef()->phi() - clust.position().phi());
      if (ndphi > TMath::Pi())
        ndphi -= TMath::TwoPi();
      // Apply loose preselection with the track
      // just to save cpu time, for this hard-coded
      if (ndeta < 0.5 && fabs(ndphi) < 1.0) {
        double distGsf = nGsfPFRecTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALShowerMax).isValid()
                             ? LinkByRecHit::testTrackAndClusterByRecHit(nGsfPFRecTrack, clust)
                             : -1.;
        if (distGsf > 0.) {
          nPFCluster.push_back(clust);
          nEnergy += clust.energy();
        } else {
          const vector<PFBrem>& primPFBrem = nGsfPFRecTrack.PFRecBrem();
          for (unsigned ipbrem = 0; ipbrem < primPFBrem.size(); ipbrem++) {
            if (primPFBrem[ipbrem].indTrajPoint() == 99)
              continue;
            const reco::PFRecTrack& pfBremTrack = primPFBrem[ipbrem];
            double dist = pfBremTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALShowerMax).isValid()
                              ? LinkByRecHit::testTrackAndClusterByRecHit(pfBremTrack, clust, true)
                              : -1.;
            if (dist > 0.) {
              nPFCluster.push_back(clust);
              nEnergy += clust.energy();
              break;
            }
          }
        }
      }

      float ideta = fabs(iGsfPFRecTrack.gsfTrackRef()->eta() - clust.position().eta());
      float idphi = fabs(iGsfPFRecTrack.gsfTrackRef()->phi() - clust.position().phi());
      if (idphi > TMath::Pi())
        idphi -= TMath::TwoPi();
      // Apply loose preselection with the track
      // just to save cpu time, for this hard-coded
      if (ideta < 0.5 && fabs(idphi) < 1.0) {
        double dist = iGsfPFRecTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALShowerMax).isValid()
                          ? LinkByRecHit::testTrackAndClusterByRecHit(iGsfPFRecTrack, clust)
                          : -1.;
        if (dist > 0.) {
          iPFCluster.push_back(clust);
          iEnergy += clust.energy();
        } else {
          vector<PFBrem> primPFBrem = iGsfPFRecTrack.PFRecBrem();
          for (unsigned ipbrem = 0; ipbrem < primPFBrem.size(); ipbrem++) {
            if (primPFBrem[ipbrem].indTrajPoint() == 99)
              continue;
            const reco::PFRecTrack& pfBremTrack = primPFBrem[ipbrem];
            double dist = LinkByRecHit::testTrackAndClusterByRecHit(pfBremTrack, clust, true);
            if (dist > 0.) {
              iPFCluster.push_back(clust);
              iEnergy += clust.energy();
              break;
            }
          }
        }
      }
    }

    if (!nPFCluster.empty() && !iPFCluster.empty()) {
      for (unsigned int npf = 0; npf < nPFCluster.size(); npf++) {
        for (unsigned int ipf = 0; ipf < iPFCluster.size(); ipf++) {
          bool isCommon = ClusterClusterMapping::overlap(nPFCluster[npf], iPFCluster[ipf]);
          if (isCommon) {
            isSharingEnergy = true;
            break;
          }
        }
        if (isSharingEnergy)
          break;
      }
    }
  }

  return isSharingEnergy;
}
bool PFElecTkProducer::isInnerMost(const reco::GsfTrackRef& nGsfTrack,
                                   const reco::GsfTrackRef& iGsfTrack,
                                   bool& sameLayer) {
  // copied by the class RecoEgamma/EgammaElectronAlgos/src/EgAmbiguityTools.cc
  // obsolete but the code is kept: now using lost hits method

  const reco::HitPattern& gsfHitPattern1 = nGsfTrack->hitPattern();
  const reco::HitPattern& gsfHitPattern2 = iGsfTrack->hitPattern();

  // retrieve first valid hit
  int gsfHitCounter1 = 0;
  for (auto const& hit : nGsfTrack->recHits()) {
    if (hit->isValid())
      break;
    gsfHitCounter1++;
  }

  int gsfHitCounter2 = 0;
  for (auto const& hit : iGsfTrack->recHits()) {
    if (hit->isValid())
      break;
    gsfHitCounter2++;
  }

  uint32_t gsfHit1 = gsfHitPattern1.getHitPattern(HitPattern::TRACK_HITS, gsfHitCounter1);
  uint32_t gsfHit2 = gsfHitPattern2.getHitPattern(HitPattern::TRACK_HITS, gsfHitCounter2);

  if (gsfHitPattern1.getSubStructure(gsfHit1) != gsfHitPattern2.getSubStructure(gsfHit2)) {
    return (gsfHitPattern2.getSubStructure(gsfHit2) < gsfHitPattern1.getSubStructure(gsfHit1));
  } else if (gsfHitPattern1.getLayer(gsfHit1) != gsfHitPattern2.getLayer(gsfHit2)) {
    return (gsfHitPattern2.getLayer(gsfHit2) < gsfHitPattern1.getLayer(gsfHit1));
  } else {
    sameLayer = true;
    return false;
  }
}
bool PFElecTkProducer::isInnerMostWithLostHits(const reco::GsfTrackRef& nGsfTrack,
                                               const reco::GsfTrackRef& iGsfTrack,
                                               bool& sameLayer) {
  // define closest using the lost hits on the expectedhitsineer
  unsigned int nLostHits = nGsfTrack->missingInnerHits();
  unsigned int iLostHits = iGsfTrack->missingInnerHits();

  if (nLostHits != iLostHits) {
    return (nLostHits > iLostHits);
  } else {
    sameLayer = true;
    return false;
  }
}

// ------------ method called once each job just before starting event loop  ------------
void PFElecTkProducer::beginRun(const edm::Run& run, const EventSetup& iSetup) {
  ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);

  ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);

  mtsTransform_ = MultiTrajectoryStateTransform(tracker.product(), magneticField.product());

  pfTransformer_ = std::make_unique<PFTrackTransformer>(math::XYZVector(magneticField->inTesla(GlobalPoint(0, 0, 0))));

  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
  TransientTrackBuilder thebuilder = *(builder.product());

  convBremFinder_ = std::make_unique<ConvBremPFTrackFinder>(thebuilder,
                                                            mvaConvBremFinderIDBarrelLowPt_,
                                                            mvaConvBremFinderIDBarrelHighPt_,
                                                            mvaConvBremFinderIDEndcapsLowPt_,
                                                            mvaConvBremFinderIDEndcapsHighPt_);
}

// ------------ method called once each job just after ending the event loop  ------------
void PFElecTkProducer::endRun(const edm::Run& run, const EventSetup& iSetup) {
  pfTransformer_.reset();
  convBremFinder_.reset();
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFElecTkProducer);
