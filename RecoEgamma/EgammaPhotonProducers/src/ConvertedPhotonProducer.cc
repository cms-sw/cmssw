/** \class ConvertedPhotonProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCaloClusterAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackEcalImpactPoint.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackPairFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionVertexFinder.h"
#include "RecoEgamma/EgammaTools/interface/ConversionLikelihoodCalculator.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"

#include <vector>

class ConvertedPhotonProducer : public edm::stream::EDProducer<> {
public:
  ConvertedPhotonProducer(const edm::ParameterSet& ps);

  void beginRun(edm::Run const&, const edm::EventSetup& es) final;
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  void buildCollections(
      edm::EventSetup const& es,
      const edm::Handle<edm::View<reco::CaloCluster>>& scHandle,
      const edm::Handle<edm::View<reco::CaloCluster>>& bcHandle,
      ElectronHcalHelper const& hcalHelper,
      const edm::Handle<reco::TrackCollection>& trkHandle,
      std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr, CompareTwoTracksVectors>& allPairs,
      reco::ConversionCollection& outputConvPhotonCollection);
  void cleanCollections(const edm::Handle<edm::View<reco::CaloCluster>>& scHandle,
                        const edm::OrphanHandle<reco::ConversionCollection>& conversionHandle,
                        reco::ConversionCollection& outputCollection);

  std::vector<reco::ConversionRef> solveAmbiguity(const edm::OrphanHandle<reco::ConversionCollection>& conversionHandle,
                                                  reco::CaloClusterPtr const& sc);

  float calculateMinApproachDistance(const reco::TrackRef& track1, const reco::TrackRef& track2);
  void getCircleCenter(const reco::TrackRef& tk, double r, double& x0, double& y0);

  edm::EDGetTokenT<reco::TrackCollection> conversionOITrackProducer_;
  edm::EDGetTokenT<reco::TrackCollection> conversionIOTrackProducer_;

  edm::EDGetTokenT<reco::TrackCaloClusterPtrAssociation> outInTrackSCAssociationCollection_;
  edm::EDGetTokenT<reco::TrackCaloClusterPtrAssociation> inOutTrackSCAssociationCollection_;

  edm::EDGetTokenT<reco::TrackCollection> generalTrackProducer_;

  // Register the product
  edm::EDPutTokenT<reco::ConversionCollection> convertedPhotonCollectionPutToken_;
  edm::EDPutTokenT<reco::ConversionCollection> cleanedConvertedPhotonCollectionPutToken_;

  edm::EDGetTokenT<edm::View<reco::CaloCluster>> bcBarrelCollection_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster>> bcEndcapCollection_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster>> scHybridBarrelProducer_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster>> scIslandEndcapProducer_;
  edm::EDGetTokenT<HBHERecHitCollection> hbheRecHits_;

  MagneticField const* magneticField_;
  TransientTrackBuilder const* transientTrackBuilder_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mFToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackToken_;

  ConversionTrackPairFinder trackPairFinder_;
  ConversionVertexFinder vertexFinder_;
  std::string algoName_;

  double hOverEConeSize_;
  double maxHOverE_;
  double minSCEt_;
  bool recoverOneTrackCase_;
  double dRForConversionRecovery_;
  double deltaCotCut_;
  double minApproachDisCut_;
  int maxNumOfCandidates_;
  bool risolveAmbiguity_;

  std::unique_ptr<ElectronHcalHelper> hcalHelper_;

  ConversionLikelihoodCalculator likelihoodCalc_;
  std::string likelihoodWeights_;

  math::XYZPointF toFConverterP(const math::XYZPoint& val) { return math::XYZPointF(val.x(), val.y(), val.z()); }

  math::XYZVectorF toFConverterV(const math::XYZVector& val) { return math::XYZVectorF(val.x(), val.y(), val.z()); }
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ConvertedPhotonProducer);

ConvertedPhotonProducer::ConvertedPhotonProducer(const edm::ParameterSet& config)
    : conversionOITrackProducer_{consumes(config.getParameter<std::string>("conversionOITrackProducer"))},
      conversionIOTrackProducer_{consumes(config.getParameter<std::string>("conversionIOTrackProducer"))},
      outInTrackSCAssociationCollection_{consumes({config.getParameter<std::string>("conversionOITrackProducer"),
                                                   config.getParameter<std::string>("outInTrackSCAssociation")})},
      inOutTrackSCAssociationCollection_{consumes({config.getParameter<std::string>("conversionIOTrackProducer"),
                                                   config.getParameter<std::string>("inOutTrackSCAssociation")})},

      generalTrackProducer_{consumes(config.getParameter<edm::InputTag>("generalTracksSrc"))},
      convertedPhotonCollectionPutToken_{
          produces<reco::ConversionCollection>(config.getParameter<std::string>("convertedPhotonCollection"))},
      cleanedConvertedPhotonCollectionPutToken_{
          produces<reco::ConversionCollection>(config.getParameter<std::string>("cleanedConvertedPhotonCollection"))},

      bcBarrelCollection_{consumes(config.getParameter<edm::InputTag>("bcBarrelCollection"))},
      bcEndcapCollection_{consumes(config.getParameter<edm::InputTag>("bcEndcapCollection"))},
      scHybridBarrelProducer_{consumes(config.getParameter<edm::InputTag>("scHybridBarrelProducer"))},
      scIslandEndcapProducer_{consumes(config.getParameter<edm::InputTag>("scIslandEndcapProducer"))},
      hbheRecHits_{consumes(config.getParameter<edm::InputTag>("hbheRecHits"))},
      caloGeomToken_{esConsumes()},
      mFToken_{esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()},
      transientTrackToken_{esConsumes<TransientTrackBuilder, TransientTrackRecord, edm::Transition::BeginRun>(
          edm::ESInputTag("", "TransientTrackBuilder"))},
      vertexFinder_{config},
      algoName_{config.getParameter<std::string>("AlgorithmName")},

      hOverEConeSize_{config.getParameter<double>("hOverEConeSize")},
      maxHOverE_{config.getParameter<double>("maxHOverE")},
      minSCEt_{config.getParameter<double>("minSCEt")},
      recoverOneTrackCase_{config.getParameter<bool>("recoverOneTrackCase")},
      dRForConversionRecovery_{config.getParameter<double>("dRForConversionRecovery")},
      deltaCotCut_{config.getParameter<double>("deltaCotCut")},
      minApproachDisCut_{config.getParameter<double>("minApproachDisCut")},

      maxNumOfCandidates_{config.getParameter<int>("maxNumOfCandidates")},
      risolveAmbiguity_{config.getParameter<bool>("risolveConversionAmbiguity")},
      likelihoodWeights_{config.getParameter<std::string>("MVA_weights_location")} {
  // instantiate the Track Pair Finder algorithm
  likelihoodCalc_.setWeightsFile(edm::FileInPath{likelihoodWeights_.c_str()}.fullPath().c_str());

  ElectronHcalHelper::Configuration cfgCone;
  cfgCone.hOverEConeSize = hOverEConeSize_;
  if (cfgCone.hOverEConeSize > 0) {
    cfgCone.onlyBehindCluster = false;
    cfgCone.checkHcalStatus = false;

    cfgCone.hbheRecHits = hbheRecHits_;

    cfgCone.eThresHB = config.getParameter<EgammaHcalIsolation::arrayHB>("recHitEThresholdHB");
    cfgCone.maxSeverityHB = config.getParameter<int>("maxHcalRecHitSeverity");
    cfgCone.eThresHE = config.getParameter<EgammaHcalIsolation::arrayHE>("recHitEThresholdHE");
    cfgCone.maxSeverityHE = cfgCone.maxSeverityHB;
  }

  hcalHelper_ = std::make_unique<ElectronHcalHelper>(cfgCone, consumesCollector());
}

void ConvertedPhotonProducer::beginRun(edm::Run const& r, edm::EventSetup const& theEventSetup) {
  magneticField_ = &theEventSetup.getData(mFToken_);

  // Transform Track into TransientTrack (needed by the Vertex fitter)
  transientTrackBuilder_ = &theEventSetup.getData(transientTrackToken_);
}

void ConvertedPhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  //
  // create empty output collections
  //
  // Converted photon candidates
  reco::ConversionCollection outputConvPhotonCollection;
  // Converted photon candidates
  reco::ConversionCollection cleanedConversionCollection;

  // Get the Super Cluster collection in the Barrel
  bool validBarrelSCHandle = true;
  auto scBarrelHandle = theEvent.getHandle(scHybridBarrelProducer_);
  if (!scBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the scHybridBarrelProducer";
    validBarrelSCHandle = false;
  }

  // Get the Super Cluster collection in the Endcap
  bool validEndcapSCHandle = true;
  edm::Handle<edm::View<reco::CaloCluster>> scEndcapHandle;
  theEvent.getByToken(scIslandEndcapProducer_, scEndcapHandle);
  if (!scEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the scIslandEndcapProducer";
    validEndcapSCHandle = false;
  }

  //// Get the Out In CKF tracks from conversions
  bool validTrackInputs = true;
  auto outInTrkHandle = theEvent.getHandle(conversionOITrackProducer_);
  if (!outInTrkHandle.isValid()) {
    //std::cout << "Error! Can't get the conversionOITrack " << "\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the conversionOITrack "
                                             << "\n";
    validTrackInputs = false;
  }
  //  LogDebug("ConvertedPhotonProducer")<< "ConvertedPhotonProducer  outInTrack collection size " << (*outInTrkHandle).size() << "\n";

  //// Get the association map between CKF Out In tracks and the SC where they originated
  auto outInTrkSCAssocHandle = theEvent.getHandle(outInTrackSCAssociationCollection_);
  if (!outInTrkSCAssocHandle.isValid()) {
    //  std::cout << "Error! Can't get the product " <<  outInTrackSCAssociationCollection_.c_str() <<"\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the outInTrackSCAssociationCollection)";
    validTrackInputs = false;
  }

  //// Get the In Out  CKF tracks from conversions
  auto inOutTrkHandle = theEvent.getHandle(conversionIOTrackProducer_);
  if (!inOutTrkHandle.isValid()) {
    // std::cout << "Error! Can't get the conversionIOTrack " << "\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the conversionIOTrack "
                                             << "\n";
    validTrackInputs = false;
  }
  //  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer inOutTrack collection size " << (*inOutTrkHandle).size() << "\n";

  //// Get the generalTracks if the recovery of one track cases is switched on

  edm::Handle<reco::TrackCollection> generalTrkHandle;
  if (recoverOneTrackCase_) {
    theEvent.getByToken(generalTrackProducer_, generalTrkHandle);
    if (!generalTrkHandle.isValid()) {
      //std::cout << "Error! Can't get the genralTracks " << "\n";
      edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the genralTracks "
                                               << "\n";
    }
  }

  //// Get the association map between CKF in out tracks and the SC  where they originated
  auto inOutTrkSCAssocHandle = theEvent.getHandle(inOutTrackSCAssociationCollection_);
  if (!inOutTrkSCAssocHandle.isValid()) {
    //std::cout << "Error! Can't get the product " <<  inOutTrackSCAssociationCollection_.c_str() <<"\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the inOutTrackSCAssociationCollection_.c_str()";
    validTrackInputs = false;
  }

  // Get the basic cluster collection in the Barrel
  edm::Handle<edm::View<reco::CaloCluster>> bcBarrelHandle;
  theEvent.getByToken(bcBarrelCollection_, bcBarrelHandle);
  if (!bcBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the bcBarrelCollection";
  }

  // Get the basic cluster collection in the Endcap
  edm::Handle<edm::View<reco::CaloCluster>> bcEndcapHandle;
  theEvent.getByToken(bcEndcapCollection_, bcEndcapHandle);
  if (!bcEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the bcEndcapCollection";
  }

  if (validTrackInputs) {
    //do the conversion:
    std::vector<reco::TransientTrack> t_outInTrk = transientTrackBuilder_->build(outInTrkHandle);
    std::vector<reco::TransientTrack> t_inOutTrk = transientTrackBuilder_->build(inOutTrkHandle);

    ///// Find the +/- pairs
    std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr, CompareTwoTracksVectors> allPairs;
    allPairs = trackPairFinder_.run(
        t_outInTrk, outInTrkHandle, outInTrkSCAssocHandle, t_inOutTrk, inOutTrkHandle, inOutTrkSCAssocHandle);
    //LogDebug("ConvertedPhotonProducer")  << "ConvertedPhotonProducer  allPairs.size " << allPairs.size() << "\n";

    hcalHelper_->beginEvent(theEvent, theEventSetup);

    buildCollections(theEventSetup,
                     scBarrelHandle,
                     bcBarrelHandle,
                     *hcalHelper_,
                     generalTrkHandle,
                     allPairs,
                     outputConvPhotonCollection);
    buildCollections(theEventSetup,
                     scEndcapHandle,
                     bcEndcapHandle,
                     *hcalHelper_,
                     generalTrkHandle,
                     allPairs,
                     outputConvPhotonCollection);
  }

  // put the product in the event
  auto const conversionHandle =
      theEvent.emplace(convertedPhotonCollectionPutToken_, std::move(outputConvPhotonCollection));

  // Loop over barrel and endcap SC collections and fill the  photon collection
  if (validBarrelSCHandle)
    cleanCollections(scBarrelHandle, conversionHandle, cleanedConversionCollection);
  if (validEndcapSCHandle)
    cleanCollections(scEndcapHandle, conversionHandle, cleanedConversionCollection);

  theEvent.emplace(cleanedConvertedPhotonCollectionPutToken_, std::move(cleanedConversionCollection));
}

void ConvertedPhotonProducer::buildCollections(
    edm::EventSetup const& es,
    const edm::Handle<edm::View<reco::CaloCluster>>& scHandle,
    const edm::Handle<edm::View<reco::CaloCluster>>& bcHandle,
    ElectronHcalHelper const& hcalHelper,
    const edm::Handle<reco::TrackCollection>& generalTrkHandle,
    std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr, CompareTwoTracksVectors>& allPairs,
    reco::ConversionCollection& outputConvPhotonCollection)

{
  // instantiate the algorithm for finding the position of the track extrapolation at the Ecal front face
  ConversionTrackEcalImpactPoint theEcalImpactPositionFinder(magneticField_);

  reco::Conversion::ConversionAlgorithm algo = reco::Conversion::algoByName(algoName_);

  std::vector<reco::TransientTrack> t_generalTrk;
  if (recoverOneTrackCase_)
    t_generalTrk = transientTrackBuilder_->build(generalTrkHandle);

  //  Loop over SC in the barrel and reconstruct converted photons
  reco::CaloClusterPtrVector scPtrVec;
  for (auto const& aClus : scHandle->ptrs()) {
    // preselection based in Et and H/E cut
    if (aClus->energy() / cosh(aClus->eta()) <= minSCEt_)
      continue;
    const reco::CaloCluster* pClus = &(*aClus);
    auto const* sc = dynamic_cast<const reco::SuperCluster*>(pClus);
    double HoE = hcalHelper.hcalESum(*sc, 0) / sc->energy();
    if (HoE >= maxHOverE_)
      continue;
    /////

    std::vector<edm::Ref<reco::TrackCollection>> trackPairRef;
    std::vector<math::XYZPointF> trackInnPos;
    std::vector<math::XYZVectorF> trackPin;
    std::vector<math::XYZVectorF> trackPout;
    float minAppDist = -99;

    //LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer SC energy " << aClus->energy() << " eta " <<  aClus->eta() << " phi " <<  aClus->phi() << "\n";

    //// Set here first quantities for the converted photon
    const reco::Particle::Point vtx(0, 0, 0);

    math::XYZVector direction = aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy());

    if (!allPairs.empty()) {
      for (auto iPair = allPairs.begin(); iPair != allPairs.end(); ++iPair) {
        scPtrVec.clear();

        reco::Vertex theConversionVertex;
        reco::CaloClusterPtr caloPtr = iPair->second;
        if (!(aClus == caloPtr))
          continue;

        scPtrVec.push_back(aClus);

        std::vector<math::XYZPointF> trkPositionAtEcal = theEcalImpactPositionFinder.find(iPair->first, bcHandle);
        std::vector<reco::CaloClusterPtr> matchingBC = theEcalImpactPositionFinder.matchingBC();

        minAppDist = -99;
        const std::string metname = "ConvertedPhotons|ConvertedPhotonProducer";
        if ((iPair->first).size() > 1) {
          try {
            vertexFinder_.run(iPair->first, theConversionVertex);

          } catch (cms::Exception& e) {
            //std::cout << " cms::Exception caught in ConvertedPhotonProducer::produce" << "\n" ;
            edm::LogWarning(metname) << "cms::Exception caught in ConvertedPhotonProducer::produce\n"
                                     << e.explainSelf();
          }

          // Old TwoTrackMinimumDistance md;
          // Old md.calculate  (  (iPair->first)[0].initialFreeState(),  (iPair->first)[1].initialFreeState() );
          // Old minAppDist = md.distance();

          /*
	for ( unsigned int i=0; i< matchingBC.size(); ++i) {
          if (  matchingBC[i].isNull() )  std::cout << " This ref to BC is null: skipping " <<  "\n";
          else 
	    std::cout << " BC energy " << matchingBC[i]->energy() <<  "\n";
	}
	*/

          //// loop over tracks in the pair  for creating a reference
          trackPairRef.clear();
          trackInnPos.clear();
          trackPin.clear();
          trackPout.clear();

          for (std::vector<reco::TransientTrack>::const_iterator iTk = (iPair->first).begin();
               iTk != (iPair->first).end();
               ++iTk) {
            //LogDebug("ConvertedPhotonProducer")  << "  ConvertedPhotonProducer Transient Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";

            auto const* ttt = dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
            reco::TrackRef myTkRef = ttt->persistentTrackRef();

            //LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer Ref to Rec Tracks in the pair  charge " << myTkRef->charge() << " Num of RecHits " << myTkRef->recHitsSize() << " inner momentum " << myTkRef->innerMomentum() << "\n";
            if (myTkRef->extra().isNonnull()) {
              trackInnPos.push_back(toFConverterP(myTkRef->innerPosition()));
              trackPin.push_back(toFConverterV(myTkRef->innerMomentum()));
              trackPout.push_back(toFConverterV(myTkRef->outerMomentum()));
            }
            trackPairRef.push_back(myTkRef);
          }

          //	std::cout << " ConvertedPhotonProducer trackPin size " << trackPin.size() << std::endl;
          //LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer SC energy " <<  aClus->energy() << "\n";
          //LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer photon p4 " << p4  << "\n";
          //LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer vtx " << vtx.x() << " " << vtx.y() << " " << vtx.z() << "\n";
          if (theConversionVertex.isValid()) {
            //LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer theConversionVertex " << theConversionVertex.position().x() << " " << theConversionVertex.position().y() << " " << theConversionVertex.position().z() << "\n";
          }
          //LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer trackPairRef  " << trackPairRef.size() <<  "\n";

          minAppDist = calculateMinApproachDistance(trackPairRef[0], trackPairRef[1]);

          double like = -999.;
          reco::Conversion newCandidate(scPtrVec,
                                        trackPairRef,
                                        trkPositionAtEcal,
                                        theConversionVertex,
                                        matchingBC,
                                        minAppDist,
                                        trackInnPos,
                                        trackPin,
                                        trackPout,
                                        like,
                                        algo);
          like = likelihoodCalc_.calculateLikelihood(newCandidate);
          //    std::cout << "like = " << like << std::endl;
          newCandidate.setMVAout(like);
          outputConvPhotonCollection.push_back(newCandidate);

          //LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Barrel " << "\n";

        } else {
          //	  std::cout << "   ConvertedPhotonProducer case with only one track found " <<  "\n";

          //std::cout << "   ConvertedPhotonProducer recovering one track " <<  "\n";
          trackPairRef.clear();
          trackInnPos.clear();
          trackPin.clear();
          trackPout.clear();
          std::vector<reco::TransientTrack>::const_iterator iTk = (iPair->first).begin();
          //std::cout  << "  ConvertedPhotonProducer Transient Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << " pt " << sqrt(iTk->track().innerMomentum().perp2()) << "\n";
          auto const* ttt = dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
          reco::TrackRef myTk = ttt->persistentTrackRef();
          if (myTk->extra().isNonnull()) {
            trackInnPos.push_back(toFConverterP(myTk->innerPosition()));
            trackPin.push_back(toFConverterV(myTk->innerMomentum()));
            trackPout.push_back(toFConverterV(myTk->outerMomentum()));
          }
          trackPairRef.push_back(myTk);
          //std::cout << " Provenance " << myTk->algoName() << std::endl;

          if (recoverOneTrackCase_) {
            float theta1 = myTk->innerMomentum().Theta();
            float dCot = 999.;
            float dCotTheta = -999.;
            reco::TrackRef goodRef;
            for (auto const& tran : t_generalTrk) {
              auto const* ttt = dynamic_cast<const reco::TrackTransientTrack*>(tran.basicTransientTrack());
              reco::TrackRef trRef = ttt->persistentTrackRef();
              if (trRef->charge() * myTk->charge() > 0)
                continue;
              float dEta = trRef->eta() - myTk->eta();
              float dPhi = trRef->phi() - myTk->phi();
              if (sqrt(dEta * dEta + dPhi * dPhi) > dRForConversionRecovery_)
                continue;
              float theta2 = trRef->innerMomentum().Theta();
              dCotTheta = 1. / tan(theta1) - 1. / tan(theta2);
              //	std::cout << "  ConvertedPhotonProducer recovering general transient track charge " << trRef->charge() << " momentum " << trRef->innerMomentum() << " dcotTheta " << fabs(dCotTheta) << std::endl;
              if (fabs(dCotTheta) < dCot) {
                dCot = fabs(dCotTheta);
                goodRef = trRef;
              }
            }

            if (goodRef.isNonnull()) {
              minAppDist = calculateMinApproachDistance(myTk, goodRef);

              // std::cout << "  ConvertedPhotonProducer chosen dCotTheta " <<  fabs(dCotTheta) << std::endl;
              if (fabs(dCotTheta) < deltaCotCut_ && minAppDist > minApproachDisCut_) {
                trackInnPos.push_back(toFConverterP(goodRef->innerPosition()));
                trackPin.push_back(toFConverterV(goodRef->innerMomentum()));
                trackPout.push_back(toFConverterV(goodRef->outerMomentum()));
                trackPairRef.push_back(goodRef);
                //	    std::cout << " ConvertedPhotonProducer adding opposite charge track from generalTrackCollection charge " <<  goodRef ->charge() << " pt " << sqrt(goodRef->innerMomentum().perp2())  << " trackPairRef size " << trackPairRef.size() << std::endl;
                //std::cout << " Track Provenenance " << goodRef->algoName() << std::endl;

                try {
                  vertexFinder_.run(iPair->first, theConversionVertex);

                } catch (cms::Exception& e) {
                  //std::cout << " cms::Exception caught in ConvertedPhotonProducer::produce" << "\n" ;
                  edm::LogWarning(metname) << "cms::Exception caught in ConvertedPhotonProducer::produce\n"
                                           << e.explainSelf();
                }
              }
            }

          }  // bool On/Off one track case recovery using generalTracks
          const double like = -999.;
          outputConvPhotonCollection.emplace_back(scPtrVec,
                                                  trackPairRef,
                                                  trkPositionAtEcal,
                                                  theConversionVertex,
                                                  matchingBC,
                                                  minAppDist,
                                                  trackInnPos,
                                                  trackPin,
                                                  trackPout,
                                                  like,
                                                  algo);
          auto& newCandidate = outputConvPhotonCollection.back();
          newCandidate.setMVAout(likelihoodCalc_.calculateLikelihood(newCandidate));

        }  // case with only on track: looking in general tracks
      }
    }
  }
}

void ConvertedPhotonProducer::cleanCollections(const edm::Handle<edm::View<reco::CaloCluster>>& scHandle,
                                               const edm::OrphanHandle<reco::ConversionCollection>& conversionHandle,
                                               reco::ConversionCollection& outputConversionCollection) {
  reco::Conversion* newCandidate = nullptr;
  for (auto const& aClus : scHandle->ptrs()) {
    // SC energy preselection
    if (aClus->energy() / cosh(aClus->eta()) <= minSCEt_)
      continue;

    if (conversionHandle.isValid()) {
      if (risolveAmbiguity_) {
        std::vector<reco::ConversionRef> bestRef = solveAmbiguity(conversionHandle, aClus);

        for (std::vector<reco::ConversionRef>::iterator iRef = bestRef.begin(); iRef != bestRef.end(); iRef++) {
          if (iRef->isNonnull()) {
            newCandidate = (*iRef)->clone();
            outputConversionCollection.push_back(*newCandidate);
            delete newCandidate;
          }
        }

      } else {
        for (unsigned int icp = 0; icp < conversionHandle->size(); icp++) {
          reco::ConversionRef cpRef(reco::ConversionRef(conversionHandle, icp));
          if (!(aClus.id() == cpRef->caloCluster()[0].id() && aClus.key() == cpRef->caloCluster()[0].key()))
            continue;
          if (!cpRef->isConverted())
            continue;
          if (cpRef->nTracks() < 2)
            continue;
          newCandidate = (&(*cpRef))->clone();
          outputConversionCollection.push_back(*newCandidate);
          delete newCandidate;
        }

      }  // solve or not the ambiguity of many conversion candidates
    }
  }
}

std::vector<reco::ConversionRef> ConvertedPhotonProducer::solveAmbiguity(
    const edm::OrphanHandle<reco::ConversionCollection>& conversionHandle, reco::CaloClusterPtr const& scRef) {
  std::multimap<double, reco::ConversionRef, std::greater<double>> convMap;

  for (unsigned int icp = 0; icp < conversionHandle->size(); icp++) {
    reco::ConversionRef cpRef{conversionHandle, icp};

    //std::cout << " cpRef " << cpRef->nTracks() << " " <<  cpRef ->caloCluster()[0]->energy() << std::endl;
    if (!(scRef.id() == cpRef->caloCluster()[0].id() && scRef.key() == cpRef->caloCluster()[0].key()))
      continue;
    if (!cpRef->isConverted())
      continue;
    double like = cpRef->MVAout();
    if (cpRef->nTracks() < 2)
      continue;
    //    std::cout << " Like " << like << std::endl;
    convMap.emplace(like, cpRef);
  }

  //  std::cout << " convMap size " << convMap.size() << std::endl;

  std::vector<reco::ConversionRef> bestRefs;
  for (auto iMap = convMap.begin(); iMap != convMap.end(); iMap++) {
    //    std::cout << " Like list in the map " <<  iMap->first << " " << (iMap->second)->EoverP() << std::endl;
    bestRefs.push_back(iMap->second);
    if (int(bestRefs.size()) == maxNumOfCandidates_)
      break;
  }

  return bestRefs;
}

float ConvertedPhotonProducer::calculateMinApproachDistance(const reco::TrackRef& track1,
                                                            const reco::TrackRef& track2) {
  double x1, x2, y1, y2;
  double xx_1 = track1->innerPosition().x(), yy_1 = track1->innerPosition().y(), zz_1 = track1->innerPosition().z();
  double xx_2 = track2->innerPosition().x(), yy_2 = track2->innerPosition().y(), zz_2 = track2->innerPosition().z();
  double radius1 =
      track1->innerMomentum().Rho() / (.3 * (magneticField_->inTesla(GlobalPoint(xx_1, yy_1, zz_1)).z())) * 100;
  double radius2 =
      track2->innerMomentum().Rho() / (.3 * (magneticField_->inTesla(GlobalPoint(xx_2, yy_2, zz_2)).z())) * 100;
  getCircleCenter(track1, radius1, x1, y1);
  getCircleCenter(track2, radius2, x2, y2);

  return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) - radius1 - radius2;
}

void ConvertedPhotonProducer::getCircleCenter(const reco::TrackRef& tk, double r, double& x0, double& y0) {
  double x1, y1, phi;
  x1 = tk->innerPosition().x();  //inner position and inner momentum need track Extra!
  y1 = tk->innerPosition().y();
  phi = tk->innerMomentum().phi();
  const int charge = tk->charge();
  x0 = x1 + r * sin(phi) * charge;
  y0 = y1 - r * cos(phi) * charge;
}
