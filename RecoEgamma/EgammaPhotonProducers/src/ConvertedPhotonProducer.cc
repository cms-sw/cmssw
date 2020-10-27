#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

//

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
//
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackEcalImpactPoint.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackPairFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionVertexFinder.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/ConvertedPhotonProducer.h"
//
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
//
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

ConvertedPhotonProducer::ConvertedPhotonProducer(const edm::ParameterSet& config)
    : conf_(config), theTrackPairFinder_(nullptr), theVertexFinder_(nullptr), theLikelihoodCalc_(nullptr) {
  //cout<< " ConvertedPhotonProducer CTOR " << "\n";

  // use onfiguration file to setup input collection names
  bcBarrelCollection_ =
      consumes<edm::View<reco::CaloCluster> >(conf_.getParameter<edm::InputTag>("bcBarrelCollection"));
  bcEndcapCollection_ =
      consumes<edm::View<reco::CaloCluster> >(conf_.getParameter<edm::InputTag>("bcEndcapCollection"));

  scHybridBarrelProducer_ =
      consumes<edm::View<reco::CaloCluster> >(conf_.getParameter<edm::InputTag>("scHybridBarrelProducer"));
  scIslandEndcapProducer_ =
      consumes<edm::View<reco::CaloCluster> >(conf_.getParameter<edm::InputTag>("scIslandEndcapProducer"));

  std::string oitrackprod = conf_.getParameter<std::string>("conversionOITrackProducer");
  std::string iotrackprod = conf_.getParameter<std::string>("conversionIOTrackProducer");

  std::string oitrackassoc = conf_.getParameter<std::string>("outInTrackSCAssociation");
  std::string iotrackassoc = conf_.getParameter<std::string>("inOutTrackSCAssociation");

  edm::InputTag oitracks(oitrackprod), oitracksassoc(oitrackprod, oitrackassoc), iotracks(iotrackprod),
      iotracksassoc(iotrackprod, iotrackassoc);

  conversionOITrackProducer_ = consumes<reco::TrackCollection>(oitracks);
  outInTrackSCAssociationCollection_ = consumes<reco::TrackCaloClusterPtrAssociation>(oitracksassoc);
  conversionIOTrackProducer_ = consumes<reco::TrackCollection>(iotracks);
  inOutTrackSCAssociationCollection_ = consumes<reco::TrackCaloClusterPtrAssociation>(iotracksassoc);

  generalTrackProducer_ = consumes<reco::TrackCollection>(conf_.getParameter<edm::InputTag>("generalTracksSrc"));

  algoName_ = conf_.getParameter<std::string>("AlgorithmName");

  hcalTowers_ = consumes<CaloTowerCollection>(conf_.getParameter<edm::InputTag>("hcalTowers"));
  hOverEConeSize_ = conf_.getParameter<double>("hOverEConeSize");
  maxHOverE_ = conf_.getParameter<double>("maxHOverE");
  minSCEt_ = conf_.getParameter<double>("minSCEt");
  recoverOneTrackCase_ = conf_.getParameter<bool>("recoverOneTrackCase");
  dRForConversionRecovery_ = conf_.getParameter<double>("dRForConversionRecovery");
  deltaCotCut_ = conf_.getParameter<double>("deltaCotCut");
  minApproachDisCut_ = conf_.getParameter<double>("minApproachDisCut");

  maxNumOfCandidates_ = conf_.getParameter<int>("maxNumOfCandidates");
  risolveAmbiguity_ = conf_.getParameter<bool>("risolveConversionAmbiguity");
  likelihoodWeights_ = conf_.getParameter<std::string>("MVA_weights_location");

  caloGeomToken_ = esConsumes();
  mFToken_ = esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>();
  transientTrackToken_ = esConsumes<TransientTrackBuilder, TransientTrackRecord, edm::Transition::BeginRun>(
      edm::ESInputTag("", "TransientTrackBuilder"));

  // use configuration file to setup output collection names
  ConvertedPhotonCollection_ = conf_.getParameter<std::string>("convertedPhotonCollection");
  CleanedConvertedPhotonCollection_ = conf_.getParameter<std::string>("cleanedConvertedPhotonCollection");

  // Register the product
  produces<reco::ConversionCollection>(ConvertedPhotonCollection_);
  produces<reco::ConversionCollection>(CleanedConvertedPhotonCollection_);

  // instantiate the Track Pair Finder algorithm
  theTrackPairFinder_ = new ConversionTrackPairFinder();
  edm::FileInPath path_mvaWeightFile(likelihoodWeights_.c_str());
  theLikelihoodCalc_ = new ConversionLikelihoodCalculator();
  theLikelihoodCalc_->setWeightsFile(path_mvaWeightFile.fullPath().c_str());
  // instantiate the Vertex Finder algorithm
  theVertexFinder_ = new ConversionVertexFinder(conf_);

  // Inizilize my global event counter
  nEvt_ = 0;
}

ConvertedPhotonProducer::~ConvertedPhotonProducer() {
  delete theTrackPairFinder_;
  delete theLikelihoodCalc_;
  delete theVertexFinder_;
}

void ConvertedPhotonProducer::beginRun(edm::Run const& r, edm::EventSetup const& theEventSetup) {
  //get magnetic field
  //edm::LogInfo("ConvertedPhotonProducer") << " get magnetic field" << "\n";
  theMF_ = theEventSetup.getHandle(mFToken_);

  // Transform Track into TransientTrack (needed by the Vertex fitter)
  theTransientTrackBuilder_ = theEventSetup.getHandle(transientTrackToken_);
}

void ConvertedPhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  using namespace edm;
  nEvt_++;

  //  LogDebug("ConvertedPhotonProducer")   << "ConvertedPhotonProduce::produce event number " <<   theEvent.id() << " Global counter " << nEvt_ << "\n";
  //  std::cout    << "ConvertedPhotonProduce::produce event number " <<   theEvent.id() << " Global counter " << nEvt_ << "\n";

  //
  // create empty output collections
  //
  // Converted photon candidates
  reco::ConversionCollection outputConvPhotonCollection;
  auto outputConvPhotonCollection_p = std::make_unique<reco::ConversionCollection>();
  // Converted photon candidates
  reco::ConversionCollection cleanedConversionCollection;
  auto cleanedConversionCollection_p = std::make_unique<reco::ConversionCollection>();

  // Get the Super Cluster collection in the Barrel
  bool validBarrelSCHandle = true;
  edm::Handle<edm::View<reco::CaloCluster> > scBarrelHandle;
  theEvent.getByToken(scHybridBarrelProducer_, scBarrelHandle);
  if (!scBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the scHybridBarrelProducer";
    validBarrelSCHandle = false;
  }

  // Get the Super Cluster collection in the Endcap
  bool validEndcapSCHandle = true;
  edm::Handle<edm::View<reco::CaloCluster> > scEndcapHandle;
  theEvent.getByToken(scIslandEndcapProducer_, scEndcapHandle);
  if (!scEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the scIslandEndcapProducer";
    validEndcapSCHandle = false;
  }

  //// Get the Out In CKF tracks from conversions
  bool validTrackInputs = true;
  Handle<reco::TrackCollection> outInTrkHandle;
  theEvent.getByToken(conversionOITrackProducer_, outInTrkHandle);
  if (!outInTrkHandle.isValid()) {
    //std::cout << "Error! Can't get the conversionOITrack " << "\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the conversionOITrack "
                                             << "\n";
    validTrackInputs = false;
  }
  //  LogDebug("ConvertedPhotonProducer")<< "ConvertedPhotonProducer  outInTrack collection size " << (*outInTrkHandle).size() << "\n";

  //// Get the association map between CKF Out In tracks and the SC where they originated
  Handle<reco::TrackCaloClusterPtrAssociation> outInTrkSCAssocHandle;
  theEvent.getByToken(outInTrackSCAssociationCollection_, outInTrkSCAssocHandle);
  if (!outInTrkSCAssocHandle.isValid()) {
    //  std::cout << "Error! Can't get the product " <<  outInTrackSCAssociationCollection_.c_str() <<"\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the outInTrackSCAssociationCollection)";
    validTrackInputs = false;
  }

  //// Get the In Out  CKF tracks from conversions
  Handle<reco::TrackCollection> inOutTrkHandle;
  theEvent.getByToken(conversionIOTrackProducer_, inOutTrkHandle);
  if (!inOutTrkHandle.isValid()) {
    // std::cout << "Error! Can't get the conversionIOTrack " << "\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the conversionIOTrack "
                                             << "\n";
    validTrackInputs = false;
  }
  //  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer inOutTrack collection size " << (*inOutTrkHandle).size() << "\n";

  //// Get the generalTracks if the recovery of one track cases is switched on

  Handle<reco::TrackCollection> generalTrkHandle;
  if (recoverOneTrackCase_) {
    theEvent.getByToken(generalTrackProducer_, generalTrkHandle);
    if (!generalTrkHandle.isValid()) {
      //std::cout << "Error! Can't get the genralTracks " << "\n";
      edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the genralTracks "
                                               << "\n";
    }
  }

  //// Get the association map between CKF in out tracks and the SC  where they originated
  Handle<reco::TrackCaloClusterPtrAssociation> inOutTrkSCAssocHandle;
  theEvent.getByToken(inOutTrackSCAssociationCollection_, inOutTrkSCAssocHandle);
  if (!inOutTrkSCAssocHandle.isValid()) {
    //std::cout << "Error! Can't get the product " <<  inOutTrackSCAssociationCollection_.c_str() <<"\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the inOutTrackSCAssociationCollection_.c_str()";
    validTrackInputs = false;
  }

  // Get the basic cluster collection in the Barrel
  edm::Handle<edm::View<reco::CaloCluster> > bcBarrelHandle;
  theEvent.getByToken(bcBarrelCollection_, bcBarrelHandle);
  if (!bcBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the bcBarrelCollection";
  }

  // Get the basic cluster collection in the Endcap
  edm::Handle<edm::View<reco::CaloCluster> > bcEndcapHandle;
  theEvent.getByToken(bcEndcapCollection_, bcEndcapHandle);
  if (!bcEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the bcEndcapCollection";
  }

  // get Hcal towers collection
  Handle<CaloTowerCollection> hcalTowersHandle;
  theEvent.getByToken(hcalTowers_, hcalTowersHandle);

  // get the geometry from the event setup:
  theCaloGeom_ = theEventSetup.getHandle(caloGeomToken_);

  if (validTrackInputs) {
    //do the conversion:
    std::vector<reco::TransientTrack> t_outInTrk = (*theTransientTrackBuilder_).build(outInTrkHandle);
    std::vector<reco::TransientTrack> t_inOutTrk = (*theTransientTrackBuilder_).build(inOutTrkHandle);

    ///// Find the +/- pairs
    std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr, CompareTwoTracksVectors> allPairs;
    allPairs = theTrackPairFinder_->run(
        t_outInTrk, outInTrkHandle, outInTrkSCAssocHandle, t_inOutTrk, inOutTrkHandle, inOutTrkSCAssocHandle);
    //LogDebug("ConvertedPhotonProducer")  << "ConvertedPhotonProducer  allPairs.size " << allPairs.size() << "\n";

    buildCollections(theEventSetup,
                     scBarrelHandle,
                     bcBarrelHandle,
                     hcalTowersHandle,
                     generalTrkHandle,
                     allPairs,
                     outputConvPhotonCollection);
    buildCollections(theEventSetup,
                     scEndcapHandle,
                     bcEndcapHandle,
                     hcalTowersHandle,
                     generalTrkHandle,
                     allPairs,
                     outputConvPhotonCollection);
  }

  // put the product in the event
  outputConvPhotonCollection_p->assign(outputConvPhotonCollection.begin(), outputConvPhotonCollection.end());
  //LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Putting in the event    converted photon candidates " << (*outputConvPhotonCollection_p).size() << "\n";
  const edm::OrphanHandle<reco::ConversionCollection> conversionHandle =
      theEvent.put(std::move(outputConvPhotonCollection_p), ConvertedPhotonCollection_);

  // Loop over barrel and endcap SC collections and fill the  photon collection
  if (validBarrelSCHandle)
    cleanCollections(scBarrelHandle, conversionHandle, cleanedConversionCollection);
  if (validEndcapSCHandle)
    cleanCollections(scEndcapHandle, conversionHandle, cleanedConversionCollection);

  cleanedConversionCollection_p->assign(cleanedConversionCollection.begin(), cleanedConversionCollection.end());
  theEvent.put(std::move(cleanedConversionCollection_p), CleanedConvertedPhotonCollection_);
}

void ConvertedPhotonProducer::buildCollections(
    edm::EventSetup const& es,
    const edm::Handle<edm::View<reco::CaloCluster> >& scHandle,
    const edm::Handle<edm::View<reco::CaloCluster> >& bcHandle,
    const edm::Handle<CaloTowerCollection>& hcalTowersHandle,
    const edm::Handle<reco::TrackCollection>& generalTrkHandle,
    std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr, CompareTwoTracksVectors>& allPairs,
    reco::ConversionCollection& outputConvPhotonCollection)

{
  // instantiate the algorithm for finding the position of the track extrapolation at the Ecal front face
  ConversionTrackEcalImpactPoint theEcalImpactPositionFinder(&(*theMF_));

  reco::Conversion::ConversionAlgorithm algo = reco::Conversion::algoByName(algoName_);

  std::vector<reco::TransientTrack> t_generalTrk;
  if (recoverOneTrackCase_)
    t_generalTrk = (*theTransientTrackBuilder_).build(generalTrkHandle);
  //const CaloGeometry* geometry = theCaloGeom_.product();

  //  Loop over SC in the barrel and reconstruct converted photons
  int myCands = 0;
  reco::CaloClusterPtrVector scPtrVec;
  for (auto const& aClus : scHandle->ptrs()) {
    // preselection based in Et and H/E cut
    if (aClus->energy() / cosh(aClus->eta()) <= minSCEt_)
      continue;
    const reco::CaloCluster* pClus = &(*aClus);
    const reco::SuperCluster* sc = dynamic_cast<const reco::SuperCluster*>(pClus);
    const CaloTowerCollection* hcalTowersColl = hcalTowersHandle.product();
    EgammaTowerIsolation towerIso(hOverEConeSize_, 0., 0., -1, hcalTowersColl);
    double HoE = towerIso.getTowerESum(sc) / sc->energy();
    if (HoE >= maxHOverE_)
      continue;
    /////

    std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;
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

    int nFound = 0;
    if (!allPairs.empty()) {
      nFound = 0;

      for (std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr>::const_iterator iPair = allPairs.begin();
           iPair != allPairs.end();
           ++iPair) {
        scPtrVec.clear();

        reco::Vertex theConversionVertex;
        reco::CaloClusterPtr caloPtr = iPair->second;
        if (!(aClus == caloPtr))
          continue;

        scPtrVec.push_back(aClus);
        nFound++;

        std::vector<math::XYZPointF> trkPositionAtEcal = theEcalImpactPositionFinder.find(iPair->first, bcHandle);
        std::vector<reco::CaloClusterPtr> matchingBC = theEcalImpactPositionFinder.matchingBC();

        minAppDist = -99;
        const std::string metname = "ConvertedPhotons|ConvertedPhotonProducer";
        if ((iPair->first).size() > 1) {
          try {
            theVertexFinder_->run(iPair->first, theConversionVertex);

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

            const reco::TrackTransientTrack* ttt =
                dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
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
          //    like = theLikelihoodCalc_->calculateLikelihood(newCandidate, es );
          like = theLikelihoodCalc_->calculateLikelihood(newCandidate);
          //    std::cout << "like = " << like << std::endl;
          newCandidate.setMVAout(like);
          outputConvPhotonCollection.push_back(newCandidate);

          myCands++;
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
          const reco::TrackTransientTrack* ttt =
              dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
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
            std::vector<reco::TransientTrack>::const_iterator iGoodGenTran;
            for (std::vector<reco::TransientTrack>::const_iterator iTran = t_generalTrk.begin();
                 iTran != t_generalTrk.end();
                 ++iTran) {
              const reco::TrackTransientTrack* ttt =
                  dynamic_cast<const reco::TrackTransientTrack*>(iTran->basicTransientTrack());
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
                iGoodGenTran = iTran;
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
                std::vector<reco::TransientTrack> mypair;
                mypair.push_back(*iTk);
                mypair.push_back(*iGoodGenTran);

                try {
                  theVertexFinder_->run(iPair->first, theConversionVertex);

                } catch (cms::Exception& e) {
                  //std::cout << " cms::Exception caught in ConvertedPhotonProducer::produce" << "\n" ;
                  edm::LogWarning(metname) << "cms::Exception caught in ConvertedPhotonProducer::produce\n"
                                           << e.explainSelf();
                }
              }
            }

          }  // bool On/Off one track case recovery using generalTracks
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
          like = theLikelihoodCalc_->calculateLikelihood(newCandidate);
          newCandidate.setMVAout(like);
          outputConvPhotonCollection.push_back(newCandidate);

        }  // case with only on track: looking in general tracks
      }
    }
  }
}

void ConvertedPhotonProducer::cleanCollections(const edm::Handle<edm::View<reco::CaloCluster> >& scHandle,
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
  std::multimap<double, reco::ConversionRef, std::greater<double> > convMap;

  for (unsigned int icp = 0; icp < conversionHandle->size(); icp++) {
    reco::ConversionRef cpRef(reco::ConversionRef(conversionHandle, icp));

    //std::cout << " cpRef " << cpRef->nTracks() << " " <<  cpRef ->caloCluster()[0]->energy() << std::endl;
    if (!(scRef.id() == cpRef->caloCluster()[0].id() && scRef.key() == cpRef->caloCluster()[0].key()))
      continue;
    if (!cpRef->isConverted())
      continue;
    double like = cpRef->MVAout();
    if (cpRef->nTracks() < 2)
      continue;
    //    std::cout << " Like " << like << std::endl;
    convMap.insert(std::make_pair(like, cpRef));
  }

  //  std::cout << " convMap size " << convMap.size() << std::endl;

  std::multimap<double, reco::ConversionRef>::iterator iMap;
  std::vector<reco::ConversionRef> bestRefs;
  for (iMap = convMap.begin(); iMap != convMap.end(); iMap++) {
    //    std::cout << " Like list in the map " <<  iMap->first << " " << (iMap->second)->EoverP() << std::endl;
    bestRefs.push_back(iMap->second);
    if (int(bestRefs.size()) == maxNumOfCandidates_)
      break;
  }

  return bestRefs;
}

float ConvertedPhotonProducer::calculateMinApproachDistance(const reco::TrackRef& track1,
                                                            const reco::TrackRef& track2) {
  float dist = 9999.;

  double x1, x2, y1, y2;
  double xx_1 = track1->innerPosition().x(), yy_1 = track1->innerPosition().y(), zz_1 = track1->innerPosition().z();
  double xx_2 = track2->innerPosition().x(), yy_2 = track2->innerPosition().y(), zz_2 = track2->innerPosition().z();
  double radius1 = track1->innerMomentum().Rho() / (.3 * (theMF_->inTesla(GlobalPoint(xx_1, yy_1, zz_1)).z())) * 100;
  double radius2 = track2->innerMomentum().Rho() / (.3 * (theMF_->inTesla(GlobalPoint(xx_2, yy_2, zz_2)).z())) * 100;
  getCircleCenter(track1, radius1, x1, y1);
  getCircleCenter(track2, radius2, x2, y2);
  dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) - radius1 - radius2;

  return dist;
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
