/**
  \class    TSGForOI
  \brief    Create L3MuonTrajectorySeeds from L2 Muons updated at vertex in an outside-in manner
  \author   Santiago Folgueras,Benjamin, Bibhuprasad Mahakud, Jan Frederik Schulte (Purdue University, West Lafayette, USA)
 */

#include "RecoMuon/TrackerSeedGenerator/plugins/TSGForOI.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <memory>

using namespace edm;
using namespace std;

TSGForOI::TSGForOI(const edm::ParameterSet & iConfig) :
  src_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("src"))),
  maxSeeds_(iConfig.getParameter<uint32_t>("maxSeeds")),
  maxHitlessSeeds_(iConfig.getParameter<uint32_t>("maxHitlessSeeds")),
  maxHitSeeds_(iConfig.getParameter<uint32_t>("maxHitSeeds")),
  numOfLayersToTry_(iConfig.getParameter<int32_t>("layersToTry")),
  numOfHitsToTry_(iConfig.getParameter<int32_t>("hitsToTry")),
  numL2ValidHitsCutAllEta_(iConfig.getParameter<uint32_t>("numL2ValidHitsCutAllEta")),
  numL2ValidHitsCutAllEndcap_(iConfig.getParameter<uint32_t>("numL2ValidHitsCutAllEndcap")),  
  fixedErrorRescalingForHits_(iConfig.getParameter<double>("fixedErrorRescaleFactorForHits")),
  fixedErrorRescalingForHitless_(iConfig.getParameter<double>("fixedErrorRescaleFactorForHitless")),
  adjustErrorsDynamicallyForHits_(iConfig.getParameter<bool>("adjustErrorsDynamicallyForHits")),
  adjustErrorsDynamicallyForHitless_(iConfig.getParameter<bool>("adjustErrorsDynamicallyForHitless")),
  estimatorName_(iConfig.getParameter<std::string>("estimator")),
  minEtaForTEC_(iConfig.getParameter<double>("minEtaForTEC")),
  maxEtaForTOB_(iConfig.getParameter<double>("maxEtaForTOB")),
  useHitLessSeeds_(iConfig.getParameter<bool>("UseHitLessSeeds")),
  updator_(new KFUpdator()), measurementTrackerTag_(consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("MeasurementTrackerEvent"))),
  pT1_(iConfig.getParameter<double>("pT1")),
  pT2_(iConfig.getParameter<double>("pT2")),
  pT3_(iConfig.getParameter<double>("pT3")),
  eta1_(iConfig.getParameter<double>("eta1")),
  eta2_(iConfig.getParameter<double>("eta2")),
  eta3_(iConfig.getParameter<double>("eta3")),
  eta4_(iConfig.getParameter<double>("eta4")),
  eta5_(iConfig.getParameter<double>("eta5")),
  eta6_(iConfig.getParameter<double>("eta6")),
  eta7_(iConfig.getParameter<double>("eta7")),
  SF1_(iConfig.getParameter<double>("SF1")),
  SF2_(iConfig.getParameter<double>("SF2")),
  SF3_(iConfig.getParameter<double>("SF3")),
  SF4_(iConfig.getParameter<double>("SF4")),
  SF5_(iConfig.getParameter<double>("SF5")),
  SF6_(iConfig.getParameter<double>("SF6")),
  tsosDiff1_(iConfig.getParameter<double>("tsosDiff1")),
  tsosDiff2_(iConfig.getParameter<double>("tsosDiff2")),
  propagatorName_(iConfig.getParameter<std::string>("propagatorName")), 
  theCategory(string("Muon|RecoMuon|TSGForOI"))
{
  produces<std::vector<TrajectorySeed> >();
}


TSGForOI::~TSGForOI() {

}


//
// Produce seeds
//
void TSGForOI::produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
    
  using namespace std;



  // Initialize variables
  unsigned int numSeedsMade = 0;
  unsigned int layerCount = 0;
  unsigned int hitlessSeedsMadeIP = 0;
  unsigned int hitlessSeedsMadeMuS = 0;
  unsigned int hitSeedsMade = 0;

  // Surface used to make a TSOS at the PCA to the beamline
  Plane::PlanePointer dummyPlane = Plane::build(Plane::PositionType(), Plane::RotationType());

  // Read ESHandles
  edm::Handle<MeasurementTrackerEvent>          measurementTrackerH;
  edm::ESHandle<Chi2MeasurementEstimatorBase>   estimatorH;
  edm::ESHandle<MagneticField>                  magfieldH;
  edm::ESHandle<Propagator>                     propagatorAlongH;
  edm::ESHandle<Propagator>                     propagatorOppositeH;
  edm::ESHandle<TrackerGeometry>                tmpTkGeometryH;
  edm::ESHandle<GlobalTrackingGeometry>         geometryH;

  iSetup.get<IdealMagneticFieldRecord>().get(magfieldH);
  iSetup.get<TrackingComponentsRecord>().get(propagatorName_, propagatorOppositeH);
  iSetup.get<TrackingComponentsRecord>().get(propagatorName_, propagatorAlongH);
  iSetup.get<GlobalTrackingGeometryRecord>().get(geometryH);
  iSetup.get<TrackerDigiGeometryRecord>().get(tmpTkGeometryH);
  iSetup.get<TrackingComponentsRecord>().get(estimatorName_,estimatorH);
  iEvent.getByToken(measurementTrackerTag_, measurementTrackerH);

  // Read L2 track collection
  edm::Handle<reco::TrackCollection> l2TrackCol;
  iEvent.getByToken(src_, l2TrackCol);

  // The product
  std::unique_ptr<std::vector<TrajectorySeed> > result(new std::vector<TrajectorySeed>());

  // Get vector of Detector layers
  std::vector<BarrelDetLayer const*> const& tob = measurementTrackerH->geometricSearchTracker()->tobLayers();
  std::vector<ForwardDetLayer const*> const& tecPositive = tmpTkGeometryH->isThere(GeomDetEnumerators::P2OTEC) ? 
                                                                measurementTrackerH->geometricSearchTracker()->posTidLayers() : 
                                                                measurementTrackerH->geometricSearchTracker()->posTecLayers(); 
  std::vector<ForwardDetLayer const*> const& tecNegative = tmpTkGeometryH->isThere(GeomDetEnumerators::P2OTEC) ? 
                                                                measurementTrackerH->geometricSearchTracker()->negTidLayers() : 
                                                                measurementTrackerH->geometricSearchTracker()->negTecLayers();

  // Get suitable propagators
  std::unique_ptr<Propagator> propagatorAlong = SetPropagationDirection(*propagatorAlongH,alongMomentum);
  std::unique_ptr<Propagator> propagatorOpposite = SetPropagationDirection(*propagatorOppositeH,oppositeToMomentum);

  // Stepping Helix Propagator for propogation from muon system to tracker
  edm::ESHandle<Propagator> SHPOpposite;
  iSetup.get<TrackingComponentsRecord>().get("hltESPSteppingHelixPropagatorOpposite", SHPOpposite);

  // Loop over the L2's and make seeds for all of them
  LogTrace(theCategory) << "TSGForOI::produce: Number of L2's: " << l2TrackCol->size();
  for (unsigned int l2TrackColIndex(0); l2TrackColIndex!=l2TrackCol->size(); ++l2TrackColIndex) {

    const reco::TrackRef l2(l2TrackCol, l2TrackColIndex);

    // Container of Seeds
    std::vector<TrajectorySeed> out;
    LogTrace("TSGForOI") << "TSGForOI::produce: L2 muon pT, eta, phi --> " << l2->pt() << " , " << l2->eta() << " , " << l2->phi() << endl;
 
    FreeTrajectoryState fts = trajectoryStateTransform::initialFreeState(*l2, magfieldH.product());

    dummyPlane->move(fts.position() - dummyPlane->position());
    TrajectoryStateOnSurface tsosAtIP = TrajectoryStateOnSurface(fts, *dummyPlane);
    LogTrace("TSGForOI") << "TSGForOI::produce: Created TSOSatIP: " << tsosAtIP << std::endl;
    
    // Get the TSOS on the innermost layer of the L2
    TrajectoryStateOnSurface tsosAtMuonSystem = trajectoryStateTransform::innerStateOnSurface(*l2, *geometryH, magfieldH.product());
    LogTrace("TSGForOI") << "TSGForOI::produce: Created TSOSatMuonSystem: " << tsosAtMuonSystem <<endl;
    
    LogTrace("TSGForOI") << "TSGForOI::produce: Check the error of the L2 parameter and use hit seeds if big errors" << endl;

    StateOnTrackerBound fromInside(propagatorAlong.get());
    TrajectoryStateOnSurface outerTkStateInside = fromInside(fts);

    StateOnTrackerBound fromOutside(&*SHPOpposite);
    TrajectoryStateOnSurface outerTkStateOutside = fromOutside(tsosAtMuonSystem);

    // Check if the two positions (using updated and not-updated TSOS) agree withing certain extent.
    // If both TSOSs agree, use only the one at vertex, as it uses more information. If they do not agree, search for seeds based on both.
    bool useBoth = false;
    if (outerTkStateInside.isValid() && outerTkStateOutside.isValid()) {
      //following commented out variables dist1 (5 par compatibility of tsos at outertracker surface)  
      //dist2 (angle between two tsos) could further be explored in combination of L2 valid hits for seeding. So kept for 
      //future developers 
      //auto dist1 = match_Chi2(outerTkStateInside,outerTkStateOutside);//for future developers
      //auto dist2 = deltaR(outerTkStateInside.globalMomentum(),outerTkStateOutside.globalMomentum());//for future developers
      //if ((dist1 > tsosDiff1_ || dist2 > tsosDiff2_) && l2->numberOfValidHits() < 20) useBoth = true;//for future developers
      if (l2->numberOfValidHits() < numL2ValidHitsCutAllEta_) useBoth = true;
      if (l2->numberOfValidHits() < numL2ValidHitsCutAllEndcap_ && (std::abs(l2->eta()) > eta7_)) useBoth = true;
      if (std::abs(l2->eta()) > eta1_ && std::abs(l2->eta()) < eta1_) useBoth = true;
    }

    numSeedsMade = 0;
    hitlessSeedsMadeIP = 0;
    hitlessSeedsMadeMuS = 0;
    hitSeedsMade = 0;

    // calculate scale factors
    double errorSFHits = 1.0;
    double errorSFHitless = 1.0;
    if (!adjustErrorsDynamicallyForHits_) errorSFHits = fixedErrorRescalingForHits_;
    else                                  errorSFHits = calculateSFFromL2(l2);
    if (!adjustErrorsDynamicallyForHitless_) errorSFHitless = fixedErrorRescalingForHitless_;
    else                                  errorSFHitless = calculateSFFromL2(l2);
      
    // BARREL
    if (std::abs(l2->eta()) < maxEtaForTOB_) {
      layerCount = 0;
      for (auto it=tob.rbegin(); it!=tob.rend(); ++it) { 
        LogTrace("TSGForOI") << "TSGForOI::produce: looping in TOB layer " << layerCount << endl;
        if ( useHitLessSeeds_ && hitlessSeedsMadeIP < maxHitlessSeeds_ && numSeedsMade < maxSeeds_ ) 
           makeSeedsWithoutHits(**it, tsosAtIP, *(propagatorAlong.get()), estimatorH, errorSFHitless, hitlessSeedsMadeIP, numSeedsMade, out);

        // Do not create hitbased seeds in barrel region
        if (std::abs(l2->eta()) > 1.0 && hitSeedsMade < maxHitSeeds_ && numSeedsMade < maxSeeds_ ) 
           makeSeedsFromHits(**it, tsosAtIP, *(propagatorAlong.get()), estimatorH, measurementTrackerH, errorSFHits, hitSeedsMade, numSeedsMade, layerCount, out);

        if (useBoth) {
          if ( useHitLessSeeds_ && hitlessSeedsMadeMuS < maxHitlessSeeds_ && numSeedsMade < maxSeeds_ ) 
             makeSeedsWithoutHits(**it, outerTkStateOutside, *(propagatorOpposite.get()), estimatorH, errorSFHitless, hitlessSeedsMadeMuS, numSeedsMade, out);
        }
      }
      LogTrace("TSGForOI") << "TSGForOI:::produce: NumSeedsMade = " << numSeedsMade << " , layerCount = " << layerCount << endl;
    }

    // Reset number of seeds if in overlap region
    if (std::abs(l2->eta()) > minEtaForTEC_ && std::abs(l2->eta()) < maxEtaForTOB_) {
      numSeedsMade = 0;
      hitlessSeedsMadeIP = 0;
      hitlessSeedsMadeMuS = 0;
      hitSeedsMade = 0;
    }

    // ENDCAP+
    if (l2->eta() > minEtaForTEC_) {
      layerCount = 0;
      for (auto it=tecPositive.rbegin(); it!=tecPositive.rend(); ++it) {
        LogTrace("TSGForOI") << "TSGForOI::produce: looping in TEC+ layer " << layerCount << endl;
        if ( useHitLessSeeds_ && hitlessSeedsMadeIP < maxHitlessSeeds_ && numSeedsMade < maxSeeds_ ) 
           makeSeedsWithoutHits(**it, tsosAtIP, *(propagatorAlong.get()), estimatorH, errorSFHitless, hitlessSeedsMadeIP, numSeedsMade, out);

        if (std::abs(l2->eta()) > 1.0 && hitSeedsMade < maxHitSeeds_ && numSeedsMade < maxSeeds_ ) 
           makeSeedsFromHits(**it, tsosAtIP, *(propagatorAlong.get()), estimatorH, measurementTrackerH, errorSFHits, hitSeedsMade, numSeedsMade, layerCount, out);

        if (useBoth) {
          if ( useHitLessSeeds_ && hitlessSeedsMadeMuS < maxHitlessSeeds_ && numSeedsMade < maxSeeds_ ) 
             makeSeedsWithoutHits(**it, outerTkStateOutside, *(propagatorOpposite.get()), estimatorH, errorSFHitless, hitlessSeedsMadeMuS, numSeedsMade, out);
        } 
      }
      LogTrace("TSGForOI") << "TSGForOI:::produce: NumSeedsMade = " << numSeedsMade << " , layerCount = " << layerCount << endl;
    }

    // ENDCAP-
    if (l2->eta() < -minEtaForTEC_) {
      layerCount = 0;
      for (auto it=tecNegative.rbegin(); it!=tecNegative.rend(); ++it) {
        LogTrace("TSGForOI") << "TSGForOI::produce: looping in TEC- layer " << layerCount << endl;
        if ( useHitLessSeeds_ && hitlessSeedsMadeIP < maxHitlessSeeds_ && numSeedsMade < maxSeeds_ ) 
           makeSeedsWithoutHits(**it, tsosAtIP, *(propagatorAlong.get()), estimatorH, errorSFHitless, hitlessSeedsMadeIP, numSeedsMade, out);

        if (std::abs(l2->eta()) > 1.0 && hitSeedsMade < maxHitSeeds_ && numSeedsMade < maxSeeds_ ) 
           makeSeedsFromHits(**it, tsosAtIP, *(propagatorAlong.get()), estimatorH, measurementTrackerH, errorSFHits, hitSeedsMade, numSeedsMade, layerCount, out);

        if (useBoth) {
          if ( useHitLessSeeds_ && hitlessSeedsMadeMuS < maxHitlessSeeds_ && numSeedsMade < maxSeeds_ ) 
             makeSeedsWithoutHits(**it, outerTkStateOutside, *(propagatorOpposite.get()), estimatorH, errorSFHitless, hitlessSeedsMadeMuS, numSeedsMade, out);
        }
      }
      LogTrace("TSGForOI") << "TSGForOI:::produce: NumSeedsMade = " << numSeedsMade << " , layerCount = " << layerCount << endl;
    }

    for (std::vector<TrajectorySeed>::iterator it=out.begin(); it!=out.end(); ++it) {
      result->push_back(*it);
    }

  } // L2Collection
 
  edm::LogInfo(theCategory) << "TSGForOI::produce: number of seeds made: " << result->size();

  iEvent.put(std::move(result));
}


//
// Create seeds without hits on a given layer (TOB or TEC)
//
void TSGForOI::makeSeedsWithoutHits(
                                const GeometricSearchDet& layer,
                                const TrajectoryStateOnSurface& tsos,
                                const Propagator& propagatorAlong,
                                edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                                double errorSF,
                                unsigned int& hitlessSeedsMade,
                                unsigned int& numSeedsMade,
                                std::vector<TrajectorySeed>& out) const {
    
  // create hitless seeds
  LogTrace("TSGForOI") << "TSGForOI::makeSeedsWithoutHits: Start hitless" << endl;
  std::vector< GeometricSearchDet::DetWithState > dets;
  layer.compatibleDetsV(tsos, propagatorAlong, *estimator, dets);
  if (!dets.empty()) {
    auto const& detOnLayer = dets.front().first;
    auto const& tsosOnLayer = dets.front().second;
    LogTrace("TSGForOI") << "TSGForOI::makeSeedsWithoutHits: tsosOnLayer " << tsosOnLayer << endl;
    if (!tsosOnLayer.isValid()) {
      edm::LogInfo(theCategory) << "ERROR!: Hitless TSOS is not valid!";
    }
    else {
      dets.front().second.rescaleError(errorSF);
      PTrajectoryStateOnDet const& ptsod = trajectoryStateTransform::persistentState(tsosOnLayer,detOnLayer->geographicalId().rawId());
      TrajectorySeed::recHitContainer rHC;
      out.push_back(TrajectorySeed(ptsod,rHC,oppositeToMomentum));
      LogTrace("TSGForOI") << "TSGForOI::makeSeedsWithoutHits: TSOD (Hitless) done " << endl;
      hitlessSeedsMade++;
      numSeedsMade++;
   }
 }
    
}


//
// Find hits on a given layer (TOB or TEC) and create seeds from updated TSOS with hit
//
void TSGForOI::makeSeedsFromHits(
                const GeometricSearchDet& layer,
                const TrajectoryStateOnSurface& tsos,
                const Propagator& propagatorAlong,
                edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                edm::Handle<MeasurementTrackerEvent>& measurementTracker,
                double errorSF,
                unsigned int& hitSeedsMade,
                unsigned int& numSeedsMade,
                unsigned int& layerCount,
                std::vector<TrajectorySeed>& out) const {

  if (layerCount > numOfLayersToTry_) return;

  // Error Rescaling
  TrajectoryStateOnSurface onLayer(tsos);
  onLayer.rescaleError(errorSF);

  std::vector< GeometricSearchDet::DetWithState > dets;
  layer.compatibleDetsV(onLayer, propagatorAlong, *estimator, dets);

  // Find Measurements on each DetWithState
  LogTrace("TSGForOI") << "TSGForOI::makeSeedsFromHits: Find measurements on each detWithState  " << dets.size() << endl;
  std::vector<TrajectoryMeasurement> meas;
  for (std::vector<GeometricSearchDet::DetWithState>::iterator it=dets.begin(); it!=dets.end(); ++it) {
    MeasurementDetWithData det = measurementTracker->idToDet(it->first->geographicalId());
    if (det.isNull()) continue;
    if (!it->second.isValid()) continue;	// Skip if TSOS is not valid

    std::vector <TrajectoryMeasurement> mymeas = det.fastMeasurements(it->second, onLayer, propagatorAlong, *estimator);	// Second TSOS is not used
    for (std::vector<TrajectoryMeasurement>::const_iterator it2 = mymeas.begin(), ed2 = mymeas.end(); it2 != ed2; ++it2) {
      if (it2->recHit()->isValid()) meas.push_back(*it2);	// Only save those which are valid
    }
  }
  
  // Update TSOS using TMs after sorting, then create Trajectory Seed and put into vector
  LogTrace("TSGForOI") << "TSGForOI::makeSeedsFromHits: Update TSOS using TMs after sorting, then create Trajectory Seed, number of TM = " << meas.size() << endl;
  std::sort(meas.begin(), meas.end(), TrajMeasLessEstim());

  unsigned int found = 0;
  for (std::vector<TrajectoryMeasurement>::const_iterator it=meas.begin(); it!=meas.end(); ++it) {
    TrajectoryStateOnSurface updatedTSOS = updator_->update(it->forwardPredictedState(), *it->recHit());
    LogTrace("TSGForOI") << "TSGForOI::makeSeedsFromHits: TSOS for TM " << found << endl;
    if (not updatedTSOS.isValid()) continue;

    
    edm::OwnVector<TrackingRecHit> seedHits;
    seedHits.push_back(*it->recHit()->hit());
    PTrajectoryStateOnDet const& pstate = trajectoryStateTransform::persistentState(updatedTSOS, it->recHit()->geographicalId().rawId());
    TrajectorySeed seed(pstate, std::move(seedHits), oppositeToMomentum);
    LogTrace("TSGForOI") << "TSGForOI::makeSeedsFromHits: Number of seedHits: " << seedHits.size() << endl;
    out.push_back(seed);
    found++;
    numSeedsMade++;
    hitSeedsMade++;
    if (found == numOfHitsToTry_) break;
    if (hitSeedsMade > maxHitSeeds_) return;
  }
 
  if (found) layerCount++;

}


//
// Calculate the dynamic error SF by analysing the L2
//
double TSGForOI::calculateSFFromL2(const reco::TrackRef track) const {
    
  double theSF = 1.0;
  // L2 direction vs pT blowup - as was previously done:
  // Split into 4 pT ranges: <pT1_, pT1_<pT2_, pT2_<pT3_, <pT4_: 13,30,70
  // Split into different eta ranges depending in pT
  double abseta = std::abs(track->eta());
  if (track->pt() <= pT1_) theSF = SF1_;
  if (track->pt() > pT1_ && track->pt() <= pT2_) {
    if (abseta <= eta3_) theSF = SF3_;
    if (abseta > eta3_ && abseta <= eta6_) theSF = SF2_;
    if (abseta > eta6_) theSF = SF3_;
  }
  if (track->pt() > pT2_ && track->pt() <= pT3_) {
    if (abseta <= eta1_) theSF = SF6_;
    if (abseta > eta1_  && abseta <= eta2_) theSF = SF4_;
    if (abseta > eta2_  && abseta <= eta3_) theSF = SF6_;
    if (abseta > eta3_ && abseta <= eta4_) theSF = SF1_;
    if (abseta > eta4_ && abseta <= eta5_) theSF = SF1_;
    if (abseta > eta5_ ) theSF = SF5_;
  }
    
  if (track->pt() > pT3_) {
    if (abseta <= eta3_) theSF = SF5_;
    if (abseta > eta3_ && abseta <= eta4_) theSF = SF4_;
    if (abseta > eta4_ && abseta <= eta5_) theSF = SF4_;
    if (abseta > eta5_ ) theSF = SF5_;
  }
    
  LogTrace(theCategory) << "TSGForOI::calculateSFFromL2: SF has been calculated as: " << theSF;
    
  return theSF;
    
}


//
// calculate Chi^2 of two trajectory states
//
double TSGForOI::match_Chi2(const TrajectoryStateOnSurface& tsos1,
                            const TrajectoryStateOnSurface& tsos2) const {

  if ( !tsos1.isValid() || !tsos2.isValid() ) return -1.;

  AlgebraicVector5 v(tsos1.localParameters().vector() - tsos2.localParameters().vector());
  AlgebraicSymMatrix55 m(tsos1.localError().matrix() + tsos2.localError().matrix());

  bool ierr = !m.Invert();

  if ( ierr ) {
    edm::LogInfo("TSGForOI") << "Error inverting covariance matrix";
    return -1;
  }

  double est = ROOT::Math::Similarity(v,m);

  return est;

}


//
//
//
void TSGForOI::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src",edm::InputTag("hltL2Muons","UpdatedAtVtx"));
  desc.add<int>("layersToTry",2);
  desc.add<double>("fixedErrorRescaleFactorForHitless",2.0);
  desc.add<int>("hitsToTry",1);
  desc.add<bool>("adjustErrorsDynamicallyForHits",false);
  desc.add<bool>("adjustErrorsDynamicallyForHitless",true);
  desc.add<edm::InputTag>("MeasurementTrackerEvent",edm::InputTag("hltSiStripClusters"));
  desc.add<bool>("UseHitLessSeeds",true);
  desc.add<std::string>("estimator","hltESPChi2MeasurementEstimator100");
  desc.add<double>("maxEtaForTOB",1.8);
  desc.add<double>("minEtaForTEC",0.7);
  desc.addUntracked<bool>("debug",false);
  desc.add<double>("fixedErrorRescaleFactorForHits",1.0);
  desc.add<unsigned int>("maxSeeds",20);
  desc.add<unsigned int>("maxHitlessSeeds",5);
  desc.add<unsigned int>("maxHitSeeds",1);
  desc.add<unsigned int>("numL2ValidHitsCutAllEta",20);
  desc.add<unsigned int>("numL2ValidHitsCutAllEndcap",30);
  desc.add<double>("pT1",13.0);
  desc.add<double>("pT2",30.0);
  desc.add<double>("pT3",70.0);
  desc.add<double>("eta1",0.2);
  desc.add<double>("eta2",0.3);
  desc.add<double>("eta3",1.0);
  desc.add<double>("eta4",1.2);
  desc.add<double>("eta5",1.6);
  desc.add<double>("eta6",1.4);
  desc.add<double>("eta7",2.1);
  desc.add<double>("SF1",3.0);
  desc.add<double>("SF2",4.0);
  desc.add<double>("SF3",5.0);
  desc.add<double>("SF4",7.0);
  desc.add<double>("SF5",10.0);
  desc.add<double>("SF6",2.0);
  desc.add<double>("tsosDiff1",0.2);
  desc.add<double>("tsosDiff2",0.02);
  desc.add<std::string>("propagatorName","PropagatorWithMaterialParabolicMf");
  descriptions.add("TSGForOI",desc);

}

DEFINE_FWK_MODULE(TSGForOI);
