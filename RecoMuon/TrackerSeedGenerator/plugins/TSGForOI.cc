/**
  \class    TSGForOI
  \brief    Create L3MuonTrajectorySeeds from L2 Muons updated at vertex in an outside in manner
  \author   Benjamin Radburn-Smith
 */

#include "RecoMuon/TrackerSeedGenerator/plugins/TSGForOI.h"
#include <memory>

TSGForOI::TSGForOI(const edm::ParameterSet & iConfig) :
			src_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("src"))),
			numOfLayersToTry_(iConfig.getParameter<int32_t>("layersToTry")),
			numOfHitsToTry_(iConfig.getParameter<int32_t>("hitsToTry")),
			fixedErrorRescalingForHits_(iConfig.getParameter<double>("fixedErrorRescaleFactorForHits")),
			fixedErrorRescalingForHitless_(iConfig.getParameter<double>("fixedErrorRescaleFactorForHitless")),
			adjustErrorsDyanmicallyForHits_(iConfig.getParameter<bool>("adjustErrorsDyanmicallyForHits")),
			adjustErrorsDyanmicallyForHitless_(iConfig.getParameter<bool>("adjustErrorsDyanmicallyForHitless")),
		    estimatorName_(iConfig.getParameter<std::string>("estimator")),
			measurementTrackerTag_(consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("MeasurementTrackerEvent"))),
			minEtaForTEC_(iConfig.getParameter<double>("minEtaForTEC")),
			maxEtaForTOB_(iConfig.getParameter<double>("maxEtaForTOB")),
			useHitlessSeeds_(iConfig.getParameter<bool>("UseHitlessSeeds")),
			useHybridSeeds_(iConfig.getParameter<bool>("UseHybridSeeds")),
			dummyPlane_(Plane::build(Plane::PositionType(), Plane::RotationType())),
			updator_(new KFUpdator()),
			pT1_(iConfig.getParameter<double>("pT1")),
			pT2_(iConfig.getParameter<double>("pT2")),
			pT3_(iConfig.getParameter<double>("pT3")),
			eta1_(iConfig.getParameter<double>("eta1")),
			eta2_(iConfig.getParameter<double>("eta2")),
			SF1_(iConfig.getParameter<double>("SF1")),
			SF2_(iConfig.getParameter<double>("SF2")),
			SF3_(iConfig.getParameter<double>("SF3")),
			SF4_(iConfig.getParameter<double>("SF4")),
			SF5_(iConfig.getParameter<double>("SF5")),
			tsosDiffDeltaR_(iConfig.getParameter<double>("tsosDiffDeltaR"))
			{
	numOfMaxSeeds_=iConfig.getParameter<uint32_t>("maxSeeds");
	produces<std::vector<TrajectorySeed> >();
	foundCompatibleDet_=false;
	numSeedsMade_=0;
	theCategory = "Muon|RecoMuon|TSGForOI";
}


TSGForOI::~TSGForOI(){
}


void TSGForOI::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
	iSetup.get<IdealMagneticFieldRecord>().get(magfield_);
	iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial", propagatorOpposite_);
	iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial", propagatorAlong_);
	iSetup.get<GlobalTrackingGeometryRecord>().get(geometry_);
    iSetup.get<TrackingComponentsRecord>().get(estimatorName_,estimator_);
	iEvent.getByToken(measurementTrackerTag_, measurementTracker_);
	edm::Handle<reco::TrackCollection> l2TrackCol;					iEvent.getByToken(src_, l2TrackCol);

//	The product:
	std::auto_ptr<std::vector<TrajectorySeed> > result(new std::vector<TrajectorySeed>());

//	Get vector of Detector layers once:
	std::vector<BarrelDetLayer const*> const& tob = measurementTracker_->geometricSearchTracker()->tobLayers();
	std::vector<ForwardDetLayer const*> const& tecPositive = measurementTracker_->geometricSearchTracker()->posTecLayers();
	std::vector<ForwardDetLayer const*> const& tecNegative = measurementTracker_->geometricSearchTracker()->negTecLayers();

//	Get the suitable propagators:
	std::unique_ptr<Propagator> propagatorAlong = SetPropagationDirection(*propagatorAlong_,alongMomentum);
	std::unique_ptr<Propagator> propagatorOpposite = SetPropagationDirection(*propagatorOpposite_,oppositeToMomentum);

	edm::ESHandle<Propagator> SmartOpposite;
	edm::ESHandle<Propagator> SHPOpposite;
	iSetup.get<TrackingComponentsRecord>().get("hltESPSmartPropagatorAnyOpposite", SmartOpposite);
	iSetup.get<TrackingComponentsRecord>().get("hltESPSteppingHelixPropagatorOpposite", SHPOpposite);

//	Loop over the L2's and making seeds for all of them:
	edm::LogInfo(theCategory) << "TSGForOI::produce: Number of L2's: " << l2TrackCol->size();
	for (unsigned int l2TrackColIndex(0);l2TrackColIndex!=l2TrackCol->size();++l2TrackColIndex){
		const reco::TrackRef l2(l2TrackCol, l2TrackColIndex);
		std::auto_ptr<std::vector<TrajectorySeed> > out(new std::vector<TrajectorySeed>());

		FreeTrajectoryState fts = trajectoryStateTransform::initialFreeState(*l2, magfield_.product());
		dummyPlane_->move(fts.position() - dummyPlane_->position());
		TrajectoryStateOnSurface tsosAtIP = TrajectoryStateOnSurface(fts, *dummyPlane_);

		FreeTrajectoryState notUpdatedFts = trajectoryStateTransform::innerFreeState(*l2,magfield_.product());
		dummyPlane_->move(notUpdatedFts.position() - dummyPlane_->position());
		TrajectoryStateOnSurface tsosAtMuonSystem = TrajectoryStateOnSurface(notUpdatedFts, *dummyPlane_);

		if (useHybridSeeds_){
			StateOnTrackerBound fromInside(propagatorAlong.get());
			TrajectoryStateOnSurface outerTkStateInside = fromInside(fts);
			StateOnTrackerBound fromOutside(&*SmartOpposite);
			TrajectoryStateOnSurface outerTkStateOutside = fromOutside(notUpdatedFts);

			auto diffDeltaR=0.0;
			if (outerTkStateInside.isValid() && outerTkStateOutside.isValid()){
				diffDeltaR = deltaR(outerTkStateInside.globalMomentum(),outerTkStateOutside.globalMomentum());
			}
			if (diffDeltaR>tsosDiffDeltaR_){
				// Switch on hybrid: add a hit-based seed:
				++numOfMaxSeeds_;
			}
		} //Hybrid

		numSeedsMade_=0;
		foundCompatibleDet_=false;
		analysedL2_ = false;

//		BARREL
		if (std::abs(l2->eta()) < maxEtaForTOB_) {
			layerCount_ = 0;
			for (auto it=tob.rbegin(); it!=tob.rend(); ++it) {	//This goes from outermost to innermost layer
				findSeedsOnLayer(**it, tsosAtIP, tsosAtMuonSystem, *(propagatorAlong.get()), *(propagatorOpposite.get()), l2, out);
			}
		}

//		Reset Number of seeds if in overlap region:
		if (std::abs(l2->eta())>minEtaForTEC_ && std::abs(l2->eta())<maxEtaForTOB_){
			numSeedsMade_=0;
		}

//		ENDCAP+
		if (l2->eta() > minEtaForTEC_) {
			layerCount_ = 0;
			for (auto it=tecPositive.rbegin(); it!=tecPositive.rend(); ++it) {
				findSeedsOnLayer(**it, tsosAtIP, tsosAtMuonSystem, *(propagatorAlong.get()), *(propagatorOpposite.get()), l2, out);
			}
		}

//		ENDCAP-
		if (l2->eta() < -minEtaForTEC_) {
			layerCount_ = 0;
			for (auto it=tecNegative.rbegin(); it!=tecNegative.rend(); ++it) {
				findSeedsOnLayer(**it, tsosAtIP, tsosAtMuonSystem, *(propagatorAlong.get()), *(propagatorOpposite.get()), l2, out);
			}
		}

		for (std::vector<TrajectorySeed>::iterator it=out->begin(); it!=out->end(); ++it){
			result->push_back(*it);
		}

	} //L2Collection
	edm::LogInfo(theCategory) << "TSGForOI::produce: number of seeds made: " << result->size();

	iEvent.put(result);
}

void TSGForOI::findSeedsOnLayer(const GeometricSearchDet &layer,
		const TrajectoryStateOnSurface &tsosAtIP,
		const TrajectoryStateOnSurface &tsosAtMuonSystem,
		const Propagator& propagatorAlong,
		const Propagator& propagatorOpposite,
		const reco::TrackRef l2,
		std::auto_ptr<std::vector<TrajectorySeed> >& out) {

	if (numSeedsMade_<numOfMaxSeeds_){
		double errorSFHits_=1.0;
		double errorSFHitless_=1.0;
		if (!adjustErrorsDyanmicallyForHits_) errorSFHits_ = fixedErrorRescalingForHits_;
		if (!adjustErrorsDyanmicallyForHitless_) errorSFHitless_ = fixedErrorRescalingForHitless_;

//		Hitless:
		std::vector< GeometricSearchDet::DetWithState > dets;
		layer.compatibleDetsV(tsosAtIP, propagatorAlong, *estimator_, dets);
		if (dets.size()>0){
			auto const& detOnLayer = dets.front().first;
			auto const& tsosOnLayer = dets.front().second;
			// See if we need to adjust the Error Scale Factors:
			if (!analysedL2_ && adjustErrorsDyanmicallyForHitless_){
				if (!tsosOnLayer.isValid()){
					edm::LogInfo(theCategory) << "ERROR!: TSOS for SF is not valid!";
				}
				else{
					errorSFHitless_=calculateSFFromL2(layer,tsosAtMuonSystem,tsosOnLayer,propagatorOpposite,l2);
					analysedL2_=true;
				}
			}
			if (useHitlessSeeds_ || useHybridSeeds_){
				// Fill first seed from L2 using only state information:
				if (!foundCompatibleDet_){
					if (!tsosOnLayer.isValid()){
						edm::LogInfo(theCategory) << "ERROR!: Hitless TSOS is not valid!";
					}
					else{
						dets.front().second.rescaleError(errorSFHitless_);
						PTrajectoryStateOnDet const& PTSOD = trajectoryStateTransform::persistentState(tsosOnLayer,detOnLayer->geographicalId().rawId());
						TrajectorySeed::recHitContainer rHC;
						out->push_back(TrajectorySeed(PTSOD,rHC,oppositeToMomentum));
						foundCompatibleDet_=true;
					}
				}
				numSeedsMade_=out->size();
			}
		}

//		Hits:
		if (numSeedsMade_<numOfMaxSeeds_ && layerCount_<numOfLayersToTry_){
			if (makeSeedsFromHits(layer, tsosAtIP, *out, propagatorAlong, *measurementTracker_, errorSFHits_)) {
				++layerCount_; //Sucessfully made seeds from hits using this layer
			}
		}
		numSeedsMade_=out->size();
	}

}

double TSGForOI::calculateSFFromL2(const GeometricSearchDet& layer,
		const TrajectoryStateOnSurface& tsosAtMuonSystem,
		const TrajectoryStateOnSurface& tsosOnLayer,
		const Propagator& propagatorOpposite,
		const reco::TrackRef track){
	double theSF=1.0;

//	L2 direction vs pT blowup - as was previously done:
//	Split into 4 pT ranges: <pT1_, pT1_<pT2_, pT2_<pT3_, <pT4_: 13,30,70
//	Split into 2 eta ranges for the middle two pT ranges: 1.0,1.4
	if (track->pt()<=pT1_) theSF=SF1_;
	if (track->pt()>pT1_ && track->pt()<=pT2_){
		if (std::abs(track->eta())<=eta1_) theSF=SF3_;
		if (std::abs(track->eta())>eta1_ && std::abs(track->eta())<=eta2_) theSF=SF2_;
		if (std::abs(track->eta())>eta2_) theSF=SF3_;
	}
	if (track->pt()>pT2_ && track->pt()<=pT3_){
		if (std::abs(track->eta())<=eta1_) theSF=SF5_;
		if (std::abs(track->eta())>eta1_ && std::abs(track->eta())<=eta2_) theSF=SF4_;
		if (std::abs(track->eta())>eta2_) theSF=SF5_;
	}
	if (track->pt()>pT3_) theSF=SF5_;

	edm::LogInfo(theCategory) << "TSGForOI::calculateSFFromL2: SF has been calculated as: " << theSF;
	return theSF;
}


int TSGForOI::makeSeedsFromHits(const GeometricSearchDet &layer,
		const TrajectoryStateOnSurface &tsosAtIP,
		std::vector<TrajectorySeed> &out,
		const Propagator& propagatorAlong,
		const MeasurementTrackerEvent &measurementTracker,
		const double errorSF) {

	std::vector< GeometricSearchDet::DetWithState > dets;
	layer.compatibleDetsV(tsosAtIP, propagatorAlong, *estimator_, dets);

//	Find Measurements on each DetWithState:
	std::vector<TrajectoryMeasurement> meas;
	for (std::vector<GeometricSearchDet::DetWithState>::iterator it=dets.begin(); it!=dets.end(); ++it) {
		MeasurementDetWithData det = measurementTracker.idToDet(it->first->geographicalId());
		if (det.isNull()) {
			continue;
		}
		if (!it->second.isValid()) continue;	//Skip if TSOS is not valid

//		Error Rescaling:
		it->second.rescaleError(errorSF);

		std::vector < TrajectoryMeasurement > mymeas = det.fastMeasurements(it->second, tsosAtIP, propagatorAlong, *estimator_);	//Second TSOS is not used
		for (std::vector<TrajectoryMeasurement>::const_iterator it2 = mymeas.begin(), ed2 = mymeas.end(); it2 != ed2; ++it2) {
			if (it2->recHit()->isValid()) meas.push_back(*it2);	//Only save those which are valid
		}
	}

//	Update TSOS using TMs after sorting, then create Trajectory Seed and put into vector:
	unsigned int found = 0;
	std::sort(meas.begin(), meas.end(), TrajMeasLessEstim());
	for (std::vector<TrajectoryMeasurement>::const_iterator it=meas.begin(); it!=meas.end(); ++it) {
		TrajectoryStateOnSurface updatedTSOS = updator_->update(it->forwardPredictedState(), *it->recHit());
		if (updatedTSOS.isValid()) {
			edm::OwnVector<TrackingRecHit> seedHits;
			seedHits.push_back(*it->recHit()->hit());
			PTrajectoryStateOnDet const& pstate = trajectoryStateTransform::persistentState(updatedTSOS, it->recHit()->geographicalId().rawId());
			TrajectorySeed seed(pstate, std::move(seedHits), oppositeToMomentum);
			out.push_back(seed);
			found++;
			if (found == numOfHitsToTry_) break;
		}
	}
	return found;
}


void TSGForOI::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src",edm::InputTag("hltL2Muons","UpdatedAtVtx"));
  desc.add<int>("layersToTry",1);
  desc.add<double>("fixedErrorRescaleFactorForHitless",2.0);
  desc.add<int>("hitsToTry",1);
  desc.add<bool>("adjustErrorsDyanmicallyForHits",false);
  desc.add<bool>("adjustErrorsDyanmicallyForHitless",false);
  desc.add<edm::InputTag>("MeasurementTrackerEvent",edm::InputTag("hltSiStripClusters"));
  desc.add<bool>("UseHitlessSeeds",true);
  desc.add<bool>("UseHybridSeeds",false);
  desc.add<std::string>("estimator","hltESPChi2MeasurementEstimator100");
  desc.add<double>("maxEtaForTOB",1.2);
  desc.add<double>("minEtaForTEC",0.8);
  desc.addUntracked<bool>("debug",true);
  desc.add<double>("fixedErrorRescaleFactorForHits",2.0);
  desc.add<unsigned int>("maxSeeds",1);
  desc.add<double>("pT1",13.0);
  desc.add<double>("pT2",30.0);
  desc.add<double>("pT3",70.0);
  desc.add<double>("eta1",1.0);
  desc.add<double>("eta2",1.4);
  desc.add<double>("SF1",3.0);
  desc.add<double>("SF2",4.0);
  desc.add<double>("SF3",5.0);
  desc.add<double>("SF4",7.0);
  desc.add<double>("SF5",10.0);
  desc.add<double>("tsosDiffDeltaR",0.03);
  descriptions.add("TSGForOI",desc);
}

DEFINE_FWK_MODULE(TSGForOI);
