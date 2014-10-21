#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

// added by me

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Error.h"
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h"
#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"


#include <boost/regex.hpp>

/**
 * Configurables:
 *
 *   Generic:
 *     tracks                  = InputTag of a collection of tracks to read from
 *     minimumHits             = Minimum hits that the output TrackCandidate must have to be saved
 *     replaceWithInactiveHits = instead of discarding hits, replace them with a invalid "inactive" hits,
 *                               so multiple scattering is accounted for correctly.
 *     stripFrontInvalidHits   = strip invalid hits at the beginning of the track
 *     stripBackInvalidHits    = strip invalid hits at the end of the track
 *     stripAllInvalidHits     = remove ALL invald hits (might be a problem for multiple scattering, use with care!)
 *
 *   Per structure:
 *      commands = list of commands, to be applied in order as they are written
 *      commands can be:
 *          "keep XYZ"  , "drop XYZ"    (XYZ = PXB, PXE, TIB, TID, TOB, TEC)
 *          "keep XYZ l", "drop XYZ n"  (XYZ as above, n = layer, wheel or disk = 1 .. 6 ; positive and negative are the same )
 *
 *   Individual modules:
 *     detsToIgnore        = individual list of detids on which hits must be discarded
 */
namespace reco { namespace modules {
	class CosmicTrackSplitter : public edm::EDProducer {
    public:
		CosmicTrackSplitter(const edm::ParameterSet &iConfig) ;
		virtual void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;

    private:
		edm::EDGetTokenT<reco::TrackCollection> tokenTracks;
		edm::EDGetTokenT<TrajTrackAssociationCollection> tokenTrajTrack;
		int totalTracks_;
		size_t minimumHits_;

		bool replaceWithInactiveHits_;
		bool stripFrontInvalidHits_;
		bool stripBackInvalidHits_;
		bool stripAllInvalidHits_;
		bool excludePixelHits_;

		double dZcut_;
		double dXYcut_;

		std::vector<uint32_t> detsToIgnore_;

		edm::ESHandle<TrackerGeometry> theGeometry;
		edm::ESHandle<MagneticField>   theMagField;

		TrackCandidate makeCandidate(const reco::Track &tk, std::vector<TrackingRecHit *>::iterator hitsBegin, std::vector<TrackingRecHit *>::iterator hitsEnd) ;

	}; // class


	CosmicTrackSplitter::CosmicTrackSplitter(const edm::ParameterSet &iConfig) :
    minimumHits_(iConfig.getParameter<uint32_t>("minimumHits")),
    replaceWithInactiveHits_(iConfig.getParameter<bool>("replaceWithInactiveHits")),
    stripFrontInvalidHits_(iConfig.getParameter<bool>("stripFrontInvalidHits")),
    stripBackInvalidHits_( iConfig.getParameter<bool>("stripBackInvalidHits") ),
    stripAllInvalidHits_(  iConfig.getParameter<bool>("stripAllInvalidHits")  ),
	excludePixelHits_(  iConfig.getParameter<bool>("excludePixelHits")  ),
    dZcut_(iConfig.getParameter<double>("dzCut") ),
    dXYcut_(iConfig.getParameter<double>("dxyCut") ),
    detsToIgnore_( iConfig.getParameter<std::vector<uint32_t> >("detsToIgnore") )
	{
		// sanity check
		if (stripAllInvalidHits_ && replaceWithInactiveHits_) {
			throw cms::Exception("Configuration") << "Inconsistent Configuration: you can't set both 'stripAllInvalidHits' and 'replaceWithInactiveHits' to true\n";
		}
		tokenTracks= consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));
		tokenTrajTrack = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("tjTkAssociationMapTag") );

		LogDebug("CosmicTrackSplitter") << "sanity check";

		// sort detids to ignore
		std::sort(detsToIgnore_.begin(), detsToIgnore_.end());

		totalTracks_ = 0;

		// issue the produce<>
		produces<TrackCandidateCollection>();
	}

	void
	CosmicTrackSplitter::produce(edm::Event &iEvent, const edm::EventSetup &iSetup)
	{
		LogDebug("CosmicTrackSplitter") << "IN THE SPLITTER!!!!!";

		// read with View, so we can read also a TrackRefVector
		edm::Handle<std::vector<reco::Track> > tracks;
		iEvent.getByToken(tokenTracks, tracks);

		// also need trajectories ...
		// Retrieve trajectories and tracks from the event
		// -> merely skip if collection is empty
		edm::Handle<TrajTrackAssociationCollection> m_TrajTracksMap;
		iEvent.getByToken( tokenTrajTrack, m_TrajTracksMap );

		// read from EventSetup
		iSetup.get<TrackerDigiGeometryRecord>().get(theGeometry);
		iSetup.get<IdealMagneticFieldRecord>().get(theMagField);

		// prepare output collection
		std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection());
		output->reserve(tracks->size());

		// working area and tools
		std::vector<TrackingRecHit *> hits;

		// Form pairs of trajectories and tracks
		//ConstTrajTrackPairCollection trajTracks;
		LogDebug("CosmicTrackSplitter") << "size of map: " << m_TrajTracksMap->size();
		int HITTOSPLITFROM = 0;
		for ( TrajTrackAssociationCollection::const_iterator iPair = m_TrajTracksMap->begin(); iPair != m_TrajTracksMap->end(); iPair++ ){
			const Trajectory* trajFromMap = &(*(*iPair).key);
			const reco::Track* trackFromMap = &(*(*iPair).val);

			// loop to find the hit to split from (by taking dot product of pT and transverse position
			std::vector<TrajectoryMeasurement> measurements = trajFromMap->measurements();
			int totalNumberOfHits = measurements.size();
			int numberOfHits = 0;
			double previousDotProduct = 0;
			for (trackingRecHit_iterator ith = trackFromMap->recHitsBegin(), edh = trackFromMap->recHitsEnd(); ith != edh; ++ith) {

				GlobalVector stateMomentum = measurements[numberOfHits].forwardPredictedState().globalMomentum();
				GlobalPoint statePosition = measurements[numberOfHits].forwardPredictedState().globalPosition();
				double dotProduct = stateMomentum.x()*statePosition.x() + stateMomentum.y()*statePosition.y();
				if ( dotProduct*previousDotProduct < 0 ){
					//found hit to split from...
					HITTOSPLITFROM = numberOfHits;
				}

				previousDotProduct = dotProduct;
				numberOfHits++;

			}
			LogDebug("CosmicTrackSplitter") << "number of rechits: " << numberOfHits;

			// check if the trajectories and rechits are in reverse order...
			trackingRecHit_iterator bIt = trackFromMap->recHitsBegin();
			trackingRecHit_iterator fIt = trackFromMap->recHitsEnd() - 1;
			const TrackingRecHit* bHit = bIt->get();
			const TrackingRecHit* fHit = fIt->get();
			// hit type valid = 0, missing = 1, inactive = 2, bad = 3
			if( bHit->type() != 0 || bHit->isValid() != 1){
				//loop over hits forwards until first Valid hit is found
				trackingRecHit_iterator ihit;
				for( ihit =  trackFromMap->recHitsBegin();
					ihit != trackFromMap->recHitsEnd(); ++ihit){
					const TrackingRecHit* iHit = ihit->get();
					if( iHit->type() == 0 && iHit->isValid() == 1){
						bHit = iHit;
						break;
					}
				}
			}
			DetId bdetid = bHit->geographicalId();
			GlobalPoint bPosHit = theGeometry->idToDetUnit( bdetid)->surface().
			toGlobal(bHit->localPosition());
			if( fHit->type() != 0 || fHit->isValid() != 1){
				//loop over hits backwards until first Valid hit is found
				trackingRecHit_iterator ihitf;
				for( ihitf =  trackFromMap->recHitsEnd()-1;
					ihitf != trackFromMap->recHitsBegin(); --ihitf){
					const TrackingRecHit* iHit = ihitf->get();
					if( iHit->type() == 0 && iHit->isValid() == 1){
						fHit = iHit;
						break;
					}
				}
			}
			DetId fdetid = fHit->geographicalId();
			GlobalPoint fPosHit =  theGeometry->
			idToDetUnit( fdetid )->surface().toGlobal(fHit->localPosition());
			GlobalPoint bPosState = measurements[0].updatedState().globalPosition();
			GlobalPoint fPosState = measurements[measurements.size()-1].
			updatedState().globalPosition();
			bool trajReversedFlag = false;
			/*
			DetId bdetid = bHit->geographicalId();
			DetId fdetid = fHit->geographicalId();
			GlobalPoint bPosHit =  theGeometry->idToDetUnit( bdetid )->surface().toGlobal(bHit->localPosition());
			GlobalPoint fPosHit =  theGeometry->idToDetUnit( fdetid )->surface().toGlobal(fHit->localPosition());
			GlobalPoint bPosState = measurements[0].updatedState().globalPosition();
			GlobalPoint fPosState = measurements[measurements.size() - 1].updatedState().globalPosition();
			bool trajReversedFlag = false;
			*/
			if (( (bPosHit - bPosState).mag() > (bPosHit - fPosState).mag() ) && ( (fPosHit - fPosState).mag() > (fPosHit - bPosState).mag() ) ){
				trajReversedFlag = true;
			}
			if (trajReversedFlag){ int temp = HITTOSPLITFROM; HITTOSPLITFROM = totalNumberOfHits - temp; }
		}

		totalTracks_ = totalTracks_ + tracks->size();
		// loop on tracks
		for (std::vector<reco::Track>::const_iterator itt = tracks->begin(), edt = tracks->end(); itt != edt; ++itt) {
			hits.clear(); // extra safety

			LogDebug("CosmicTrackSplitter") << "ntracks: " << tracks->size();

			// try to find distance of closest approach
			GlobalPoint v( itt->vx(), itt->vy(), itt->vz() );

			//checks on impact parameter
			bool continueWithTrack = true;
			if (fabs(v.z()) > dZcut_) continueWithTrack = false;
			if (v.perp() > dXYcut_) continueWithTrack = false;
			if (continueWithTrack == false) return;

			// LOOP TWICE, ONCE FOR TOP AND ONCE FOR BOTTOM
			for (int i = 0; i < 2; ++i){
				hits.clear(); // extra safety
				LogDebug("CosmicTrackSplitter") << "   loop on hits of track #" << (itt - tracks->begin());
				int usedHitCtr = 0;
				int hitCtr = 0;
				for (trackingRecHit_iterator ith = itt->recHitsBegin(), edh = itt->recHitsEnd(); ith != edh; ++ith) {
					//hitCtr++;
					const TrackingRecHit * hit = ith->get(); // ith is an iterator on edm::Ref to rechit
					LogDebug("CosmicTrackSplitter") << "         hit number " << (ith - itt->recHitsBegin());
					// let's look at valid hits
					if (hit->isValid()) {
						LogDebug("CosmicTrackSplitter") << "            valid, detid = " << hit->geographicalId().rawId();
						DetId detid = hit->geographicalId();

						if (detid.det() == DetId::Tracker) {  // check for tracker hits
							LogDebug("CosmicTrackSplitter") << "            valid, tracker ";
							bool  verdict = false;

							//trying to get the global position of the hit
							//const GeomDetUnit* geomDetUnit =  theGeometry->idToDetUnit( detid ).;

							const GlobalPoint pos =  theGeometry->idToDetUnit( detid )->surface().toGlobal(hit->localPosition());
							LogDebug("CosmicTrackSplitter") << "hit pos: " << pos << ", dca pos: " << v;

							// top half
							if ((i == 0)&&(hitCtr < HITTOSPLITFROM)){
								verdict = true;
								LogDebug("CosmicTrackSplitter") << "tophalf";
							}
							// bottom half
							if ((i == 1)&&(hitCtr >= HITTOSPLITFROM)){
								verdict = true;
								LogDebug("CosmicTrackSplitter") << "bottomhalf";
							}

							// if the hit is good, check again at module level
							if ( verdict  && std::binary_search(detsToIgnore_.begin(), detsToIgnore_.end(), detid.rawId())) {
								verdict = false;
							}

							// if hit is good check to make sure that we are keeping pixel hits
							if ( excludePixelHits_){
								if ((detid.det() == DetId::Tracker)&&((detid.subdetId() == 1)||(detid.subdetId() == 2))) {  // check for pixel hits
								 	verdict = false;
								}
							}

							LogDebug("CosmicTrackSplitter") << "                   verdict after module list: " << (verdict ? "ok" : "no");
							if (verdict == true) {
								// just copy the hit
								hits.push_back(hit->clone());
								usedHitCtr++;
							}
							else {
								// still, if replaceWithInactiveHits is true we have to put a new hit
								if (replaceWithInactiveHits_) {
									hits.push_back(new InvalidTrackingRecHit(*hit->det(), TrackingRecHit::inactive));
								}
							}
						}
						else { // just copy non tracker hits
							hits.push_back(hit->clone());
						}
					}
					else {
						if (!stripAllInvalidHits_) {
							hits.push_back(hit->clone());
						}
					} // is valid hit
					LogDebug("CosmicTrackSplitter") << "         end of hit " << (ith - itt->recHitsBegin());
					hitCtr++;
				} // loop on hits
				LogDebug("CosmicTrackSplitter") << "   end of loop on hits of track #" << (itt - tracks->begin());

				std::vector<TrackingRecHit *>::iterator begin = hits.begin(), end = hits.end();

				LogDebug("CosmicTrackSplitter") << "   selected " << hits.size() << " hits ";

				// strip invalid hits at the beginning
				if (stripFrontInvalidHits_) {
					while ( (begin != end) && ( (*begin)->isValid() == false ) ) ++begin;
				}

				LogDebug("CosmicTrackSplitter") << "   after front stripping we have " << (end - begin) << " hits ";

				// strip invalid hits at the end
				if (stripBackInvalidHits_ && (begin != end)) {
					--end;
					while ( (begin != end) && ( (*end)->isValid() == false ) ) --end;
					++end;
				}

				LogDebug("CosmicTrackSplitter") << "   after back stripping we have " << (end - begin) << " hits ";

				// if we still have some hits
				//if ((end - begin) >= int(minimumHits_)) {
				if ( usedHitCtr >= int(minimumHits_)) {
					output->push_back( makeCandidate ( *itt, begin, end ) );
					LogDebug("CosmicTrackSplitter") << "we made a candidate of " << hits.size() << " hits!";
				}
				// now delete the hits not used by the candidate
				for (begin = hits.begin(), end = hits.end(); begin != end; ++begin) {
					if (*begin) delete *begin;
				}
				LogDebug("CosmicTrackSplitter") << "loop: " << i << " has " << usedHitCtr << " active hits and " << hits.size() << " total hits...";
				hits.clear();
			} // loop twice for top and bottom
		} // loop on tracks
		LogDebug("CosmicTrackSplitter") << "totalTracks_ = " << totalTracks_;
		iEvent.put(output);
	}

	TrackCandidate
	CosmicTrackSplitter::makeCandidate(const reco::Track &tk, std::vector<TrackingRecHit *>::iterator hitsBegin, std::vector<TrackingRecHit *>::iterator hitsEnd) {

		LogDebug("CosmicTrackSplitter") << "Making a candidate!";


		PropagationDirection   pdir = tk.seedDirection();
		PTrajectoryStateOnDet state;
		if ( pdir == anyDirection ) throw cms::Exception("UnimplementedFeature") << "Cannot work with tracks that have 'anyDirecton' \n";
		//if ( (pdir == alongMomentum) == ( tk.p() >= tk.outerP() ) ) {
		if ( (pdir == alongMomentum) == (  (tk.outerPosition()-tk.innerPosition()).Dot(tk.momentum()) >= 0    ) ) {
			// use inner state
			TrajectoryStateOnSurface originalTsosIn(trajectoryStateTransform::innerStateOnSurface(tk, *theGeometry, &*theMagField));
			state = trajectoryStateTransform::persistentState( originalTsosIn, DetId(tk.innerDetId()) );
		} else {
			// use outer state
			TrajectoryStateOnSurface originalTsosOut(trajectoryStateTransform::outerStateOnSurface(tk, *theGeometry, &*theMagField));
			state = trajectoryStateTransform::persistentState( originalTsosOut, DetId(tk.outerDetId()) );
		}

		TrajectorySeed seed(state, TrackCandidate::RecHitContainer(), pdir);

		TrackCandidate::RecHitContainer ownHits;
		ownHits.reserve(hitsEnd - hitsBegin);
		for ( ; hitsBegin != hitsEnd; ++hitsBegin) { ownHits.push_back( *hitsBegin ); }

		TrackCandidate cand(ownHits, seed, state, tk.seedRef());


		LogDebug("CosmicTrackSplitter") << "   dumping the hits now: ";
		for (TrackCandidate::range hitR = cand.recHits(); hitR.first != hitR.second; ++hitR.first) {
		      LogTrace("CosmicTrackSplitter") << "     hit detid = " << hitR.first->geographicalId().rawId() <<
			", type  = " << typeid(*hitR.first).name();
		}

		return cand;
	}

}} //namespaces


// ========= MODULE DEF ==============
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using reco::modules::CosmicTrackSplitter;
DEFINE_FWK_MODULE(CosmicTrackSplitter);
