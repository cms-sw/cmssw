#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMapRecord.h"

#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer2.h"





#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"

//Propagator withMaterial
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
//analyticalpropagator
//#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

//

//for debug only 
//#define FAMOS_DEBUG

TrajectorySeedProducer2::TrajectorySeedProducer2(const edm::ParameterSet& conf):
    TrajectorySeedProducer(conf)
{  
} 

bool
TrajectorySeedProducer2::passSimTrackQualityCuts(const SimTrack& theSimTrack, const SimVertex& theSimVertex, unsigned int trackingAlgorithmId)
{

	//require min pT of the simtrack
	if ( theSimTrack.momentum().Perp2() < pTMin[trackingAlgorithmId] )
	{
		return false;
	}

	//require impact parameter of the simtrack
	BaseParticlePropagator theParticle = BaseParticlePropagator(
		RawParticle(
			XYZTLorentzVector(
				theSimTrack.momentum().px(),
				theSimTrack.momentum().py(),
				theSimTrack.momentum().pz(),
				theSimTrack.momentum().e()
			),
			XYZTLorentzVector(
				theSimVertex.position().x(),
				theSimVertex.position().y(),
				theSimVertex.position().z(),
				theSimVertex.position().t())
			),
			0.,0.,4.
	);
	theParticle.setCharge(theSimTrack.charge());
	if ( theParticle.xyImpactParameter(x0,y0) > maxD0[trackingAlgorithmId] )
	{
		return false;
	}
    if ( fabs( theParticle.zImpactParameter(x0,y0) - z0 ) > maxZ0[trackingAlgorithmId] )
	{
    	return false;
	}
    return true;
}

bool
TrajectorySeedProducer2::passTrackerRecHitQualityCuts(std::vector<TrackerRecHit>& trackerRecHits, std::vector<unsigned int> previousHits, TrackerRecHit& currentHit, unsigned int trackingAlgorithmId)
{
	//TODO: it seems that currently only a PV compatibility check for the first 2 hits is really important

	//return true;
	if (previousHits.size()==1)
	{
		TrackerRecHit& theSeedHits0 = trackerRecHits[previousHits[0]];
		TrackerRecHit& theSeedHits1 = currentHit;
		GlobalPoint gpos1 = theSeedHits0.globalPosition();
		GlobalPoint gpos2 = theSeedHits1.globalPosition();
		bool forward = theSeedHits0.isForward();
		double error = std::sqrt(theSeedHits0.largerError()+theSeedHits1.largerError());
		//	  compatible = compatibleWithVertex(gpos1,gpos2,ialgo);
		//added out of desperation
		bool compatible=false;
		if(seedingAlgo[trackingAlgorithmId] == "PixelLess" ||  seedingAlgo[trackingAlgorithmId] ==  "TobTecLayerPairs")
		{
			compatible = true;
		} else {
			compatible = compatibleWithBeamAxis(gpos1,gpos2,error,forward,trackingAlgorithmId);
			//if (!compatible) std::cout<<"reject beam axis"<<std::endl;
		}

		// Check if the pair is on the requested dets
		if ( numberOfHits[trackingAlgorithmId] == 2 )
		{

			if ( seedingAlgo[trackingAlgorithmId] ==  "ThirdMixedPairs" )
			{
				//may not be necessary anymore
				compatible = compatible && theSeedHits0.makesAPairWith3rd(theSeedHits1);
			}
			else
			{
				//may not be necessary anymore
				compatible = compatible && theSeedHits0.makesAPairWith(theSeedHits1);
			}
		}
		return compatible;
	}
	else if (previousHits.size()==2)
	{
		TrackerRecHit& theSeedHits0 = trackerRecHits[previousHits[0]];
		TrackerRecHit& theSeedHits1 = trackerRecHits[previousHits[1]];
		TrackerRecHit& theSeedHits2 = currentHit;
		//may not be necessary anymore
		return theSeedHits0.makesATripletWith(theSeedHits1,theSeedHits2);
	}
	return true;
}

int TrajectorySeedProducer2::iterateHits(
		SiTrackerGSMatchedRecHit2DCollection::const_iterator start,
		SiTrackerGSMatchedRecHit2DCollection::range range,
		std::vector<std::vector<unsigned int>> hitNumbers,
		std::vector<TrackerRecHit>& trackerRecHits,
		unsigned int trackingAlgorithmId,
		std::vector<unsigned int>& seedHitNumbers
	)
{


	bool nextHitOnSameLayer=false;
	bool thisHitOnSameLayer=false;
	for (SiTrackerGSMatchedRecHit2DCollection::const_iterator itRecHit = start; itRecHit!=range.second; ++itRecHit)
	{
	    unsigned int currentHitIndex = itRecHit-range.first;

		//trackerRecHits.size() is 'absMinRecHits' + including hits on the same layer
		if ( currentHitIndex >= trackerRecHits.size())
		{
			return -1;
		}
		TrackerRecHit& currentTrackerHit = trackerRecHits[currentHitIndex];

		thisHitOnSameLayer=nextHitOnSameLayer;
		if (currentHitIndex+1 <= trackerRecHits.size())
		{
			nextHitOnSameLayer=trackerRecHits[currentHitIndex].isOnTheSameLayer(trackerRecHits[currentHitIndex+1]);
		}
		if (nextHitOnSameLayer)
		{
			//branch here to process an alternative seeding hypothesis using the next hit which will be skip by the main loop
			int result=TrajectorySeedProducer2::iterateHits(
					itRecHit+1,
					range,
					hitNumbers,
					trackerRecHits,
					trackingAlgorithmId,
					seedHitNumbers
				);
			if (result>=0)
			{
				return result;
			}
		}

		//skip this hit if it was already processed in previous iteration
		if (!thisHitOnSameLayer)
		{
			for (unsigned int ilayerset=0; ilayerset<theLayersInSets.size(); ++ ilayerset)
			{
				unsigned int currentlayer = hitNumbers[ilayerset].size();

				//speed things up by testing if the current hit is already further outside than the current layer
				//-> faster rejection of invalid hits if there is no seed possible at all for a given simtrack
				if (theLayersInSets[ilayerset][currentlayer].subDet==currentTrackerHit.subDetId() && theLayersInSets[ilayerset][currentlayer].idLayer==currentTrackerHit.layerNumber())
				{
					if (this->passTrackerRecHitQualityCuts(trackerRecHits, hitNumbers[ilayerset], currentTrackerHit, trackingAlgorithmId))
					{
						hitNumbers[ilayerset].push_back(itRecHit-range.first);

						//seed found!
						if (theLayersInSets[ilayerset].size()<=hitNumbers[ilayerset].size())
						{
							for (unsigned int j=0; j<hitNumbers[ilayerset].size(); ++ j)
							{
								//save result
								seedHitNumbers.push_back(hitNumbers[ilayerset][j]);
							}
							return ilayerset;
						}
					}
				}
			}
		}
	}
	return -1;
}

void 
TrajectorySeedProducer2::produce(edm::Event& e, const edm::EventSetup& es) {        


  //  if( seedingAlgo[0] ==  "FourthPixelLessPairs") std::cout << "Seed producer in 4th iteration " << std::endl;

#ifdef FAMOS_DEBUG
  std::cout << "################################################################" << std::endl;
  std::cout << " TrajectorySeedProducer produce init " << std::endl;
#endif

	//Retrieve tracker topology from geometry
	edm::ESHandle<TrackerTopology> tTopoHand;
	es.get<IdealGeometryRecord>().get(tTopoHand);
	const TrackerTopology *tTopo=tTopoHand.product();


	//  unsigned nTrackCandidates = 0;
	PTrajectoryStateOnDet initialState;

	// Output
	std::vector<TrajectorySeedCollection*> output(seedingAlgo.size());
	for ( unsigned ialgo=0; ialgo<seedingAlgo.size(); ++ialgo )
	{
		//TODO: is this really destroyed?
		output[ialgo] = new TrajectorySeedCollection();
	}

	// Beam spot
	edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
	e.getByLabel(theBeamSpot,recoBeamSpotHandle);
	math::XYZPoint BSPosition_ = recoBeamSpotHandle->position();

	//not used anymore. take the value from the py

	//double sigmaZ=recoBeamSpotHandle->sigmaZ();
	//double sigmaZ0Error=recoBeamSpotHandle->sigmaZ0Error();
	//double sigmaz0=sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
	x0 = BSPosition_.X();
	y0 = BSPosition_.Y();
	z0 = BSPosition_.Z();

	// SimTracks and SimVertices
	edm::Handle<edm::SimTrackContainer> theSimTracks;
	e.getByLabel("famosSimHits",theSimTracks);

	edm::Handle<edm::SimVertexContainer> theSimVtx;
	e.getByLabel("famosSimHits",theSimVtx);
  
	//  edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
	edm::Handle<SiTrackerGSMatchedRecHit2DCollection> theGSRecHits;
	e.getByLabel(hitProducer, theGSRecHits);

	//std::cout<<"event contains: "<<theSimTracks->size()<<" simtracks"<<std::endl;
	//std::cout<<"event contains: "<<theGSRecHits->ids().size()<<" simtracks associated to hits"<<std::endl;
	//std::cout<<"event contains: "<<theGSRecHits->size()<<" hits"<<std::endl;

	//if no hits -> directly write empty collection
	if(theGSRecHits->size() == 0)
	{
	
		for ( unsigned ialgo=0; ialgo<seedingAlgo.size(); ++ialgo )
		{
			std::auto_ptr<TrajectorySeedCollection> p(output[ialgo]);
			e.put(p,seedingAlgo[ialgo]);
		}
		
		return;
		
	}

	  // Primary vertices
	vertices = std::vector<const reco::VertexCollection*>(seedingAlgo.size(),static_cast<const reco::VertexCollection*>(0));
	for ( unsigned ialgo=0; ialgo<seedingAlgo.size(); ++ialgo )
	{
	    edm::Handle<reco::VertexCollection> aHandle;
	    bool isVertexCollection = e.getByLabel(primaryVertices[ialgo],aHandle);
	    if (!isVertexCollection ) continue;
	    vertices[ialgo] = &(*aHandle);
	}

	for (SiTrackerGSMatchedRecHit2DCollection::id_iterator itSimTrackId=theGSRecHits->id_begin();  itSimTrackId!=theGSRecHits->id_end(); ++itSimTrackId )
	{
		const unsigned int currentSimTrackId = *itSimTrackId;
		//if (currentSimTrackId!=628) continue;
		//std::cout<<"processing simtrack with id: "<<currentSimTrackId<<std::endl;
		const SimTrack& theSimTrack = (*theSimTracks)[currentSimTrackId];

		int vertexIndex = theSimTrack.vertIndex();
		if (vertexIndex<0)
		{
			//tracks are required to be associated to a vertex
			continue;
		}
		const SimVertex& theSimVertex = (*theSimVtx)[vertexIndex];

		for ( unsigned int ialgo = 0; ialgo < seedingAlgo.size(); ++ialgo )
		{
			if (!this->passSimTrackQualityCuts(theSimTrack,theSimVertex,ialgo))
			{
				continue;
			}
			SiTrackerGSMatchedRecHit2DCollection::range recHitRange = theGSRecHits->get(currentSimTrackId);
			//std::cout<<"\ttotal produced: "<<recHitRange.second-recHitRange.first<<" hits"<<std::endl;

			TrackerRecHit previousTrackerHit;
			TrackerRecHit currentTrackerHit;

			std::vector<std::vector<unsigned int>> hitNumbers;
			hitNumbers.resize(theLayersInSets.size());

			//TODO: store the created valid hits here - later create them from the numbers above as needed
			// -> saves the unnecessary creation of the objects
			std::vector<std::vector<TrackerRecHit>> seedHitCandiates;
			seedHitCandiates.resize(theLayersInSets.size());

			//Just for debugging
			std::vector<std::vector<std::string>> layerNames;
			layerNames.resize(theLayersInSets.size());


			unsigned int numberOfNonEqualHits=0;

			//store the converted objects
			std::vector<TrackerRecHit> trackerRecHits;

			for (SiTrackerGSMatchedRecHit2DCollection::const_iterator itRecHit = recHitRange.first; itRecHit!=recHitRange.second; ++itRecHit)
			{
				const SiTrackerGSMatchedRecHit2D& vec = *itRecHit;
				previousTrackerHit=currentTrackerHit;
				currentTrackerHit = TrackerRecHit(&vec,theGeometry,tTopo);
				trackerRecHits.push_back(currentTrackerHit);
				if (currentTrackerHit.isOnTheSameLayer(previousTrackerHit))
				{
					continue;
				}
				++numberOfNonEqualHits;
				if ( numberOfNonEqualHits > absMinRecHits ) break;
			}
			if ( numberOfNonEqualHits < minRecHits[ialgo] ) continue;


			std::vector<unsigned int> seedHitNumbers;
			int seedLayerSetIndex = iterateHits(recHitRange.first,recHitRange,hitNumbers,trackerRecHits, ialgo,seedHitNumbers);


			if (seedLayerSetIndex>=0)
			{

				edm::OwnVector<TrackingRecHit> recHits;
				for ( unsigned ihit=0; ihit<seedHitNumbers.size(); ++ihit )
				{
					TrackingRecHit* aTrackingRecHit = trackerRecHits[seedHitNumbers[ihit]].hit()->clone();
					recHits.push_back(aTrackingRecHit);
				}


				GlobalPoint  position((*theSimVtx)[vertexIndex].position().x(),
				(*theSimVtx)[vertexIndex].position().y(),
				(*theSimVtx)[vertexIndex].position().z());

				GlobalVector momentum(theSimTrack.momentum().x(),theSimTrack.momentum().y(),theSimTrack.momentum().z());
				float charge = theSimTrack.charge();
				GlobalTrajectoryParameters initialParams(position,momentum,(int)charge,theMagField);
				AlgebraicSymMatrix55 errorMatrix= AlgebraicMatrixID();
				//this line help the fit succeed in the case of pixelless tracks (4th and 5th iteration)
				//for the future: probably the best thing is to use the mini-kalmanFilter
				if(trackerRecHits[seedHitNumbers[0]].subDetId() !=1 ||trackerRecHits[seedHitNumbers[0]].subDetId() !=2)
				{
					errorMatrix = errorMatrix * 0.0000001;
				}
				CurvilinearTrajectoryError initialError(errorMatrix);
				FreeTrajectoryState initialFTS(initialParams, initialError);
				const GeomDet* initialLayer = theGeometry->idToDet( recHits.front().geographicalId() );
				const TrajectoryStateOnSurface initialTSOS = thePropagator->propagate(initialFTS,initialLayer->surface()) ;


				if (!initialTSOS.isValid())
				{
					break; //continues with the next seeding algorithm
				}
				
				const AlgebraicSymMatrix55& m = initialTSOS.localError().matrix();
				int dim = 5; /// should check if corresponds to m
				float localErrors[15];
				int k = 0;
				for (int i=0; i<dim; ++i)
				{
					for (int j=0; j<=i; ++j)
					{
						localErrors[k++] = m(i,j);
					}
				}
				int surfaceSide = static_cast<int>(initialTSOS.surfaceSide());
				initialState = PTrajectoryStateOnDet( initialTSOS.localParameters(),localErrors, recHits.front().geographicalId().rawId(), surfaceSide);
				output[ialgo]->push_back(TrajectorySeed(initialState, recHits, PropagationDirection::alongMomentum));
			}



		} //end loop over seeding algorithms
	} //end loop over simtracks

    
	for ( unsigned ialgo=0; ialgo<seedingAlgo.size(); ++ialgo )
	{
		std::auto_ptr<TrajectorySeedCollection> p(output[ialgo]);
		e.put(p,seedingAlgo[ialgo]);
	}
  
}


