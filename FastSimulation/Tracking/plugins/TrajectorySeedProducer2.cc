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
			//std::cout<<"break absHits"<<std::endl;
			return -1;
		}
		TrackerRecHit& currentTrackerHit = trackerRecHits[currentHitIndex];
		//std::cout<<"\thit="<<currentHitIndex<<", subId="<<currentTrackerHit.subDetId()<<", layer="<<currentTrackerHit.layerNumber()<<", globalX="<<currentTrackerHit.globalPosition().x()<<std::endl;


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


		//skip this hit because in previous iteration this hit was already processed
		if (thisHitOnSameLayer)
		{
			//std::cout<<"hit on same layer"<<std::endl;
		}
		else
		{
			for (unsigned int ilayerset=0; ilayerset<theLayersInSets.size(); ++ ilayerset)
			{
				unsigned int currentlayer = hitNumbers[ilayerset].size();

				//TODO: speed things up by testing if the current hit is already further outside than the current layer
				//-> faster rejection of invalid hits if there is no seed possible at all for a given simtrack
				if (theLayersInSets[ilayerset][currentlayer].subDet==currentTrackerHit.subDetId() && theLayersInSets[ilayerset][currentlayer].idLayer==currentTrackerHit.layerNumber())
				{
					if (this->passTrackerRecHitQualityCuts(trackerRecHits, hitNumbers[ilayerset], currentTrackerHit, trackingAlgorithmId))
					{
						hitNumbers[ilayerset].push_back(itRecHit-range.first);
						//std::cout<<"hit accepted: layerset="<<ilayerset<<std::endl;

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

	std::vector<TrajectorySeedCollection*> output_new(seedingAlgo.size());
	for ( unsigned ialgo=0; ialgo<seedingAlgo.size(); ++ialgo )
	{
		output_new[ialgo] = new TrajectorySeedCollection();
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

	//for comparison
	std::vector<std::vector<int>> seedHit_new;
	seedHit_new.resize(theSimTracks->size());
	std::vector<std::vector<int>> seedHit_old;
	seedHit_old.resize(theSimTracks->size());

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

			/*
			if (seedLayerSetIndex>=0)
			{


				std::cout<<"produce seed for ialgo="<<ialgo<<", simtrackid="<<currentSimTrackId<<", #recHits=";
				for (unsigned int j=0; j<seedHitNumbers.size();++j)
				{
					std::cout<<seedHitNumbers[j]<<",";
				}
				std::cout<<std::endl;

				for (unsigned int j=0; j<trackerRecHits.size();++j)
				{
					std::cout<<"hit: "<<j<<", subId="<<trackerRecHits[j].subDetId()<<", layer="<<trackerRecHits[j].layerNumber()<<", globalX="<<trackerRecHits[j].globalPosition().x()<<std::endl;
				}
			}*/


			if (seedLayerSetIndex>=0)
			{

				edm::OwnVector<TrackingRecHit> recHits;
				for ( unsigned ihit=0; ihit<seedHitNumbers.size(); ++ihit )
				{
					TrackingRecHit* aTrackingRecHit = trackerRecHits[seedHitNumbers[ihit]].hit()->clone();
					recHits.push_back(aTrackingRecHit);
				}

				//TODO: the following is keep as it is form the previous implementation - check if all of it is needed
				// 2) Create the initial state
				//   a) origin vertex
				GlobalPoint  position((*theSimVtx)[vertexIndex].position().x(),
				(*theSimVtx)[vertexIndex].position().y(),
				(*theSimVtx)[vertexIndex].position().z());

				//   b) initial momentum
				GlobalVector momentum(theSimTrack.momentum().x(),theSimTrack.momentum().y(),theSimTrack.momentum().z());
				//   c) electric charge
				float charge = theSimTrack.charge();
				//  -> inital parameters
				GlobalTrajectoryParameters initialParams(position,momentum,(int)charge,theMagField);
				//  -> large initial errors
				AlgebraicSymMatrix55 errorMatrix= AlgebraicMatrixID();
				// errorMatrix = errorMatrix * 10;

				//this line help the fit succeed in the case of pixelless tracks (4th and 5th iteration)
				//for the future: probably the best thing is to use the mini-kalmanFilter
				if(trackerRecHits[seedHitNumbers[0]].subDetId() !=1 ||trackerRecHits[seedHitNumbers[0]].subDetId() !=2)
				{
					errorMatrix = errorMatrix * 0.0000001;
				}
				CurvilinearTrajectoryError initialError(errorMatrix);
				// -> initial state
				FreeTrajectoryState initialFTS(initialParams, initialError);
				// const GeomDetUnit* initialLayer = theGeometry->idToDetUnit( recHits.front().geographicalId() );
				const GeomDet* initialLayer = theGeometry->idToDet( recHits.front().geographicalId() );

				//this is wrong because the FTS is defined at vertex, and it need to be properly propagated.
				//      const TrajectoryStateOnSurface initialTSOS(initialFTS, initialLayer->surface());
				const TrajectoryStateOnSurface initialTSOS = thePropagator->propagate(initialFTS,initialLayer->surface()) ;


				if (!initialTSOS.isValid())
				{
					//std::cout<<"tsos rejected"<<std::endl;
					break; //continues with the next seeding algorithm
				}

				//std::cout<<"produce seed for ialgo="<<ialgo<<", simtrackid="<<currentID<<", #recHits=";
				/*
				for (unsigned int i=0;i<seedHitNumbers.size();++i)
				{
					seedHit_new[currentSimTrackId].push_back(seedHitNumbers[i]);
					//std::cout<<hitNumbers[seedLayerSet][i]<<",";
				}
				*/
				//=std::cout<<std::endl;
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
				output_new[ialgo]->push_back(TrajectorySeed(initialState, recHits, PropagationDirection::alongMomentum));
			}



		} //end loop over seeding algorithms
	} //end loop over simtracks


	for ( unsigned ialgo=0; ialgo<seedingAlgo.size(); ++ialgo )
	{
		std::auto_ptr<TrajectorySeedCollection> p(output[ialgo]);
		e.put(p,seedingAlgo[ialgo]);
	}

/*
	std::cout<<std::endl;
    std::cout<<"old result:"<<std::endl;

  
#ifdef FAMOS_DEBUG
  std::cout << " Step C: Loop over the RecHits, track by track " << std::endl;
#endif

  // The vector of simTrack Id's carrying GSRecHits
  const std::vector<unsigned> theSimTrackIds = theGSRecHits->ids();

  // loop over SimTrack Id's
  for ( unsigned tkId=0;  tkId != theSimTrackIds.size(); ++tkId ) {




    ++nSimTracks;
    unsigned simTrackId = theSimTrackIds[tkId];
    //if (simTrackId!=628) continue;
    std::cout<<"processing simtrack with id: "<<theSimTrackIds[tkId]<<std::endl;
    const SimTrack& theSimTrack = (*theSimTracks)[simTrackId];
#ifdef FAMOS_DEBUG
    std::cout << "Track number " << simTrackId << "--------------------------------" <<std::endl;
#endif
#ifdef FAMOS_DEBUG
    std::cout << "Pt = " << std::sqrt(theSimTrack.momentum().Perp2()) 
	      << " eta " << theSimTrack.momentum().Eta()
	      << " pdg ID " << theSimTrack.type()
	      << std::endl;
#endif

    // Select only muons, if requested
    if (selectMuons && abs(theSimTrack.type()) != 13) continue;
    
    // Check that the sim track comes from the main vertex (loose cut)
    int vertexIndex = theSimTrack.vertIndex();
    const SimVertex& theSimVertex = (*theSimVtx)[vertexIndex]; 
#ifdef FAMOS_DEBUG
    std::cout << " o SimTrack " << theSimTrack << std::endl;
    std::cout << " o SimVertex " << theSimVertex << std::endl;
#endif
    
    BaseParticlePropagator theParticle = 
      BaseParticlePropagator( 
	 RawParticle(XYZTLorentzVector(theSimTrack.momentum().px(),
				       theSimTrack.momentum().py(),
				       theSimTrack.momentum().pz(),
				       theSimTrack.momentum().e()),
		     XYZTLorentzVector(theSimVertex.position().x(),
				       theSimVertex.position().y(),
				       theSimVertex.position().z(),
				       theSimVertex.position().t())),
	             0.,0.,4.);
    theParticle.setCharge((*theSimTracks)[simTrackId].charge());

    SiTrackerGSMatchedRecHit2DCollection::range theRecHitRange = theGSRecHits->get(simTrackId);
    SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
    SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
    SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit;
    SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit1;
    SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit2;
    SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit3;

    unsigned int hit1,hit2,hit3 =0;


    // Check the number of layers crossed
    unsigned numberOfRecHits = 0;
    TrackerRecHit previousHit, currentHit;
    for ( iterRecHit = theRecHitRangeIteratorBegin; iterRecHit != theRecHitRangeIteratorEnd; ++iterRecHit) {
      previousHit = currentHit;
      currentHit = TrackerRecHit(&(*iterRecHit),theGeometry,tTopo);
      if ( currentHit.isOnTheSameLayer(previousHit) ) continue;
      ++numberOfRecHits;
      if ( numberOfRecHits == absMinRecHits ) break;
    }

    // Loop on the successive seedings
    for ( unsigned int ialgo = 0; ialgo < seedingAlgo.size(); ++ialgo ) {
#ifdef FAMOS_DEBUG
      std::cout << "Algo " << seedingAlgo[ialgo] << std::endl;
#endif

      // Request a minimum number of RecHits for the track to give a seed.
#ifdef FAMOS_DEBUG
      std::cout << "The number of RecHits = " << numberOfRecHits << std::endl;
#endif
      if ( numberOfRecHits < minRecHits[ialgo] ) continue;
      ++nTracksWithHits;

      // Request a minimum pT for the sim track
      if ( theSimTrack.momentum().Perp2() < pTMin[ialgo] ) continue;
      ++nTracksWithPT;
      
      // Cut on sim track impact parameters
      if ( theParticle.xyImpactParameter(x0,y0) > maxD0[ialgo] ) continue;
      if ( fabs( theParticle.zImpactParameter(x0,y0) - z0 ) > maxZ0[ialgo] ) continue;
      ++nTracksWithD0Z0;
      
      std::vector<TrackerRecHit > theSeedHits(numberOfHits[ialgo],static_cast<TrackerRecHit >(TrackerRecHit()));
      TrackerRecHit& theSeedHits0 = theSeedHits[0];
      TrackerRecHit& theSeedHits1 = theSeedHits[1];
      TrackerRecHit& theSeedHits2 = theSeedHits[2];
      
      bool compatible = false;
      
      for ( iterRecHit1 = theRecHitRangeIteratorBegin; iterRecHit1 != theRecHitRangeIteratorEnd; ++iterRecHit1) {
    	  hit1=iterRecHit1-theRecHitRangeIteratorBegin;
	theSeedHits[0] = TrackerRecHit(&(*iterRecHit1),theGeometry,tTopo);
	//std::cout<<"old hit 1 - "<<hit1<<" subDet="<<theSeedHits0.subDetId()<<", layer="<<theSeedHits0.layerNumber()<<", globalX="<<theSeedHits0.globalPosition().x()<<std::endl;

#ifdef FAMOS_DEBUG
	std::cout << "The first hit position = " << theSeedHits0.globalPosition() << std::endl;
	std::cout << "The first hit subDetId = " << theSeedHits0.subDetId() << std::endl;
	std::cout << "The first hit layer    = " << theSeedHits0.layerNumber() << std::endl;
#endif  
	// Check if inside the requested detectors
	bool isInside = true;
	if (!selectMuons) {
    //(newSyntax) ? std::cout << "TRUE " : std::cout << "FALSE "; //J-
	  if (newSyntax)
	    isInside = false; // AG placeholder true 
	  else
	    isInside = theSeedHits0.subDetId() < firstHitSubDetectors[ialgo][0];
	  //	bool isInside = theSeedHits0.subDetId() < firstHitSubDetectors[ialgo][0];
	  if ( isInside ) continue;
	}
	// Check if on requested detectors
	// bool isOndet =  theSeedHits0.isOnRequestedDet(firstHitSubDetectors[ialgo]);
	bool isOndet = true;
	if (!selectMuons) {
	  if (newSyntax)
	  {
      isOndet = theSeedHits[0].isOnRequestedDet(theLayersInSets);
	  //std::cout<<hit1<<" = "<<isOndet<<" (selected hit 1)"<<std::endl;
	  }
	  else
	    isOndet = theSeedHits0.isOnRequestedDet(firstHitSubDetectors[ialgo], seedingAlgo[ialgo]);
      //std::cout << firstHitSubDetectors[ialgo][0] << " | " << seedingAlgo[ialgo] << " " << std::endl;  //seedingAlgo[iAlgo]: PixelTriplet, LowPtPixelTriplets, PixelPair, DetachedPixelTriplets, MixedTriplets, PixelLessPairs, TobTecLayerPairs......
      //	bool isOndet =  theSeedHits0.isOnRequestedDet(firstHitSubDetectors[ialgo], seedingAlgo[ialgo]);
      //	if ( !isOndet ) break;
	  if ( !isOndet ) continue;
	}
#ifdef FAMOS_DEBUG
	std::cout << "Apparently the first hit is on the requested detector! " << std::endl;
#endif
	for ( iterRecHit2 = iterRecHit1+1; iterRecHit2 != theRecHitRangeIteratorEnd; ++iterRecHit2) {
		hit2=iterRecHit2-theRecHitRangeIteratorBegin;
		theSeedHits[1] = TrackerRecHit(&(*iterRecHit2),theGeometry,tTopo);
		//std::cout<<"old hit 2 - "<<hit2<<" subDet="<<theSeedHits1.subDetId()<<", layer="<<theSeedHits1.layerNumber()<<", globalX="<<theSeedHits1.globalPosition().x()<<std::endl;
#ifdef FAMOS_DEBUG
	  std::cout << "The second hit position = " << theSeedHits1.globalPosition() << std::endl;
	  std::cout << "The second hit subDetId = " << theSeedHits1.subDetId() << std::endl;
	  std::cout << "The second hit layer    = " << theSeedHits1.layerNumber() << std::endl;
#endif

	  if (!selectMuons) {
	    // Check if inside the requested detectors
	    if (newSyntax) 
	      isInside = false; // AG placeholder true
	    else
	      isInside = theSeedHits1.subDetId() < secondHitSubDetectors[ialgo][0];
	    if ( isInside ) continue;
	    // Check if on requested detectors
	    if (newSyntax)
	    {
        isOndet = theSeedHits[0].isOnRequestedDet(theLayersInSets, theSeedHits[1]);
	    //std::cout<<hit2<<" = "<<isOndet<<" (selected hit 2)"<<std::endl;
	    }
	    else
	      isOndet =  theSeedHits1.isOnRequestedDet(secondHitSubDetectors[ialgo], seedingAlgo[ialgo]);
	    if ( !isOndet ) continue;
	  }
	  // Check if on the same layer as previous hit
	  if ( theSeedHits1.isOnTheSameLayer(theSeedHits0) ) continue;
#ifdef FAMOS_DEBUG
	  std::cout << "Apparently the second hit is on the requested detector! " << std::endl;
#endif
	  GlobalPoint gpos1 = theSeedHits0.globalPosition();
	  GlobalPoint gpos2 = theSeedHits1.globalPosition();
	  bool forward = theSeedHits0.isForward();
	  double error = std::sqrt(theSeedHits0.largerError()+theSeedHits1.largerError());
	  //	  compatible = compatibleWithVertex(gpos1,gpos2,ialgo);
	  //added out of desperation	
	  if(seedingAlgo[ialgo] == "PixelLess" ||  seedingAlgo[ialgo] ==  "TobTecLayerPairs"){
	    compatible = true;
	    //std::cout << "Algo " << seedingAlgo[0] << "Det/layer = " << theSeedHits0.subDetId() << "/" <<  theSeedHits0.layerNumber() << std::endl;
	  } else {
	    compatible = compatibleWithBeamAxis(gpos1,gpos2,error,forward,ialgo);
	  }

#ifdef FAMOS_DEBUG
	  std::cout << "Algo" << seedingAlgo[0] << "\t Are the two hits compatible with the PV? " << compatible << std::endl;
#endif

	  if (!selectMuons) {
	    // Check if the pair is on the requested dets
	    if ( numberOfHits[ialgo] == 2 ) {
	      
	      if ( seedingAlgo[ialgo] ==  "ThirdMixedPairs" ){
		compatible = compatible && theSeedHits[0].makesAPairWith3rd(theSeedHits[1]);
	      } else {
		compatible = compatible && theSeedHits[0].makesAPairWith(theSeedHits[1]);
		//check

	      }
	    }	
	  }    
	  
	  // Reject non suited pairs
	  if ( !compatible ) continue;

#ifdef FAMOS_DEBUG
	  std::cout << "Pair kept! " << std::endl;
#endif

	  // Leave here if only two hits are required.
	  if ( numberOfHits[ialgo] == 2 ) break; 
	  
	  compatible = false;
	  // Check if there is a third satisfying hit otherwise
	  for ( iterRecHit3 = iterRecHit2+1; iterRecHit3 != theRecHitRangeIteratorEnd; ++iterRecHit3) {
		hit3=iterRecHit3-theRecHitRangeIteratorBegin;
	    theSeedHits[2] = TrackerRecHit(&(*iterRecHit3),theGeometry,tTopo);
#ifdef FAMOS_DEBUG
	    std::cout << "The third hit position = " << theSeedHits2.globalPosition() << std::endl;
	    std::cout << "The third hit subDetId = " << theSeedHits2.subDetId() << std::endl;
	    std::cout << "The third hit layer    = " << theSeedHits2.layerNumber() << std::endl;
#endif

	    if (!selectMuons) {
	      // Check if inside the requested detectors
	      if (newSyntax) 
          isInside = false; // AG placeholder
	      else 
          isInside = theSeedHits2.subDetId() < thirdHitSubDetectors[ialgo][0];
	      if ( isInside ) continue;
	    
	      // Check if on requested detectors
	      if (newSyntax) 
	      {
          isOndet = theSeedHits[0].isOnRequestedDet(theLayersInSets, theSeedHits[1], theSeedHits[2]);
	      }
	      else 
          isOndet =  theSeedHits2.isOnRequestedDet(thirdHitSubDetectors[ialgo], seedingAlgo[ialgo]);
	      //	    if ( !isOndet ) break;
	      if ( !isOndet ) continue;

	    }

	    // Check if on the same layer as previous hit
	    compatible = !(theSeedHits2.isOnTheSameLayer(theSeedHits1));

	    // Check if the triplet is on the requested det combination
	    if (!selectMuons) compatible = compatible && theSeedHits[0].makesATripletWith(theSeedHits[1],theSeedHits[2]); //J- maybe it's not necessary, as newSyntax layerlist is already checking?

#ifdef FAMOS_DEBUG
	    if ( compatible ) 
	      std::cout << "Apparently the third hit is on the requested detector! " << std::endl;
#endif

	    if ( compatible ) break;	  

	  }

	  if ( compatible ) break;

	}

	if ( compatible ) break;
  
      }

      // There is no compatible seed for this track with this seeding algorithm 
      // Go to next algo
      if ( !compatible ) continue;
#ifdef FAMOS_DEBUG
      if ( compatible ) 
	std::cout << "@@@ There is at least a compatible seed" << std::endl;
      else
	std::cout << "@@@ There is no compatible seed" << std::endl;
#endif
      

#ifdef FAMOS_DEBUG
      std::cout << "Preparing to create the TrajectorySeed" << std::endl;
#endif
      // The seed is validated -> include in the collection
      // 1) Create the vector of RecHits
      edm::OwnVector<TrackingRecHit> recHits;
      for ( unsigned ih=0; ih<theSeedHits.size(); ++ih ) {
	TrackingRecHit* aTrackingRecHit = theSeedHits[ih].hit()->clone();
	recHits.push_back(aTrackingRecHit);
      }
#ifdef FAMOS_DEBUG
      std::cout << "with " << recHits.size() << " hits." << std::endl;
#endif

      // 2) Create the initial state
      //   a) origin vertex
      GlobalPoint  position((*theSimVtx)[vertexIndex].position().x(),
			    (*theSimVtx)[vertexIndex].position().y(),
			    (*theSimVtx)[vertexIndex].position().z());
      
      //   b) initial momentum
      GlobalVector momentum( (*theSimTracks)[simTrackId].momentum().x() , 
			     (*theSimTracks)[simTrackId].momentum().y() , 
			     (*theSimTracks)[simTrackId].momentum().z() );
      //   c) electric charge
      float        charge   = (*theSimTracks)[simTrackId].charge();
      //  -> inital parameters
      GlobalTrajectoryParameters initialParams(position,momentum,(int)charge,theMagField);
      //  -> large initial errors
      AlgebraicSymMatrix55 errorMatrix= AlgebraicMatrixID();      
      // errorMatrix = errorMatrix * 10;

      //this line help the fit succeed in the case of pixelless tracks (4th and 5th iteration)
      //for the future: probably the best thing is to use the mini-kalmanFilter
      if(theSeedHits0.subDetId() !=1 || theSeedHits0.subDetId() !=2) errorMatrix = errorMatrix * 0.0000001;



#ifdef FAMOS_DEBUG
      std::cout << "TrajectorySeedProducer: SimTrack parameters " << std::endl;
      std::cout << "\t\t pT  = " << (*theSimTracks)[simTrackId].momentum().Pt() << std::endl;
      std::cout << "\t\t eta = " << (*theSimTracks)[simTrackId].momentum().Eta()  << std::endl;
      std::cout << "\t\t phi = " << (*theSimTracks)[simTrackId].momentum().Phi()  << std::endl;
      std::cout << "TrajectorySeedProducer: AlgebraicSymMatrix " << errorMatrix << std::endl;
#endif
      CurvilinearTrajectoryError initialError(errorMatrix);
      // -> initial state
      FreeTrajectoryState initialFTS(initialParams, initialError);      
#ifdef FAMOS_DEBUG
      std::cout << "TrajectorySeedProducer: FTS momentum " << initialFTS.momentum() << std::endl;
#endif
      // const GeomDetUnit* initialLayer = theGeometry->idToDetUnit( recHits.front().geographicalId() );
      const GeomDet* initialLayer = theGeometry->idToDet( recHits.front().geographicalId() );

      //this is wrong because the FTS is defined at vertex, and it need to be properly propagated.
      //      const TrajectoryStateOnSurface initialTSOS(initialFTS, initialLayer->surface());      

      const TrajectoryStateOnSurface initialTSOS = thePropagator->propagate(initialFTS,initialLayer->surface()) ;
      if (!initialTSOS.isValid())
    	  {
    	  //std::cout<<"tsos rejected"<<std::endl;
    	  continue;
    	  }

#ifdef FAMOS_DEBUG
      std::cout << "TrajectorySeedProducer: TSOS global momentum "    << initialTSOS.globalMomentum() << std::endl;
      std::cout << "\t\t\tpT = "                                      << initialTSOS.globalMomentum().perp() << std::endl;
      std::cout << "\t\t\teta = "                                     << initialTSOS.globalMomentum().eta() << std::endl;
      std::cout << "\t\t\tphi = "                                     << initialTSOS.globalMomentum().phi() << std::endl;
      std::cout << "TrajectorySeedProducer: TSOS local momentum "     << initialTSOS.localMomentum()  << std::endl;
      std::cout << "TrajectorySeedProducer: TSOS local error "        << initialTSOS.localError().positionError() << std::endl;
      std::cout << "TrajectorySeedProducer: TSOS local error matrix " << initialTSOS.localError().matrix() << std::endl;
      std::cout << "TrajectorySeedProducer: TSOS surface side "       << initialTSOS.surfaceSide()    << std::endl;
#endif
      stateOnDet(initialTSOS, 
		 recHits.front().geographicalId().rawId(),
		 initialState);
      // Create a new Trajectory Seed    


      std::cout<<"produce seed for ialgo="<<ialgo<<", simtrackid="<<simTrackId<<", #recHits="<<hit1<<","<<hit2<<","<<hit3<<std::endl;
      seedHit_old[simTrackId].push_back(hit1);
      seedHit_old[simTrackId].push_back(hit2);
      seedHit_old[simTrackId].push_back(hit3);

      output[ialgo]->push_back(TrajectorySeed(initialState, recHits, PropagationDirection::alongMomentum));
#ifdef FAMOS_DEBUG
      std::cout << "Trajectory seed created ! " << std::endl;
#endif
      break;
      // End of the loop over seeding algorithms
    }
    // End on the loop over simtracks
  }

  for (unsigned int ialgo = 0; ialgo<seedingAlgo.size(); ++ ialgo)
  {
	  TrajectorySeedCollection* seed_new = output_new[ialgo];
	  TrajectorySeedCollection* seed_old = output[ialgo];
	  if (seed_new->size()!=seed_old->size())
	  {
		  std::cout<<"NOT EQUAL: NUMBER OF SEEDS! ("<<seedingAlgo[ialgo]<<"), old:"<<seed_old->size()<<", new:"<<seed_new->size()<<std::endl;
	  }

  }

  for ( unsigned ialgo=0; ialgo<seedingAlgo.size(); ++ialgo ) {
    std::auto_ptr<TrajectorySeedCollection> p(output[ialgo]);
    e.put(p,seedingAlgo[ialgo]);
  }


  for (unsigned int isimtrack=0; isimtrack<seedHit_old.size(); ++isimtrack)
  {
	  if (seedHit_old[isimtrack].size()==seedHit_new[isimtrack].size())
	  {
	  	  for (unsigned int ihit=0; ihit<seedHit_old[isimtrack].size();++ihit)
	  	  {

			  if (seedHit_old[isimtrack][ihit]!=seedHit_new[isimtrack][ihit])
			  {
				  std::cout<<"NOT EQUAL: simtrack="<<isimtrack<<", hit="<<ihit<<" number is different!"<<std::endl;
				  //assert(false);
			  }
		  }
	  }
	  else
	  {
		  if (seedHit_old[isimtrack].size()<seedHit_new[isimtrack].size())
		  {
			  std::cout<<"NOT EQUAL: simtrack="<<isimtrack<<", new number of hits are more!"<<std::endl;
		  }
		  else
		  {
			  std::cout<<"NOT EQUAL: simtrack="<<isimtrack<<", new number of hits are less!"<<std::endl;
			  assert(false);
		  }

	  }
  }




  std::cout<<"-------------------"<<std::endl;
  std::cout<<"-------------------"<<std::endl;
  */
}


