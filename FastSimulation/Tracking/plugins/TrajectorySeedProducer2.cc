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

template class SeedingTree<TrackingLayer>;
template class SeedingNode<TrackingLayer>;

TrajectorySeedProducer2::TrajectorySeedProducer2(const edm::ParameterSet& conf):
    TrajectorySeedProducer(conf)
{  
    std::cout<<std::endl;
	std::cout<<"config: "<<seedingAlgo[0]<<std::endl;
	for (unsigned int ilayerset=0; ilayerset<theLayersInSets.size(); ++ ilayerset)
	{
		_seedingTree.insert(theLayersInSets[ilayerset]);
	}
    std::cout<<"---------"<<std::endl;
    _seedingTree.sort();
    _seedingTree.printRecursive();
    
    for (auto const& layer: _seedingTree.getSingleSet())
    //for (SeedingTree<LayerSpec>::SingleSet::const_iterator it = _seedingTree.getSingleSet().begin(); it!=_seedingTree.getSingleSet().end(); ++it)
    {
        
        //std::cout<<it-_seedingTree.getSingleSet().begin()<<",";
        std::cout<<layer.toString().c_str()<<" ["<<layer.toIdString().c_str()<<"], ";
    }
    std::cout<<std::endl;
    std::cout<<std::endl;
} 

bool
TrajectorySeedProducer2::passSimTrackQualityCuts(const SimTrack& theSimTrack, const SimVertex& theSimVertex, unsigned int trackingAlgorithmId) const
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
TrajectorySeedProducer2::pass2HitsCuts(const TrajectorySeedHitCandidate& hit1, const TrajectorySeedHitCandidate& hit2, unsigned int trackingAlgorithmId) const
{
	GlobalPoint gpos1 = hit1.globalPosition();
	GlobalPoint gpos2 = hit2.globalPosition();
	bool forward = hit1.isForward();
	double error = std::sqrt(hit1.largerError()+hit2.largerError());
	bool compatible=false;

	//TODO: get rid of evil string comp!!!!
	//if this is important => add a config option to specify if compatibility with PV is required
	
	if(seedingAlgo[trackingAlgorithmId] == "PixelLess" ||  seedingAlgo[trackingAlgorithmId] ==  "TobTecLayerPairs")
	{
		compatible = true;
	} else {
		compatible = compatibleWithBeamAxis(gpos1,gpos2,error,forward,trackingAlgorithmId);
		//if (!compatible) std::cout<<"reject beam axis"<<std::endl;
	}
	return compatible;
}

const SeedingNode<TrackingLayer>* TrajectorySeedProducer2::insertHit(
    const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
    std::vector<int>& hitIndicesInTree,
    const SeedingNode<TrackingLayer>* node, unsigned int trackerHit,
    const unsigned int trackingAlgorithmId
) const
{
    //std::cout<<"\tchecking: ";
    //std::cout<<"\t\t";
    //node->print();
    //std::cout<<"\t\thitIndex="<<hitIndicesInTree[node->getIndex()]<<std::endl;
    if (hitIndicesInTree[node->getIndex()]<0)
    {
        const TrajectorySeedHitCandidate& currentTrackerHit = trackerRecHits[trackerHit];
        if (!isHitOnLayer(currentTrackerHit,node->getData()))
        {
            return nullptr;
        }
        //std::cout<<"\t\tpassed layer"<<std::endl;
        if (!passHitTuplesCuts(*node,trackerRecHits,hitIndicesInTree,currentTrackerHit,trackingAlgorithmId))
        {
            return nullptr;
        }
        //std::cout<<"\t\t\tpassed cuts"<<std::endl;
        hitIndicesInTree[node->getIndex()]=trackerHit;
        if (node->getChildrenSize()==0)
        {
            //std::cout<<"\t\t\t\tseed found"<<std::endl;
            return node;
        }
        return nullptr;
    }
    else
    {
        //std::cout<<"\t\tprocess children"<<std::endl;
        for (unsigned int ichild = 0; ichild<node->getChildrenSize(); ++ichild)
        {
            const SeedingNode<TrackingLayer>* seed = insertHit(trackerRecHits,hitIndicesInTree,node->getChild(ichild),trackerHit,trackingAlgorithmId);
            if (seed)
            {
                return seed;
            }
        }
    }
    return nullptr;
}


std::vector<unsigned int> TrajectorySeedProducer2::iterateHits(
		unsigned int start,
		const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
		std::vector<int> hitIndicesInTree,
		bool processSkippedHits,
		unsigned int trackingAlgorithmId
	) const
{
	for (unsigned int irecHit = start; irecHit<trackerRecHits.size(); ++irecHit)
	{
        unsigned int currentHitIndex=irecHit;
        //std::cout<<"hit="<<currentHitIndex<<std::endl;
        /*
		if ( currentHitIndex >= trackerRecHits.size())
		{
		    //TODO: one can speed things up here if a hit is already too far from the seeding layers (barrel/disks)
			return std::vector<unsigned int>();
		}
		*/
		//const TrajectorySeedHitCandidate& currentTrackerHit = trackerRecHits[irecHit];
		
		//process all hits after currentHit on the same layer
		
		for (unsigned int inext=currentHitIndex+1; inext< trackerRecHits.size(); ++inext)
		{
			if (trackerRecHits[currentHitIndex].getTrackingLayer()==trackerRecHits[inext].getTrackingLayer())
			{
			    if (processSkippedHits)
			    {
			        
			        //std::cout<<"skipping hit: "<<inext<<std::endl;
			        
	                //std::cout<<"\t\t"<<trackerRecHits[inext].getSeedingLayer().subDet<<":"<<trackerRecHits[inext].getSeedingLayer().idLayer<<std::endl;//<<", pos=("<<trackerRecHits[inext].globalPosition().x()<<","<<trackerRecHits[inext].globalPosition().y()<<","<<trackerRecHits[inext].globalPosition().z()<<")"<<std::endl;
	                //TODO: check if hit can even be on free position within tree's layer
			        std::vector<unsigned int> seedHits = iterateHits(
		                inext,
		                trackerRecHits,
		                hitIndicesInTree,
		                false,
		                trackingAlgorithmId
	                );
	                if (seedHits.size()>0)
	                {
	                    return seedHits;
	                }
	                
	                
                }
                irecHit+=1; //skipping all following hits on the same layer
	            
			}
			else
			{
			    break;
			}
		}

		processSkippedHits=true;
		
		//process currentHit
		
		//iterate through the seeding tree
		const SeedingNode<TrackingLayer>* seedNode = nullptr;
		for (unsigned int iroot=0; seedNode==nullptr && iroot<_seedingTree.numberOfRoots(); ++iroot)
		{
		    //std::cout<<"\troot: "<<iroot<<std::endl;
		    seedNode=insertHit(trackerRecHits,hitIndicesInTree,_seedingTree.getRoot(iroot), currentHitIndex,trackingAlgorithmId);
		}
		if (seedNode)
		{
		    std::vector<unsigned int> seedIndices(seedNode->getDepth()+1);
		    while (seedNode)
		    {
		        seedIndices[seedNode->getDepth()]=hitIndicesInTree[seedNode->getIndex()];
		        seedNode=seedNode->getParent();
		    }
		    return seedIndices;
		}
		
	}

	return std::vector<unsigned int>();
	
}

void 
TrajectorySeedProducer2::produce(edm::Event& e, const edm::EventSetup& es) {        

    /*
    std::cout<<std::endl;
	std::cout<<std::endl;
	std::cout<<"-------------------"<<std::endl;
	std::cout<<seedingAlgo[0]<<std::endl;
	std::cout<<"-------------------"<<std::endl;
    */


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
	
	
	std::vector<std::vector<std::pair<int,TrajectorySeedHitCandidate >>> newhits;
	newhits.resize(theSimTracks->size());
	
	std::vector<std::vector<std::pair<int,TrajectorySeedHitCandidate >>> oldhits;
	oldhits.resize(theSimTracks->size());
    
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
		//if (currentSimTrackId!=286) continue;
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

			TrajectorySeedHitCandidate previousTrackerHit;
			TrajectorySeedHitCandidate currentTrackerHit;
			unsigned int numberOfNonEqualHits=0;

			std::vector<TrajectorySeedHitCandidate> trackerRecHits;
			for (SiTrackerGSMatchedRecHit2DCollection::const_iterator itRecHit = recHitRange.first; itRecHit!=recHitRange.second; ++itRecHit)
			{
				const SiTrackerGSMatchedRecHit2D& vec = *itRecHit;
				previousTrackerHit=currentTrackerHit;
				
				currentTrackerHit = TrajectorySeedHitCandidate(&vec,theGeometry,tTopo);
				//std::cout<<"creating TrajectorySeedHitCandidate: "<<itRecHit-recHitRange.first<<": "<<currentTrackerHit.getSeedingLayer().subDet<<":"<<currentTrackerHit.getSeedingLayer().idLayer<<", pos=("<<currentTrackerHit.globalPosition().x()<<","<<currentTrackerHit.globalPosition().y()<<","<<currentTrackerHit.globalPosition().z()<<")"<<std::endl;
				
				if (!currentTrackerHit.isOnTheSameLayer(previousTrackerHit))
				{
					++numberOfNonEqualHits;
				}
				trackerRecHits.push_back(std::move(currentTrackerHit));
				/*
				if (_seedingTree.getSingleSet().find(currentTrackerHit.getSeedingLayer())!=_seedingTree.getSingleSet().end())
				{
				    
				}
                */
			}
			if ( numberOfNonEqualHits < minRecHits[ialgo] ) continue;

            std::vector<int> hitIndicesInTree(_seedingTree.numberOfNodes(),-1);
			//A SeedingNode is associated by its index to this list. The list stores the indices of the hits in 'trackerRecHits'
			/* example
			    SeedingNode                     | hit index                 | hit
			    -------------------------------------------------------------------------------
			    index=  0:  [BPix1]             | hitIndicesInTree[0] (=1)  | trackerRecHits[1]
                index=  1:   -- [BPix2]         | hitIndicesInTree[1] (=3)  | trackerRecHits[3]
                index=  2:   --  -- [BPix3]     | hitIndicesInTree[2] (=4)  | trackerRecHits[4]
                index=  3:   --  -- [FPix1_pos] | hitIndicesInTree[3] (=6)  | trackerRecHits[6]
                index=  4:   --  -- [FPix1_neg] | hitIndicesInTree[4] (=7)  | trackerRecHits[7]
			
			    The implementation has been chosen such that the tree only needs to be build once upon construction.
			*/
			
			
			
			std::vector<unsigned int> seedHitNumbers = iterateHits(0,trackerRecHits,hitIndicesInTree,true,ialgo);

			if (seedHitNumbers.size()>0)
			{

				edm::OwnVector<TrackingRecHit> recHits;
				for ( unsigned ihit=0; ihit<seedHitNumbers.size(); ++ihit )
				{
					TrackingRecHit* aTrackingRecHit = trackerRecHits[seedHitNumbers[ihit]].hit()->clone();
					recHits.push_back(aTrackingRecHit);
					
					//DEBUG
					newhits[currentSimTrackId].push_back(std::pair<int,TrajectorySeedHitCandidate >(seedHitNumbers[ihit],trackerRecHits[seedHitNumbers[ihit]]));
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
    
    
    /*
	for ( unsigned ialgo=0; ialgo<seedingAlgo.size(); ++ialgo )
	{
		std::auto_ptr<TrajectorySeedCollection> p(output[ialgo]);
		e.put(p,seedingAlgo[ialgo]);
	}
	*/
	
	
	
	TrajectorySeedProducer::produce(e, es, oldhits);
	
	
	int new_seeds=0;
	int missed_seeds=0;
	int total_seeds=0;
	for (unsigned int itrack = 0; itrack<newhits.size(); ++itrack)
	{
	    if (newhits[itrack].size()>0 || oldhits[itrack].size()>0)
	    {
	        ++total_seeds;
	    }
	    if ((newhits[itrack].size()>0 && oldhits[itrack].size()==0) || (newhits[itrack].size()==0 && oldhits[itrack].size()>0))
	    {

	        
	        if (newhits[itrack].size()>0)
	        {
	            
	            /*std::cout<<"simtrack = "<<itrack<<": new seed"<<std::endl;
	            for (unsigned int ihit = 0; ihit<newhits[itrack].size(); ++ ihit)
	            {
	                std::cout<<"\t hit: "<<newhits[itrack][ihit].first<<", "<<newhits[itrack][ihit].second.getSeedingLayer().print().c_str()<<", pos=("<<newhits[itrack][ihit].second.globalPosition().x()<<","<<newhits[itrack][ihit].second.globalPosition().y()<<","<<newhits[itrack][ihit].second.globalPosition().z()<<")"<<std::endl;
	            }
	            */
	            ++new_seeds;
	        }
	        if (oldhits[itrack].size()>0)
	        {
	            
	            /*std::cout<<"simtrack = "<<itrack<<": old seed"<<std::endl;
	            for (unsigned int ihit = 0; ihit<oldhits[itrack].size(); ++ ihit)
	            {
	                std::cout<<"\t hit: "<<oldhits[itrack][ihit].first<<", "<<oldhits[itrack][ihit].second.getSeedingLayer().print().c_str()<<", pos=("<<oldhits[itrack][ihit].second.globalPosition().x()<<","<<oldhits[itrack][ihit].second.globalPosition().y()<<","<<oldhits[itrack][ihit].second.globalPosition().z()<<")"<<std::endl;
	            }
	            */
	            ++missed_seeds;
	        }
        }
	}
	std::cout<<"summary: total seeds="<<total_seeds<<", missed seeds="<<missed_seeds<<", new seed="<<new_seeds<<std::endl;
	
  
}


