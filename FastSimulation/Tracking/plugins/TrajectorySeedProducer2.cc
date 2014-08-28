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
    thePropagator(nullptr) //TODO:: what else should be initialized properly?
{  

    // The input tag for the beam spot
    theBeamSpot = conf.getParameter<edm::InputTag>("beamSpot");

    // The name of the TrajectorySeed Collections
    seedingAlgo = conf.getParameter<std::vector<std::string> >("seedingAlgo");
    for ( unsigned i=0; i<seedingAlgo.size(); ++i )
    {
        produces<TrajectorySeedCollection>(seedingAlgo[i]);
    }
    
    
    
    // The smallest true pT for a track candidate
    pTMin = conf.getParameter<std::vector<double> >("pTMin");
    if ( pTMin.size() != seedingAlgo.size() ) 
    {
        throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
        << " WARNING : pTMin does not have the proper size "
        << std::endl;
    }
    
    for ( unsigned i=0; i<pTMin.size(); ++i )
    {
        pTMin[i] *= pTMin[i];  // Cut is done of perp2() - CPU saver
    }
    
    // The smallest number of Rec Hits for a track candidate
    minRecHits = conf.getParameter<std::vector<unsigned int> >("minRecHits");
    if ( minRecHits.size() != seedingAlgo.size() ) 
    {
        throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
        << " WARNING : minRecHits does not have the proper size "
        << std::endl;
    }
    
    // Set the overall number hits to be checked
    absMinRecHits = 0;
    for ( unsigned ialgo=0; ialgo<minRecHits.size(); ++ialgo ) 
    {
        if ( minRecHits[ialgo] > absMinRecHits ) 
        {
            absMinRecHits = minRecHits[ialgo];
        }
    }
    
    // The smallest true impact parameters (d0 and z0) for a track candidate
    maxD0 = conf.getParameter<std::vector<double> >("maxD0");
    if ( maxD0.size() != seedingAlgo.size() )
    {
        throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
        << " WARNING : maxD0 does not have the proper size "
        << std::endl;
    }
    maxZ0 = conf.getParameter<std::vector<double> >("maxZ0");
    if ( maxZ0.size() != seedingAlgo.size() ) 
    {
        throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
        << " WARNING : maxZ0 does not have the proper size "
        << std::endl;
    }
    // The name of the hit producer
    hitProducer = conf.getParameter<edm::InputTag>("HitProducer");

    // The cuts for seed cleaning
    seedCleaning = conf.getParameter<bool>("seedCleaning");

    // Number of hits needed for a seed
    numberOfHits = conf.getParameter<std::vector<unsigned int> >("numberOfHits");
    if ( numberOfHits.size() != seedingAlgo.size() ) 
    {
        throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
        << " WARNING : numberOfHits does not have the proper size "
        << std::endl;
    }
    // Seeding based on muons
    selectMuons = conf.getParameter<bool>("selectMuons");

    // read Layers
    std::vector<std::string> layerList = conf.getParameter<std::vector<std::string> >("layerList");
    //for (unsigned i=0; i<layerList.size();i++) std::cout << "------- Layers = " << layerList[i] << std::endl; 

    for(std::vector<std::string>::const_iterator it=layerList.begin(); it < layerList.end(); ++it) 
    {
        std::vector<TrackingLayer> tempResult;
        std::string line = *it;
        std::string::size_type pos=0;
        while (pos != std::string::npos) 
        {
            pos=line.find("+");
            std::string layer = line.substr(0, pos);
            TrackingLayer layerSpec = TrackingLayer::createFromString(layer);

            tempResult.push_back(layerSpec);
            line=line.substr(pos+1,std::string::npos); 
        }
        theLayersInSets.push_back(tempResult);
    }

    originRadius = conf.getParameter<std::vector<double> >("originRadius");
    if ( originRadius.size() != seedingAlgo.size() ) 
    {
        throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
        << " WARNING : originRadius does not have the proper size "
        << std::endl;
    }
    originHalfLength = conf.getParameter<std::vector<double> >("originHalfLength");
    if ( originHalfLength.size() != seedingAlgo.size() ) 
    {
        throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
        << " WARNING : originHalfLength does not have the proper size "
        << std::endl;
    }
    originpTMin = conf.getParameter<std::vector<double> >("originpTMin");
    if ( originpTMin.size() != seedingAlgo.size() ) 
    {
        throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
        << " WARNING : originpTMin does not have the proper size "
        << std::endl;
    }
    primaryVertices = conf.getParameter<std::vector<edm::InputTag> >("primaryVertices");
    if ( primaryVertices.size() != seedingAlgo.size() ) 
    {
        throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
        << " WARNING : primaryVertices does not have the proper size "
        << std::endl;
    }
    zVertexConstraint = conf.getParameter<std::vector<double> >("zVertexConstraint");
    if ( zVertexConstraint.size() != seedingAlgo.size() ) 
    {
        throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
        << " WARNING : zVertexConstraint does not have the proper size "
        << std::endl;
    }


    // consumes
    beamSpotToken = consumes<reco::BeamSpot>(theBeamSpot);
    edm::InputTag _label("famosSimHits");
    simTrackToken = consumes<edm::SimTrackContainer>(_label);
    simVertexToken = consumes<edm::SimVertexContainer>(_label);
    recHitToken = consumes<SiTrackerGSMatchedRecHit2DCollection>(hitProducer);
    for ( unsigned ialgo=0; ialgo<seedingAlgo.size(); ++ialgo ) 
    {
        _label = edm::InputTag(primaryVertices[ialgo]);
        recoVertexToken.push_back(consumes<reco::VertexCollection>(_label));
    }


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


void 
TrajectorySeedProducer2::beginRun(edm::Run const&, const edm::EventSetup & es) 
{

    //services
    //  es.get<TrackerRecoGeometryRecord>().get(theGeomSearchTracker);

    edm::ESHandle<MagneticField>          magField;
    edm::ESHandle<TrackerGeometry>        geometry;
    edm::ESHandle<MagneticFieldMap>       magFieldMap;


    es.get<IdealMagneticFieldRecord>().get(magField);
    es.get<TrackerDigiGeometryRecord>().get(geometry);
    es.get<MagneticFieldMapRecord>().get(magFieldMap);

    theMagField = &(*magField);
    theGeometry = &(*geometry);
    theFieldMap = &(*magFieldMap);

    thePropagator = new PropagatorWithMaterial(alongMomentum,0.105,&(*theMagField)); 

    const GlobalPoint g(0.,0.,0.);

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


bool
TrajectorySeedProducer2::compatibleWithBeamAxis(GlobalPoint& gpos1, 
					       GlobalPoint& gpos2,
					       double error,
					       bool forward,
					       unsigned algo) const {

  if ( !seedCleaning ) return true;

  // The hits 1 and 2 positions, in HepLorentzVector's
  XYZTLorentzVector thePos1(gpos1.x(),gpos1.y(),gpos1.z(),0.);
  XYZTLorentzVector thePos2(gpos2.x(),gpos2.y(),gpos2.z(),0.);
#ifdef FAMOS_DEBUG
  std::cout << "ThePos1 = " << thePos1 << std::endl;
  std::cout << "ThePos2 = " << thePos2 << std::endl;
#endif


  // Create new particles that pass through the second hit with pT = ptMin 
  // and charge = +/-1
  
  // The momentum direction is by default joining the two hits 
  XYZTLorentzVector theMom2 = (thePos2-thePos1);

  // The corresponding RawParticle, with an (irrelevant) electric charge
  // (The charge is determined in the next step)
  ParticlePropagator myPart(theMom2,thePos2,1.,theFieldMap);

  /// Check that the seed is compatible with a track coming from within
  /// a cylinder of radius originRadius, with a decent pT, and propagate
  /// to the distance of closest approach, for the appropriate charge
  bool intersect = myPart.propagateToBeamCylinder(thePos1,originRadius[algo]*1.);
  if ( !intersect ) return false;

#ifdef FAMOS_DEBUG
  std::cout << "MyPart R = " << myPart.R() << "\t Z = " << myPart.Z() 
	    << "\t pT = " << myPart.Pt() << std::endl;
#endif

  // Check if the constraints are satisfied
  // 1. pT at cylinder with radius originRadius
  if ( myPart.Pt() < originpTMin[algo] ) return false;

  // 2. Z compatible with beam spot size
  if ( fabs(myPart.Z()-z0) > originHalfLength[algo] ) return false;

  // 3. Z compatible with one of the primary vertices (always the case if no primary vertex)
  const reco::VertexCollection* theVertices = vertices[algo];
  if (!theVertices) return true;
  unsigned nVertices = theVertices->size();
  if ( !nVertices || zVertexConstraint[algo] < 0. ) return true;
  // Radii of the two hits with respect to the beam spot position
  double R1 = std::sqrt ( (thePos1.X()-x0)*(thePos1.X()-x0) 
			+ (thePos1.Y()-y0)*(thePos1.Y()-y0) );
  double R2 = std::sqrt ( (thePos2.X()-x0)*(thePos2.X()-x0) 
			+ (thePos2.Y()-y0)*(thePos2.Y()-y0) );
  // Loop on primary vertices
  for ( unsigned iv=0; iv<nVertices; ++iv ) { 
    // Z position of the primary vertex
    double zV = (*theVertices)[iv].z();
    // Constraints on the inner hit
    double checkRZ1 = forward ?
      (thePos1.Z()-zV+zVertexConstraint[algo]) / (thePos2.Z()-zV+zVertexConstraint[algo]) * R2 : 
      -zVertexConstraint[algo] + R1/R2*(thePos2.Z()-zV+zVertexConstraint[algo]);
    double checkRZ2 = forward ?
      (thePos1.Z()-zV-zVertexConstraint[algo])/(thePos2.Z()-zV-zVertexConstraint[algo]) * R2 :
      +zVertexConstraint[algo] + R1/R2*(thePos2.Z()-zV-zVertexConstraint[algo]);
    double checkRZmin = std::min(checkRZ1,checkRZ2)-3.*error;
    double checkRZmax = std::max(checkRZ1,checkRZ2)+3.*error;
    // Check if the innerhit is within bounds
    bool compat = forward ?
      checkRZmin < R1 && R1 < checkRZmax : 
      checkRZmin < thePos1.Z()-zV && thePos1.Z()-zV < checkRZmax; 
    // If it is, just return ok
    if ( compat ) return compat;
  }
  // Otherwise, return not ok
  return false;

}  

