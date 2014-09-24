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
    thePropagator(nullptr),
    vertices(nullptr) //TODO:: what else should be initialized properly?
{  
    outputSeedCollectionName="seeds";
    if (conf.exists("outputSeedCollectionName"))
    {
        outputSeedCollectionName=conf.getParameter<std::string>("outputSeedCollectionName");
    }
    if (conf.exists("seedingAlgo"))
    {
        std::vector<std::string> algoNames=conf.getParameter<std::vector<std::string>>("seedingAlgo");
    }
    // The input tag for the beam spot
    theBeamSpot = conf.getParameter<edm::InputTag>("beamSpot");

    // The name of the TrajectorySeed Collections
    produces<TrajectorySeedCollection>(outputSeedCollectionName);
    
    
    
    // The smallest true pT for a track candidate
    pTMin = conf.getParameter<double>("pTMin");

    
    // The smallest number of Rec Hits for a track candidate
    minRecHits = conf.getParameter<unsigned int>("minRecHits");

    //TODO: REMOVE
    // Set the overall number hits to be checked
    absMinRecHits = minRecHits;

    // The smallest true impact parameters (d0 and z0) for a track candidate
    maxD0 = conf.getParameter<double>("maxD0");
    
    maxZ0 = conf.getParameter<double>("maxZ0");
    
    // The name of the hit producer
    hitProducer = conf.getParameter<edm::InputTag>("HitProducer");

    //TODO: What is this???
    // The cuts for seed cleaning
    seedCleaning = conf.getParameter<bool>("seedCleaning");

    // Number of hits needed for a seed
    numberOfHits = conf.getParameter<unsigned int>("numberOfHits");
    
    //TODO: What is this??
    // Seeding based on muons
    selectMuons = conf.getParameter<bool>("selectMuons");

    // read Layers
    std::vector<std::string> layerStringList = conf.getParameter<std::vector<std::string>>("layerList");

    for(auto it=layerStringList.cbegin(); it < layerStringList.cend(); ++it) 
    {
        std::vector<TrackingLayer> trackingLayerList;
        std::string line = *it;
        std::string::size_type pos=0;
        while (pos != std::string::npos) 
        {
            pos=line.find("+");
            std::string layer = line.substr(0, pos);
            TrackingLayer layerSpec = TrackingLayer::createFromString(layer);

            trackingLayerList.push_back(layerSpec);
            line=line.substr(pos+1,std::string::npos); 
        }
        _seedingTree.insert(trackingLayerList);
        seedingLayers.push_back(std::move(trackingLayerList));
    }

    originRadius = conf.getParameter<double>("originRadius");

    originHalfLength = conf.getParameter<double>("originHalfLength");

    originpTMin = conf.getParameter<double>("originpTMin");
 
    edm::InputTag primaryVertex = conf.getParameter<edm::InputTag>("primaryVertex");

    zVertexConstraint = conf.getParameter<double>("zVertexConstraint");

   


    skipPVCompatibility=false;
    if (conf.exists("skipPVCompatibility"))
    {
        skipPVCompatibility = conf.getParameter<bool>("skipPVCompatibility");
    }


    // consumes
    beamSpotToken = consumes<reco::BeamSpot>(theBeamSpot);
    edm::InputTag("famosSimHits");
    simTrackToken = consumes<edm::SimTrackContainer>(edm::InputTag("famosSimHits"));
    simVertexToken = consumes<edm::SimVertexContainer>(edm::InputTag("famosSimHits"));
    recHitToken = consumes<SiTrackerGSMatchedRecHit2DCollection>(hitProducer);
    
    recoVertexToken=consumes<reco::VertexCollection>(primaryVertex);
} 


void 
TrajectorySeedProducer2::beginRun(edm::Run const&, const edm::EventSetup & es) 
{
    edm::ESHandle<MagneticField> magneticFieldHandle;
    edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
    edm::ESHandle<MagneticFieldMap> magneticFieldMapHandle;
	edm::ESHandle<TrackerTopology> trackerTopologyHandle;
	
    es.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
    es.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
    es.get<MagneticFieldMapRecord>().get(magneticFieldMapHandle);
    es.get<IdealGeometryRecord>().get(trackerTopologyHandle);
    
    magneticField = &(*magneticFieldHandle);
    trackerGeometry = &(*trackerGeometryHandle);
    magneticFieldMap = &(*magneticFieldMapHandle);
    trackerTopology = &(*trackerTopologyHandle);

    //TODO: What is this???
    thePropagator = new PropagatorWithMaterial(alongMomentum,0.105,magneticField); 
}

bool
TrajectorySeedProducer2::passSimTrackQualityCuts(const SimTrack& theSimTrack, const SimVertex& theSimVertex, unsigned int trackingAlgorithmId) const
{

	//require min pT of the simtrack
	if ( theSimTrack.momentum().Perp2() < pTMin*pTMin)
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
    const double x0 = beamspotPosition.X();
    const double y0 = beamspotPosition.Y();
    const double z0 = beamspotPosition.Z();
	if ( theParticle.xyImpactParameter(x0,y0) > maxD0 )
	{
		return false;
	}
    if ( fabs( theParticle.zImpactParameter(x0,y0) - z0 ) > maxZ0)
	{
    	return false;
	}
    return true;
}

bool
TrajectorySeedProducer2::pass2HitsCuts(const TrajectorySeedHitCandidate& hit1, const TrajectorySeedHitCandidate& hit2, unsigned int trackingAlgorithmId) const
{
	bool compatible=false;
	if(skipPVCompatibility)
	{
		compatible = true;
	} 
	else 
	{
	    GlobalPoint gpos1 = hit1.globalPosition();
	    GlobalPoint gpos2 = hit2.globalPosition();
	    bool forward = hit1.isForward();
	    double error = std::sqrt(hit1.largerError()+hit2.largerError());
		compatible = compatibleWithBeamAxis(gpos1,gpos2,error,forward,0);
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

        /*
		if ( currentHitIndex >= trackerRecHits.size())
		{
		    //TODO: one can speed things up here if a hit is already too far from the seeding layers (barrel/disks)
			return std::vector<unsigned int>();
		}
		*/
		
		for (unsigned int inext=currentHitIndex+1; inext< trackerRecHits.size(); ++inext)
		{
			if (trackerRecHits[currentHitIndex].getTrackingLayer()==trackerRecHits[inext].getTrackingLayer())
			{
			    if (processSkippedHits)
			    {
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
                //skipping all following hits on the same layer
                irecHit+=1; 
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
TrajectorySeedProducer2::produce(edm::Event& e, const edm::EventSetup& es) 
{        






	//  unsigned nTrackCandidates = 0;
	PTrajectoryStateOnDet initialState;

	// Beam spot
	edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
	e.getByToken(beamSpotToken,recoBeamSpotHandle);
	beamspotPosition = recoBeamSpotHandle->position();

	// SimTracks and SimVertices
	edm::Handle<edm::SimTrackContainer> theSimTracks;
	e.getByToken(simTrackToken,theSimTracks);

	edm::Handle<edm::SimVertexContainer> theSimVtx;
	e.getByToken(simVertexToken,theSimVtx);
  
	//  edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
	edm::Handle<SiTrackerGSMatchedRecHit2DCollection> theGSRecHits;
	e.getByToken(recHitToken, theGSRecHits);

    // Primary vertices
    edm::Handle<reco::VertexCollection> theRecVtx;
    if (e.getByToken(recoVertexToken,theRecVtx))
    {
    
        //this can be nullptr if the PV compatiblity should not be tested against
        vertices = &(*theRecVtx);
    }
	    
	    
    // Output - gets moved, no delete needed
	std::auto_ptr<TrajectorySeedCollection> output{new TrajectorySeedCollection()};
	    
	//if no hits -> directly write empty collection
	if(theGSRecHits->size() == 0)
	{
	    e.put(output,outputSeedCollectionName);
		return;
	}
	std::cout<<std::endl;
	std::cout<<std::endl;
    std::cout<<"collection: "<<outputSeedCollectionName.c_str()<<std::endl;
    std::cout<<"-----------------------------"<<std::endl;
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

		if (!this->passSimTrackQualityCuts(theSimTrack,theSimVertex,0))
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
			
			currentTrackerHit = TrajectorySeedHitCandidate(&vec,trackerGeometry,trackerTopology);
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
		if ( numberOfNonEqualHits < minRecHits) continue;

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
		
		
		
		std::vector<unsigned int> seedHitNumbers = iterateHits(0,trackerRecHits,hitIndicesInTree,true,0);

		if (seedHitNumbers.size()>0)
		{
            std::cout<<"accept: "<<currentSimTrackId<<std::endl;
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
			GlobalTrajectoryParameters initialParams(position,momentum,(int)charge,magneticField);
			AlgebraicSymMatrix55 errorMatrix= AlgebraicMatrixID();
			//this line help the fit succeed in the case of pixelless tracks (4th and 5th iteration)
			//for the future: probably the best thing is to use the mini-kalmanFilter
			if(trackerRecHits[seedHitNumbers[0]].subDetId() !=1 ||trackerRecHits[seedHitNumbers[0]].subDetId() !=2)
			{
				errorMatrix = errorMatrix * 0.0000001;
			}
			CurvilinearTrajectoryError initialError(errorMatrix);
			FreeTrajectoryState initialFTS(initialParams, initialError);
			
			
			const GeomDet* initialLayer = trackerGeometry->idToDet( recHits.front().geographicalId() );
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
			output->push_back(TrajectorySeed(initialState, recHits, PropagationDirection::alongMomentum));
		
		
		    
		}
	} //end loop over simtracks
    

	e.put(output,outputSeedCollectionName);
}


bool
TrajectorySeedProducer2::compatibleWithBeamAxis(
        const GlobalPoint& gpos1, 
        const GlobalPoint& gpos2,
        double error,
        bool forward,
        unsigned algo
    ) const 
{

    const double x0 = beamspotPosition.X();
    const double y0 = beamspotPosition.Y();
    const double z0 = beamspotPosition.Z();
    
    if ( !seedCleaning ) 
    {
        return true;
    }

    // The hits 1 and 2 positions, in HepLorentzVector's
    XYZTLorentzVector thePos1(gpos1.x(),gpos1.y(),gpos1.z(),0.);
    XYZTLorentzVector thePos2(gpos2.x(),gpos2.y(),gpos2.z(),0.);
    
    // Create new particles that pass through the second hit with pT = ptMin 
    // and charge = +/-1

    // The momentum direction is by default joining the two hits 
    XYZTLorentzVector theMom2 = (thePos2-thePos1);

    // The corresponding RawParticle, with an (irrelevant) electric charge
    // (The charge is determined in the next step)
    ParticlePropagator myPart(theMom2,thePos2,1.,magneticFieldMap);

    /// Check that the seed is compatible with a track coming from within
    /// a cylinder of radius originRadius, with a decent pT, and propagate
    /// to the distance of closest approach, for the appropriate charge
    bool intersect = myPart.propagateToBeamCylinder(thePos1,originRadius*1.);
    if ( !intersect ) 
    {
        return false;
    }

    // Check if the constraints are satisfied
    // 1. pT at cylinder with radius originRadius
    if ( myPart.Pt() < originpTMin ) 
    {
        return false;
    }

    // 2. Z compatible with beam spot size
    if ( fabs(myPart.Z()-z0) > originHalfLength ) 
    {
        return false;
    }


    // 3. Z compatible with one of the primary vertices (always the case if no primary vertex)
    if (!vertices) 
    {
        return true;
    }
    unsigned int nVertices = vertices->size();
    if ( !nVertices || zVertexConstraint < 0. ) 
    {
        return true;
    }
    // Radii of the two hits with respect to the beam spot position
    double R1 = std::sqrt ( (thePos1.X()-x0)*(thePos1.X()-x0) + (thePos1.Y()-y0)*(thePos1.Y()-y0) );
    double R2 = std::sqrt ( (thePos2.X()-x0)*(thePos2.X()-x0) + (thePos2.Y()-y0)*(thePos2.Y()-y0) );
    // Loop on primary vertices
    
    //TODO: Check if pTMin is correctly used (old code stored pTMin^2 in pTMin) 
    
    for ( unsigned iv=0; iv<nVertices; ++iv ) 
    { 
        // Z position of the primary vertex
        double zV = (*vertices)[iv].z();
        // Constraints on the inner hit
        double checkRZ1 = forward ?
        (thePos1.Z()-zV+zVertexConstraint) / (thePos2.Z()-zV+zVertexConstraint) * R2 : 
        -zVertexConstraint + R1/R2*(thePos2.Z()-zV+zVertexConstraint);
        double checkRZ2 = forward ?
        (thePos1.Z()-zV-zVertexConstraint)/(thePos2.Z()-zV-zVertexConstraint) * R2 :
        +zVertexConstraint + R1/R2*(thePos2.Z()-zV-zVertexConstraint);
        double checkRZmin = std::min(checkRZ1,checkRZ2)-3.*error;
        double checkRZmax = std::max(checkRZ1,checkRZ2)+3.*error;
        // Check if the innerhit is within bounds
        bool compat = forward ?
        checkRZmin < R1 && R1 < checkRZmax : 
        checkRZmin < thePos1.Z()-zV && thePos1.Z()-zV < checkRZmax; 
        // If it is, just return ok
        if ( compat ) 
        {
            return compat;
        }
    }
    // Otherwise, return not ok
    return false;

}  

