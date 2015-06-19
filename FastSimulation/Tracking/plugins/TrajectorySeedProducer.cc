
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
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMapRecord.h"

#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"

#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"

//Propagator withMaterial
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "FastSimulation/Tracking/plugins/SimTrackIdProducer.h"


#include <unordered_set>


template class SeedingTree<TrackingLayer>;
template class SeedingNode<TrackingLayer>;

TrajectorySeedProducer::TrajectorySeedProducer(const edm::ParameterSet& conf):
    magneticField(nullptr),
    magneticFieldMap(nullptr),
    trackerGeometry(nullptr),
    trackerTopology(nullptr),
    testBeamspotCompatibility(false),
    beamSpot(nullptr),
    testPrimaryVertexCompatibility(false),
    primaryVertices(nullptr)
{  
    // The name of the TrajectorySeed Collection
    produces<TrajectorySeedCollection>();



    const edm::ParameterSet& simTrackSelectionConfig = conf.getParameter<edm::ParameterSet>("simTrackSelection");
    // The smallest pT,dxy,dz for a simtrack
    simTrack_pTMin = simTrackSelectionConfig.getParameter<double>("pTMin");
    simTrack_maxD0 = simTrackSelectionConfig.getParameter<double>("maxD0");
    simTrack_maxZ0 = simTrackSelectionConfig.getParameter<double>("maxZ0");
    //simtracks to skip (were processed in previous iterations)
    std::vector<edm::InputTag> skipSimTrackTags = simTrackSelectionConfig.getParameter<std::vector<edm::InputTag> >("skipSimTrackIds");
    
    for ( unsigned int k=0; k<skipSimTrackTags.size(); ++k)
    {
        skipSimTrackIdTokens.push_back(consumes<std::vector<unsigned int> >(skipSimTrackTags[k]));
    }


    // The smallest number of hits for a track candidate
    minLayersCrossed = conf.getParameter<unsigned int>("minLayersCrossed");

    edm::InputTag beamSpotTag = conf.getParameter<edm::InputTag>("beamSpot");
    if (beamSpotTag.label()!="")
    {
        testBeamspotCompatibility=true;
        beamSpotToken = consumes<reco::BeamSpot>(beamSpotTag);
    }
    edm::InputTag primaryVertexTag = conf.getParameter<edm::InputTag>("primaryVertex");
    if (primaryVertexTag.label()!="")
    {
        testPrimaryVertexCompatibility=true;
        recoVertexToken=consumes<reco::VertexCollection>(primaryVertexTag);
    }

    //make sure that only one test is performed
    if (testBeamspotCompatibility && testPrimaryVertexCompatibility)
    {
        throw cms::Exception("FastSimulation/Tracking/TrajectorySeedProducer: bad configuration","Either 'beamSpot' or 'primaryVertex' compatiblity should be configured; not both");
    }
    
    // The name of the hit producer
    edm::InputTag recHitTag = conf.getParameter<edm::InputTag>("recHits");
    recHitToken = consumes<FastTMRecHitCombinations>(recHitTag);

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
    ptMin = conf.getParameter<double>("ptMin");
    nSigmaZ = conf.getParameter<double>("nSigmaZ");

    //make sure that only one cut is configured
    if (originHalfLength>=0 && nSigmaZ>=0)
    {
        throw cms::Exception("FastSimulation/Tracking/TrajectorySeedProducer: bad configuration","Either 'originHalfLength' or 'nSigmaZ' selection should be configured; not both. Deactivate one (or both) by setting it to <0.");
    }

    //make sure that performance cuts are not interfering with selection on reconstruction
    if ((ptMin>=0 && simTrack_pTMin>=0) && (ptMin<simTrack_pTMin))
    {
        throw cms::Exception("FastSimulation/Tracking/TrajectorySeedProducer: bad configuration","Performance cut on SimTrack pT is tighter than cut on pT estimate from seed.");
    }
    if ((originHalfLength>=0 && simTrack_maxZ0>=0) && (originHalfLength>simTrack_maxZ0))
    {
        throw cms::Exception("FastSimulation/Tracking/TrajectorySeedProducer: bad configuration","Performance cut on SimTrack dz is tighter than cut on dz estimate from seed.");
    }
    if ((originRadius>=0 && simTrack_maxD0>=0) && (originRadius>simTrack_maxD0))
    {
        throw cms::Exception("FastSimulation/Tracking/TrajectorySeedProducer: bad configuration","Performance cut on SimTrack dxy is tighter than cut on dxy estimate from seed.");
    }
    simTrackToken = consumes<edm::SimTrackContainer>(edm::InputTag("famosSimHits"));
    simVertexToken = consumes<edm::SimVertexContainer>(edm::InputTag("famosSimHits"));
}

void
TrajectorySeedProducer::beginRun(edm::Run const&, const edm::EventSetup & es) 
{
    edm::ESHandle<MagneticField> magneticFieldHandle;
    edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
    edm::ESHandle<MagneticFieldMap> magneticFieldMapHandle;
    edm::ESHandle<TrackerTopology> trackerTopologyHandle;

    es.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
    es.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
    es.get<MagneticFieldMapRecord>().get(magneticFieldMapHandle);
    es.get<TrackerTopologyRcd>().get(trackerTopologyHandle);
    
    magneticField = &(*magneticFieldHandle);
    trackerGeometry = &(*trackerGeometryHandle);
    magneticFieldMap = &(*magneticFieldMapHandle);
    trackerTopology = &(*trackerTopologyHandle);

    thePropagator = std::make_shared<PropagatorWithMaterial>(alongMomentum,0.105,magneticField);
}

bool
TrajectorySeedProducer::passSimTrackQualityCuts(const SimTrack& theSimTrack, const SimVertex& theSimVertex) const
{
    //require min pT of the simtrack
    if ((simTrack_pTMin>0) && ( theSimTrack.momentum().Perp2() < simTrack_pTMin*simTrack_pTMin))
    {
        return false;
    }
    if(((simTrack_maxD0<0) && (simTrack_maxZ0<0)) || (!testPrimaryVertexCompatibility && !testBeamspotCompatibility))
      {
	return true;
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


    //this are just some cuts on the SimTrack for speed up
    std::vector<const math::XYZPoint*> origins;
    if (testBeamspotCompatibility)
    {
        origins.push_back(&beamSpot->position());
    }
    if (testPrimaryVertexCompatibility)
    {
        for (unsigned int iv = 0; iv < primaryVertices->size(); ++iv)
        {
            origins.push_back(&(*primaryVertices)[iv].position());
        }
    }

    //only one possible origin is required to succeed
    for (unsigned int i = 0; i < origins.size(); ++i)
    {
        if ((simTrack_maxD0>0.0) && ( theParticle.xyImpactParameter(origins[i]->X(),origins[i]->Y()) > simTrack_maxD0 ))
        {
            continue;
        }
        if ((simTrack_maxZ0>0.0) && ( fabs( theParticle.zImpactParameter(origins[i]->X(),origins[i]->Y()) - origins[i]->Z()) > simTrack_maxZ0))
        {
            continue;
        }
        return true;
    }
    return false;
}

bool
TrajectorySeedProducer::pass2HitsCuts(const TrajectorySeedHitCandidate& hit1, const TrajectorySeedHitCandidate& hit2) const
{

    const GlobalPoint& globalHitPos1 = hit1.globalPosition();
    const GlobalPoint& globalHitPos2 = hit2.globalPosition();
    bool forward = hit1.isForward(); // true if hit is in endcap, false = barrel
    double error = std::sqrt(hit1.largerError()+hit2.largerError());
    if (testBeamspotCompatibility)
      {
	return compatibleWithBeamSpot(globalHitPos1,globalHitPos2,error,forward);
      }
    else if(testPrimaryVertexCompatibility)
      {
	return compatibleWithPrimaryVertex(globalHitPos1,globalHitPos2,error,forward);
      }
    else
      {
	return true;
      }
 
}

const SeedingNode<TrackingLayer>* TrajectorySeedProducer::insertHit(
    const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
    std::vector<int>& hitIndicesInTree,
    const SeedingNode<TrackingLayer>* node, unsigned int trackerHit
) const
{
    if (!node->getParent() || hitIndicesInTree[node->getParent()->getIndex()]>=0)
    {
        if (hitIndicesInTree[node->getIndex()]<0)
        {
            const TrajectorySeedHitCandidate& currentTrackerHit = trackerRecHits[trackerHit];
            if (!isHitOnLayer(currentTrackerHit,node->getData()))
            {
                return nullptr;
            }
            if (!passHitTuplesCuts(*node,trackerRecHits,hitIndicesInTree,currentTrackerHit))
            {
                return nullptr;
            }
            hitIndicesInTree[node->getIndex()]=trackerHit;
            if (node->getChildrenSize()==0)
            {
                return node;
            }
            return nullptr;
        }
        else
        {
            for (unsigned int ichild = 0; ichild<node->getChildrenSize(); ++ichild)
            {
                const SeedingNode<TrackingLayer>* seed = insertHit(trackerRecHits,hitIndicesInTree,node->getChild(ichild),trackerHit);
                if (seed)
                {
                    return seed;
                }
            }
        }
    }
    return nullptr;
}


std::vector<unsigned int> TrajectorySeedProducer::iterateHits(
        unsigned int start,
        const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
        std::vector<int> hitIndicesInTree,
        bool processSkippedHits
    ) const
{
    for (unsigned int irecHit = start; irecHit<trackerRecHits.size(); ++irecHit)
    {
        unsigned int currentHitIndex=irecHit;

        for (unsigned int inext=currentHitIndex+1; inext< trackerRecHits.size(); ++inext)
        {
            //if multiple hits are on the same layer -> follow all possibilities by recusion
            if (trackerRecHits[currentHitIndex].getTrackingLayer()==trackerRecHits[inext].getTrackingLayer())
            {
                if (processSkippedHits)
                {
                    //recusively call the method again with hit 'inext' but skip all following on the same layer though 'processSkippedHits=false'
                    std::vector<unsigned int> seedHits = iterateHits(
                        inext,
                        trackerRecHits,
                        hitIndicesInTree,
                        false
                    );
                    if (seedHits.size()>0)
                    {
                        return seedHits;
                    }
                }
                irecHit+=1; 
            }
            else
            {
                break;
            }
        }

        processSkippedHits=true;

        const SeedingNode<TrackingLayer>* seedNode = nullptr;
        for (unsigned int iroot=0; seedNode==nullptr && iroot<_seedingTree.numberOfRoots(); ++iroot)
        {
            seedNode=insertHit(trackerRecHits,hitIndicesInTree,_seedingTree.getRoot(iroot), currentHitIndex);
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
TrajectorySeedProducer::produce(edm::Event& e, const edm::EventSetup& es) 
{        
    PTrajectoryStateOnDet initialState;

    // the tracks to be skipped
    std::unordered_set<unsigned int> skipSimTrackIds;
    for ( unsigned int i=0; i<skipSimTrackIdTokens.size(); ++i )
      {
        edm::Handle<std::vector<unsigned int> > skipSimTrackIds_temp;
	e.getByToken(skipSimTrackIdTokens[i],skipSimTrackIds_temp);	
	skipSimTrackIds.insert(skipSimTrackIds_temp->begin(),skipSimTrackIds_temp->end());
      }
    // Beam spot
    if (testBeamspotCompatibility)
      {
        edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
        e.getByToken(beamSpotToken,recoBeamSpotHandle);
        beamSpot = recoBeamSpotHandle.product();
      }
    
    // Primary vertices
    if (testPrimaryVertexCompatibility)
    {
      edm::Handle<reco::VertexCollection> theRecVtx;
        e.getByToken(recoVertexToken,theRecVtx);
        primaryVertices = theRecVtx.product();
    }
    
    // SimTracks and SimVertices
    edm::Handle<edm::SimTrackContainer> theSimTracks;
    e.getByToken(simTrackToken,theSimTracks);
    
    edm::Handle<edm::SimVertexContainer> theSimVtx;
    e.getByToken(simVertexToken,theSimVtx);
    
    edm::Handle<FastTMRecHitCombinations> recHitCombinations;
    e.getByToken(recHitToken, recHitCombinations);

    std::auto_ptr<TrajectorySeedCollection> output{new TrajectorySeedCollection()};

    for (const auto & recHitCombination : *recHitCombinations)
    {

        uint32_t currentSimTrackId = recHitCombination.back().simTrackId(0);

        if(skipSimTrackIds.find(currentSimTrackId)!=skipSimTrackIds.end())
        {
            continue;
        }

        const SimTrack& theSimTrack = (*theSimTracks)[currentSimTrackId];

        int vertexIndex = theSimTrack.vertIndex();
        if (vertexIndex<0)
        {
            //tracks are required to be associated to a vertex
            continue;
        }
        const SimVertex& theSimVertex = (*theSimVtx)[vertexIndex];

        if (!this->passSimTrackQualityCuts(theSimTrack,theSimVertex))
        {
            continue;
        }

        TrajectorySeedHitCandidate previousTrackerHit;
        TrajectorySeedHitCandidate currentTrackerHit;
        unsigned int layersCrossed=0;

        std::vector<TrajectorySeedHitCandidate> trackerRecHits;
        for (const auto & _hit : recHitCombination )
        {
            previousTrackerHit=currentTrackerHit;

            currentTrackerHit = TrajectorySeedHitCandidate(&_hit,trackerGeometry,trackerTopology);

            if (!currentTrackerHit.isOnTheSameLayer(previousTrackerHit))
            {
                ++layersCrossed;
            }
            if (_seedingTree.getSingleSet().find(currentTrackerHit.getTrackingLayer())!=_seedingTree.getSingleSet().end())
            {
                //add only the hits which are actually on the requested layers
                trackerRecHits.push_back(std::move(currentTrackerHit));
            }
        }

        if ( layersCrossed < minLayersCrossed)
        {
            continue;
        }

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

        std::vector<unsigned int> seedHitNumbers = iterateHits(0,trackerRecHits,hitIndicesInTree,true);

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

            const GeomDet* initialLayer = trackerGeometry->idToDet( recHits.back().geographicalId() );
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
            initialState = PTrajectoryStateOnDet( initialTSOS.localParameters(),initialTSOS.globalMomentum().perp(),localErrors, recHits.back().geographicalId().rawId(), surfaceSide);
            output->push_back(TrajectorySeed(initialState, recHits, PropagationDirection::alongMomentum));

        }
    } //end loop over simtracks
    

    e.put(output);
}



bool
TrajectorySeedProducer::compatibleWithBeamSpot(
        const GlobalPoint& gpos1,
        const GlobalPoint& gpos2,
        double error,
        bool forward
    ) const
{

    // The hits 1 and 2 positions, in HepLorentzVector's
    XYZTLorentzVector thePos1(gpos1.x(),gpos1.y(),gpos1.z(),0.);
    XYZTLorentzVector thePos2(gpos2.x(),gpos2.y(),gpos2.z(),0.);

    // create a particle with following properties
    //  - charge = +1
    //  - vertex at second rechit
    //  - momentum direction: from first to second rechit
    //  - magnitude of momentum: nonsense (distance between 1st and 2nd rechit)  
    ParticlePropagator myPart(thePos2 - thePos1,thePos2,1.,magneticFieldMap);

    /*
    propagateToBeamCylinder does the following
       - check there exists a track through the 2 hits and through a
    cylinder with radius "originRadius" centered around the CMS axis
       - if such tracks exists, pick the one with maximum pt
       - track vertex z coordinate is z coordinate of closest approach of
    track to (x,y) = (0,0)
       - the particle gets the charge that allows the highest pt
    */
    if (originRadius>0)
    {
        bool intersect = myPart.propagateToBeamCylinder(thePos1,originRadius*1.);
        if ( !intersect )
        {
            return false;
        }
    }

    // Check if the constraints are satisfied
    // 1. pT at cylinder with radius originRadius
    if ((ptMin>0) && ( myPart.Pt() < ptMin ))
    {
        return false;
    }
    // 2. Z compatible with beam spot size 
    // in constuctur only one of originHalfLength,nSigmaZ is allowed to be >= 0
    double zConstraint = std::max(originHalfLength,beamSpot->sigmaZ()*nSigmaZ);
    if ((zConstraint>0) && ( fabs(myPart.Z()-beamSpot->position().Z()) > zConstraint ))
    {
        return false;
    }
    return true;
}
//this fucntion is currently poorly understood and needs clearer comments in the future
bool
TrajectorySeedProducer::compatibleWithPrimaryVertex(
        const GlobalPoint& gpos1, 
        const GlobalPoint& gpos2,
        double error,
        bool forward
    ) const 
{

    unsigned int nVertices = primaryVertices->size();
    if ( nVertices==0 || ((originHalfLength < 0.0) && (nSigmaZ < 0.0)))
    {
        return true;
    }

    // Loop on primary vertices
    for ( unsigned iv=0; iv<nVertices; ++iv ) 
    { 
        // Z position of the primary vertex
        const reco::Vertex& vertex = (*primaryVertices)[iv];

        double xV = vertex.x();
        double yV = vertex.y();
        double zV = vertex.z();

        // Radii of the two hits with respect to the vertex position
        double R1 = std::sqrt ( (gpos1.x()-xV)*(gpos1.x()-xV) + (gpos1.y()-yV)*(gpos1.y()-yV) );
        double R2 = std::sqrt ( (gpos2.x()-xV)*(gpos2.x()-xV) + (gpos2.y()-yV)*(gpos2.y()-yV) );

        double zConstraint = std::max(originHalfLength,vertex.zError()*nSigmaZ);
        //inner hit must be within a sort of pyramid using
        //the outer hit and the cylinder around the PV
        double checkRZ1 = forward ?
        (gpos1.z()-zV+zConstraint) / (gpos2.z()-zV+zConstraint) * R2 :
        -zConstraint + R1/R2*(gpos2.z()-zV+zConstraint);
        double checkRZ2 = forward ?
        (gpos1.z()-zV-zConstraint)/(gpos2.z()-zV-zConstraint) * R2 :
        +zConstraint + R1/R2*(gpos2.z()-zV-zConstraint);
        double checkRZmin = std::min(checkRZ1,checkRZ2)-3.*error;
        double checkRZmax = std::max(checkRZ1,checkRZ2)+3.*error;
        // Check if the innerhit is within bounds
        bool compatible = forward ?
        checkRZmin < R1 && R1 < checkRZmax : 
        checkRZmin < gpos1.z()-zV && gpos1.z()-zV < checkRZmax;
        // If it is, just return ok
        if ( compatible )
        {
            return compatible;
        }
    }
    // Otherwise, return not ok
    return false;

}  

