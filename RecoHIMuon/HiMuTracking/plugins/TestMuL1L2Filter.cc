// -*- C++ -*-
//
// Package:    TestMuL1L2
// Class:      TestMuL1L2
//
// \class TestMuL1L2 TestMuL1L2.cc TestMuonL1L2/TestMuL1L2/src/TestMuL1L2.cc
//
// Original Author:  Dong Ho Moon
//         Created:  Wed May  9 06:22:36 CEST 2007
// $Id: TestMuL1L2Filter.cc,v 1.1 2008/07/14 16:59:28 kodolova Exp $
//
//
// Comment: Dimuon reconstruction need primary vertex
// For the moment it is taking from SimVertex
//
 
#include "RecoHIMuon/HiMuTracking/plugins/TestMuL1L2Filter.h" 

#include <memory>

// C++ Headers

#include<iostream>
#include<iomanip>
#include<vector>
#include<cmath>

//user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkNavigation/interface/NavigationSchoolFactory.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryFiltering/interface/RegionalTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MinPtTrajectoryFilter.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoHIMuon/HiMuSeed/interface/HICFTSfromL1orL2.h"
#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"
#include "RecoHIMuon/HiMuPropagator/interface/FastMuPropagator.h"
#include "RecoHIMuon/HiMuPropagator/interface/FmpConst.h"
#include "RecoHIMuon/HiMuPropagator/interface/HICTkOuterStartingLayerFinder.h"
#include "RecoHIMuon/HiMuSeed/interface/DiMuonSeedGeneratorHIC.h"
#include "RecoHIMuon/HiMuTracking/interface/HICTrajectoryBuilder.h"
#include "RecoHIMuon/HiMuTracking/interface/HICMuonUpdator.h"
#include "RecoHIMuon/HiMuTracking/interface/HICMeasurementEstimator.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"

//Constructor

using namespace reco;
using namespace std;

namespace cms{
TestMuL1L2Filter::TestMuL1L2Filter(const edm::ParameterSet& ps1)
{
   std::cout<<" Start TestMuL1L2 constructor "<<std::endl;
   pset_ = ps1;

}

void TestMuL1L2Filter::beginJob(const edm::EventSetup& es1)
{
   theHICConst = new HICConst();
   theFmpConst = new FmpConst();
   theTrackVertexMaker = new HITrackVertexMaker(pset_,es1);
   edm::ESHandle<GeometricDet>  geom;
   es1.get<IdealGeometryRecord>().get(geom);
   trackerg= TrackerGeomBuilderFromGeometricDet().build(&*geom);
   
}
void TestMuL1L2Filter::endJob()
{
   delete theHICConst;
   delete theFmpConst;
   delete theTrackVertexMaker;
   
}


//Destructor

//void TestMuL1L2::endJob()
TestMuL1L2Filter::~TestMuL1L2Filter()
{
   std::cout<<" End constructor "<<std::endl;  
} 

bool TestMuL1L2Filter::filter(edm::Event& e1, const edm::EventSetup& es1)
{

cout<<"TESTING THE NEW CODE"<<endl;

runno = e1.id().run();
edm::LogVerbatim("GMTDump")<<"run :"<<runno<<endl;
cout<<"runno "<<runno<<endl;

int igmt1 = 0;

/*
  std::vector<edm::Provenance const*> theProvenance;
  e1.getAllProvenance(theProvenance);
  for( std::vector<edm::Provenance const*>::const_iterator ip = theProvenance.begin();
                                                      ip != theProvenance.end(); ip++)
  {
     cout<<" Print all module/label names "<<(**ip).moduleName()<<" "<<(**ip).moduleLabel()<<
     " "<<(**ip).productInstanceName()<<endl;
  }
*/
//
// HepMC is extracted
//
/*
  const HepMC::GenEvent* Evt;
  edm::Handle< edm::HepMCProduct >  EvtHandles ;
  e1.getByType( EvtHandles ) ;
   int iplus = 0;
   int iminus = 0;
  if (!EvtHandles.isValid()) {*EvtHandles;} else
  {
      Evt = EvtHandles->GetEvent() ;
// Cycle on particle conatiner
   for ( HepMC::GenEvent::particle_const_iterator p = Evt->particles_begin();   p != Evt->particles_end(); ++p ) {
    std::cout<<" Generator particle "<<(*p)->pdg_id()<<" pt "<<(*p)->momentum().perp()<<" "<<(*p)->momentum().eta()<<
    std::endl;
     if(fabs((*p)->momentum().eta())<2.4&&(*p)->pdg_id()==13) iminus = 1;
     if(fabs((*p)->momentum().eta())<2.4&&(*p)->pdg_id()==-13) iplus = 1; 
   }
  }
  if( iplus == 1 && iminus == 1) std::cout<<" Upsilon in eta acceptance "<<std::endl;
*/
//  if( iplus == 0 || iminus == 0 ) return;
//
// Get Global geometry for Muons
//
/*
   edm::ESHandle<GlobalTrackingGeometry> geometry;
   es1.get<GlobalTrackingGeometryRecord>().get(geometry);
   if ( ! geometry.isValid() )
     throw cms::Exception("FatalError") << "Unable to find GlobalTrackingGeometryRecord in event!\n";

//
// Get list of SimHits
//
   iminus = 0;
   iplus = 0;
   edm::Handle<edm::PSimHitContainer> simHitsCSC;
   e1.getByLabel<edm::PSimHitContainer>("g4SimHits","MuonCSCHits",simHitsCSC);
   if (! simHitsCSC.isValid() ) { LogDebug("") << "No simhitsCSC found"<<std::endl;}
   else
   {
//   std::cout<<" Size of simhitsCSC "<<simHitsCSC->size()<<std::endl;
   for(edm::PSimHitContainer::const_iterator it=simHitsCSC->begin(); it != simHitsCSC->end(); it++ )
   {
        if( abs((*it).particleType()) == 13 ) {
           if((*it).particleType() == 13) iminus = 1;
           if((*it).particleType() == -13) iplus = 1; 
           const GeomDet* simUnitGeometry = geometry->idToDet( DetId((*it).detUnitId()) );
           
//           std::cout<<"Entry point CSC"<<(*it).particleType()<<" "<<(*it).entryPoint()<<std::endl;
            std::cout<<"Entry point CSC"<<(*it).particleType()<<" "<<simUnitGeometry->toGlobal( (*it).entryPoint()  )
            <<std::endl; 
         }
        //cout<<" Vertex position "<<vertex.position().rho()<<" "<<vertex.position().z()<<endl;
   }
   }
   edm::Handle<edm::PSimHitContainer> simHitsDT;
   e1.getByLabel<edm::PSimHitContainer>("g4SimHits","MuonDTHits",simHitsCSC);
   if (! simHitsDT.isValid() ) { LogDebug("") << "No simhitsDT found"<<std::endl;}
   else
   {
//   std::cout<<" Size of simhitsDT "<<simHitsDT->size()<<std::endl;
   for(edm::PSimHitContainer::const_iterator it=simHitsDT->begin(); it != simHitsDT->end(); it++ )
   {
        if( abs((*it).particleType()) == 13 ) {
           if((*it).particleType() == 13) iminus = 1; 
           if((*it).particleType() == -13) iplus = 1;
//           std::cout<<"Entry point DT"<<(*it).particleType()<<" "<<(*it).entryPoint()<<std::endl;
            const GeomDet* simUnitGeometry = geometry->idToDet( DetId((*it).detUnitId()) );
            std::cout<<"Entry point DT"<<(*it).particleType()<<" "<<simUnitGeometry->toGlobal( (*it).entryPoint()  )
            <<std::endl;

         }
        //cout<<" Vertex position "<<vertex.position().rho()<<" "<<vertex.position().z()<<endl;
   }
   }
*/
//
// If no simhit in CSC or DT from both muons return. Only for testing.
//
//  if(iminus == 0 || iplus == 0) return;


//  const TrackingGeometry::DetIdContainer detc=  trackerg->detIds();
//   std::cout<<" Number of DetIds in Trackergeometry "<<detc.size()<<std::endl;
//  for (TrackingGeometry::DetIdContainer::const_iterator id = detc.begin(); id != detc.end(); id++)
//  {
//    std::cout<<" DetId container "<<(*id).rawId()<<std::endl;
//  } 

//
// Get tracker
//
/*
   edm::Handle<TrackingParticleCollection> tracktruth;
   e1.getByLabel<TrackingParticleCollection>( "trackingtruthprod",tracktruth);
   if(!tracktruth.isValid()) {LogDebug("") << "No TrackTruth found"<<std::endl;}
   for(TrackingParticleCollection::const_iterator it=tracktruth->begin(); it != tracktruth->end(); it++ )
   {
     if(abs((*it).pdgId())==13)
     {
     //std::cout<<" True Particle "<<(*it).pdgId()<<" Size of psimhits "<<(*it).trackPSimHit().size()<<std::endl;
     int igen = 0;
     for(TrackingParticle::genp_iterator ip = (*it).genParticle_begin(); ip != (*it).genParticle_end(); ip++)
     {
       std::cout<<" Gen particle "<<(**ip).pdg_id()<<std::endl;
       igen = 1;
     }
      
        if(igen == 1 )
        {
          for(std::vector<PSimHit>::const_iterator is= (*it).pSimHit_begin(); is != (*it).pSimHit_end(); is++)
          {
            int detid = (*is).detUnitId();
            DetId detId = DetId(detid);

//            unsigned int subdetId = static_cast<unsigned int>(detId.subdetId());
//           std::cout<<" Detunit "<<detid<<" "<<detId.rawId()<<std::endl;
//           const DetLayer* theDetLayer = tracker->detLayer(detId); 
           const GeomDet* det = trackerg->idToDet(detId);
           GlobalPoint tt = det->surface().toGlobal((*is).localPosition());
           std::cout<<" Position : "<<tt.perp()<<" "<<tt.phi()<<" "<<tt.z()<<" detid "<<detId.rawId()<<" detector position "<<det->position().perp()<<
	   " "<<det->position().phi()<<" "<<det->position().z()<<std::endl;
          
          }
        }
     }
   }
*/
//
// SimTracks are extracted
//

   // get list of tracks and their vertices
   edm::Handle<edm::SimTrackContainer> simTracks;
   e1.getByType<edm::SimTrackContainer>(simTracks);
//
// Get SimVertex till Primary RecoVertex is availbale 
//

   edm::Handle<edm::SimVertexContainer> simVertices;
   e1.getByType<edm::SimVertexContainer>(simVertices);
   if (! simVertices.isValid() ) throw cms::Exception("FatalError") << "No vertices found\n";
   int inum = 0;
   double zvert = 0.;

   for(edm::SimVertexContainer::const_iterator it=simVertices->begin(); it != simVertices->end(); it++ )
   {
        
        SimVertex vertex = (*it);
        if(inum == 0) { theHICConst->setVertex(vertex.position().z()); theFmpConst->setVertex(vertex.position().z()); break; }
        inum++;  
        //cout<<" Vertex position "<<vertex.position().rho()<<" "<<vertex.position().z()<<endl;
   }

   cout << " Vertex is set to "<<theHICConst->zvert<<endl;
    
//
// Cycle on SimTrack Container
//
/*
   for(edm::SimTrackContainer::const_iterator tracksCI = simTracks->begin(); 
       tracksCI != simTracks->end(); tracksCI++){
      // skip low Pt tracks
      if (tracksCI->momentum().pt() < 1.) {
         continue;
      }
        int partIndex = tracksCI->genpartIndex();

//        std::cout<<" PartIndex "<<partIndex<<" "<<Evt->particles_size()<<std::endl;

        if(partIndex>=0 && partIndex<Evt->particles_size())
        {
             HepMC::GenParticle* particle = Evt->barcode_to_particle(partIndex);
       //   if(abs(particle->pdg_id()) == 13 ) std::cout<<" PartID "<<particle->pdg_id()<<" pthepmc "<<particle->momentum().perp()<<" pttrack "<<
         // tracksCI->momentum().pt()<<" eta "<<particle->momentum().eta()<<std::endl;
        }

        int vertexIndex = tracksCI->vertIndex();
      // uint trackIndex = tracksCI->genpartIndex();
      
      SimVertex vertex(Hep3Vector(0.,0.,0.),0);
      if (vertexIndex >= 0) vertex = (*simVertices)[vertexIndex];
      
      // skip tracks originated away from the IP
      if (vertex.position().rho() > 50) {
         continue;
      }
   }
*/
// Start track finder

   bool dimuon = theTrackVertexMaker->produceTracks(e1,es1,theHICConst,theFmpConst);
   if(dimuon) cout<<" The vertex is found : "<<endl; 
   return dimuon;
} 
//define the plugin
}
//DEFINE_FWK_MODULE(TestMuL1L2);

