/**
 *  See header file for a description of this class.
 *  
 *  \author Shih-Chuan Kao, Dominique Fortin - UCR
 */


#include <RecoMuon/MuonSeedGenerator/src/MuonSeedBuilder.h>
#include <RecoMuon/MuonSeedGenerator/src/MuonSeedCreator.h>

// Data Formats 
#include <DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h>
#include <DataFormats/TrajectorySeed/interface/TrajectorySeed.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4D.h>

// Geometry
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <TrackingTools/DetLayers/interface/DetLayer.h>
#include <RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h>
#include <RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h>
#include <RecoMuon/Records/interface/MuonRecoGeometryRecord.h>

// muon service
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h>
#include <DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h>

// Framework
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/Framework/interface/Event.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 
#include <DataFormats/Common/interface/Handle.h>

// C++
#include <vector>
#include <deque>
#include <utility>

//typedef std::pair<double, TrajectorySeed> seedpr ;
//static bool ptDecreasing(const seedpr s1, const seedpr s2) { return ( s1.first > s2.first ); }
static bool lengthSorting(const TrajectorySeed s1, const TrajectorySeed s2) { return ( s1.nHits() > s2.nHits() ); }

/*
 * Constructor
 */
MuonSeedBuilder::MuonSeedBuilder(const edm::ParameterSet& pset){

  // Local Debug flag
  debug                = pset.getParameter<bool>("DebugMuonSeed");

  // enable the DT chamber
  enableDTMeasurement  = pset.getParameter<bool>("EnableDTMeasurement");
  theDTSegmentLabel    = pset.getParameter<edm::InputTag>("DTSegmentLabel");

  // enable the CSC chamber
  enableCSCMeasurement = pset.getParameter<bool>("EnableCSCMeasurement");
  theCSCSegmentLabel   = pset.getParameter<edm::InputTag>("CSCSegmentLabel");

  // Parameters for seed creation in endcap region
  minCSCHitsPerSegment = pset.getParameter<int>("minCSCHitsPerSegment");
  maxDeltaEtaCSC       = pset.getParameter<double>("maxDeltaEtaCSC");
  maxDeltaPhiCSC       = pset.getParameter<double>("maxDeltaPhiCSC");

  // Parameters for seed creation in overlap region
  maxDeltaEtaOverlap   = pset.getParameter<double>("maxDeltaEtaOverlap");
  maxDeltaPhiOverlap   = pset.getParameter<double>("maxDeltaPhiOverlap");

  // Parameters for seed creation in barrel region
  minDTHitsPerSegment  = pset.getParameter<int>("minDTHitsPerSegment");
  maxDeltaEtaDT        = pset.getParameter<double>("maxDeltaEtaDT");
  maxDeltaPhiDT        = pset.getParameter<double>("maxDeltaPhiDT");

  // Parameters to suppress combinatorics (ghosts)
  maxEtaResolutionDT   = pset.getParameter<double>("maxEtaResolutionDT"); 
  maxPhiResolutionDT   = pset.getParameter<double>("maxPhiResolutionDT"); 
  maxEtaResolutionCSC  = pset.getParameter<double>("maxEtaResolutionCSC"); 
  maxPhiResolutionCSC  = pset.getParameter<double>("maxPhiResolutionCSC"); 

  // muon service
  edm::ParameterSet serviceParameters = pset.getParameter<edm::ParameterSet>("ServiceParameters");
  theService        = new MuonServiceProxy(serviceParameters);

  // Class for forming muon seeds
  muonSeedCreate_      = new MuonSeedCreator( pset ); 

}

/*
 * Destructor
 */
MuonSeedBuilder::~MuonSeedBuilder(){

  delete muonSeedCreate_;
  if (theService) delete theService;
}

/* 
 * build
 *
 * Where the segments are sorted out to make a protoTrack (vector of matching segments in different 
 * stations/layers).  The protoTrack is then passed to the seed creator to create CSC, DT or overlap seeds
 *
 */
//void MuonSeedBuilder::build( MuonDetLayerMeasurements muonMeasurements, TrajectorySeedCollection& theSeeds ) {
int MuonSeedBuilder::build( edm::Event& event, const edm::EventSetup& eventSetup, TrajectorySeedCollection& theSeeds ) {


  // Pass the Magnetic Field to where the seed is create
  muonSeedCreate_->setBField(BField);

  // Create temporary collection of seeds which will be cleaned up to remove combinatorics
  std::vector<TrajectorySeed> rawSeeds;
  std::vector<float> etaOfSeed;
  std::vector<float> phiOfSeed;
  std::vector<int> nSegOnSeed;

 // Instantiate the accessor (get the segments: DT + CSC but not RPC=false)
  MuonDetLayerMeasurements muonMeasurements(theDTSegmentLabel,theCSCSegmentLabel,edm::InputTag(),
                                            enableDTMeasurement,enableCSCMeasurement,false);

 // Instantiate the accessor (get the segments: DT + CSC but not RPC=false)
 // MuonDetLayerMeasurements muonMeasurements(enableDTMeasurement,enableCSCMeasurement,false,
 //                                          theDTSegmentLabel.label(),theCSCSegmentLabel.label());


  // 1) Get the various stations and store segments in containers for each station (layers)
 
  // 1a. get the DT segments by stations (layers):
  std::vector<DetLayer*> dtLayers = muonLayers->allDTLayers();
 
  SegmentContainer DTlist4 = muonMeasurements.recHits( dtLayers[3], event );
  SegmentContainer DTlist3 = muonMeasurements.recHits( dtLayers[2], event );
  SegmentContainer DTlist2 = muonMeasurements.recHits( dtLayers[1], event );
  SegmentContainer DTlist1 = muonMeasurements.recHits( dtLayers[0], event );

  // Initialize flags that a given segment has been allocated to a seed
  BoolContainer usedDTlist4(DTlist4.size(), false);
  BoolContainer usedDTlist3(DTlist3.size(), false);
  BoolContainer usedDTlist2(DTlist2.size(), false);
  BoolContainer usedDTlist1(DTlist1.size(), false);

  if (debug) {
    std::cout << "*** Number of DT segments is: " << DTlist4.size()+DTlist3.size()+DTlist2.size()+DTlist1.size() << std::endl;
    std::cout << "In MB1: " << DTlist1.size() << std::endl;
    std::cout << "In MB2: " << DTlist2.size() << std::endl;
    std::cout << "In MB3: " << DTlist3.size() << std::endl;
    std::cout << "In MB4: " << DTlist4.size() << std::endl;
  }

  // 1b. get the CSC segments by stations (layers):
  // 1b.1 Global z < 0
  std::vector<DetLayer*> cscBackwardLayers = muonLayers->backwardCSCLayers();    
  SegmentContainer CSClist4B = muonMeasurements.recHits( cscBackwardLayers[4], event );
  SegmentContainer CSClist3B = muonMeasurements.recHits( cscBackwardLayers[3], event );
  SegmentContainer CSClist2B = muonMeasurements.recHits( cscBackwardLayers[2], event );
  SegmentContainer CSClist1B = muonMeasurements.recHits( cscBackwardLayers[1], event ); // ME1/2 and 1/3
  SegmentContainer CSClist0B = muonMeasurements.recHits( cscBackwardLayers[0], event ); // ME11

  BoolContainer usedCSClist4B(CSClist4B.size(), false);
  BoolContainer usedCSClist3B(CSClist3B.size(), false);
  BoolContainer usedCSClist2B(CSClist2B.size(), false);
  BoolContainer usedCSClist1B(CSClist1B.size(), false);
  BoolContainer usedCSClist0B(CSClist0B.size(), false);

  // 1b.2 Global z > 0
  std::vector<DetLayer*> cscForwardLayers = muonLayers->forwardCSCLayers();
  SegmentContainer CSClist4F = muonMeasurements.recHits( cscForwardLayers[4], event );
  SegmentContainer CSClist3F = muonMeasurements.recHits( cscForwardLayers[3], event );
  SegmentContainer CSClist2F = muonMeasurements.recHits( cscForwardLayers[2], event );
  SegmentContainer CSClist1F = muonMeasurements.recHits( cscForwardLayers[1], event );
  SegmentContainer CSClist0F = muonMeasurements.recHits( cscForwardLayers[0], event );

  BoolContainer usedCSClist4F(CSClist4F.size(), false);
  BoolContainer usedCSClist3F(CSClist3F.size(), false);
  BoolContainer usedCSClist2F(CSClist2F.size(), false);
  BoolContainer usedCSClist1F(CSClist1F.size(), false);
  BoolContainer usedCSClist0F(CSClist0F.size(), false);

  if (debug) {
    std::cout << "*** Number of CSC segments is " << CSClist4F.size()+CSClist3F.size()+CSClist2F.size()+CSClist1F.size()+CSClist0F.size()+CSClist4B.size()+CSClist3B.size()+CSClist2B.size()+CSClist1B.size()+CSClist0B.size()<< std::endl;
    std::cout << "In ME11: " << CSClist0F.size()+CSClist0B.size() << std::endl;
    std::cout << "In ME12: " << CSClist1F.size()+CSClist1B.size() << std::endl;
    std::cout << "In ME2 : " << CSClist2F.size()+CSClist2B.size() << std::endl;
    std::cout << "In ME3 : " << CSClist3F.size()+CSClist3B.size() << std::endl;
    std::cout << "In ME4 : " << CSClist4F.size()+CSClist4B.size() << std::endl;
  }


  /* ******************************************************************************************************************
   * Form seeds in barrel region
   *
   * Proceed from inside -> out
   * ******************************************************************************************************************/


  // Loop over all possible MB1 segment to form seeds:
  int index = -1;
  for (SegmentContainer::iterator it = DTlist1.begin(); it != DTlist1.end(); ++it ){

    index++;

    if (usedDTlist1[index] == true) continue;
    if ( int ((*it)->recHits().size()) < minDTHitsPerSegment ) continue;
    if ((*it)->dimension() != 4) continue;

    //double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    //if ( ((*it)->chi2()/dof) > 20000.0 ) continue;
    
    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    bool showeringBefore = false;
    NShowerSeg = 0; 
    if ( IdentifyShowering( DTlist1, usedDTlist1, eta_temp, phi_temp, -1, NShowerSeg )  ) showeringBefore = true ;
    int NShowers = 0; 
    if ( showeringBefore ) {
        //usedDTlist1[index] = true;
        NShowers++ ;
    }
     
    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(-1);

    // Try adding segment from other stations
    if (foundMatchingSegment(3, protoTrack, DTlist2, usedDTlist2, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(-2);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(3, protoTrack, DTlist3, usedDTlist3, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(-3);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(3, protoTrack, DTlist4, usedDTlist4, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(-4);
    if ( showeringBefore )  NShowers++ ;

    // Check if in overlap region
    if (eta_temp > 0.8) {
      if (foundMatchingSegment(2, protoTrack, CSClist1F, usedCSClist1F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(1);
      if ( showeringBefore )  NShowers++ ;
      if (foundMatchingSegment(2, protoTrack, CSClist2F, usedCSClist2F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(2);
      if ( showeringBefore )  NShowers++ ;
      if (foundMatchingSegment(2, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(3);
      if ( showeringBefore )  NShowers++ ;
    } 
    else if (eta_temp < -0.8) {
      if (foundMatchingSegment(2, protoTrack, CSClist1B, usedCSClist1B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(1);
      if ( showeringBefore )  NShowers++ ;
      if (foundMatchingSegment(2, protoTrack, CSClist2B, usedCSClist2B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(2);
      if ( showeringBefore )  NShowers++ ;
      if (foundMatchingSegment(2, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(3);
      if ( showeringBefore )  NShowers++ ;
    }
 
    unsigned nLayers = layers.size();

    // adding showering information   
    if ( nLayers < 2 && ShoweringSegments.size() > 0 ) {
       for (size_t i=0; i< ShoweringSegments.size(); i++) {
           protoTrack.push_back( ShoweringSegments[i] );
           layers.push_back( ShoweringLayers[i] );
       }
       ShoweringSegments.clear() ;  
       ShoweringLayers.clear() ;  
    }

    TrajectorySeed thisSeed;

    if ( nLayers < 2 ) {
        thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, NShowers, NShowerSeg );
    } else {
      if ( layers[nLayers-1] > 0 ) {
        thisSeed = muonSeedCreate_->createSeed(2, protoTrack, layers, NShowers, NShowerSeg );
      } else {
        thisSeed = muonSeedCreate_->createSeed(3, protoTrack, layers, NShowers, NShowerSeg ); 
      }
    }
    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed); 
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( protoTrack.size() );

    // Marked segment as used
    usedDTlist1[index] = true;
  }


  // Loop over all possible MB2 segment to form seeds:
  index = -1;
  for (SegmentContainer::iterator it = DTlist2.begin(); it != DTlist2.end(); ++it ){

    index++;

    if (usedDTlist2[index] == true) continue;
    if ( int ((*it)->recHits().size()) < minDTHitsPerSegment ) continue;  
    if ((*it)->dimension() != 4) continue;

    //double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    //if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    bool showeringBefore = false; 
    NShowerSeg = 0; 
    if ( IdentifyShowering( DTlist2, usedDTlist2, eta_temp, phi_temp, -2, NShowerSeg)  ) showeringBefore = true ;
    int NShowers = 0; 
    if ( showeringBefore ) {
       // usedDTlist2[index] = true;
        NShowers++ ;
    }

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(-2);

 
    // Try adding segment from other stations
    if (foundMatchingSegment(3, protoTrack, DTlist3, usedDTlist3, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(-3);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(3, protoTrack, DTlist4, usedDTlist4, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(-4);
    if ( showeringBefore )  NShowers++ ;

    // Check if in overlap region
    if (eta_temp > 0.8) {
      if (foundMatchingSegment(2, protoTrack, CSClist1F, usedCSClist1F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(1);
      if ( showeringBefore )  NShowers++ ;
      if (foundMatchingSegment(2, protoTrack, CSClist2F, usedCSClist2F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(2);
      if ( showeringBefore )  NShowers++ ;
      if (foundMatchingSegment(2, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(3);
      if ( showeringBefore )  NShowers++ ;
    }
    else if (eta_temp < -0.8) {
      if (foundMatchingSegment(2, protoTrack, CSClist1B, usedCSClist1B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(1);
      if ( showeringBefore )  NShowers++ ;
      if (foundMatchingSegment(2, protoTrack, CSClist2B, usedCSClist2B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(2);
      if ( showeringBefore )  NShowers++ ;
      if (foundMatchingSegment(2, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(3);
      if ( showeringBefore )  NShowers++ ;
    }

    unsigned nLayers = layers.size();
    // adding showering information   
    if ( nLayers < 2 && ShoweringSegments.size() > 0 ) {
       for (size_t i=0; i< ShoweringSegments.size(); i++) {
           protoTrack.push_back( ShoweringSegments[i] );
           layers.push_back( ShoweringLayers[i] );
       }
       ShoweringSegments.clear() ;  
       ShoweringLayers.clear() ;  
    }

  
    //if ( nLayers < 2 ) continue;
    TrajectorySeed thisSeed;

    if ( nLayers < 2 ) {
        thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, NShowers, NShowerSeg );
    } else {
      if ( layers[nLayers-1] > 0 ) {
        thisSeed = muonSeedCreate_->createSeed(2, protoTrack, layers, NShowers, NShowerSeg );
      } else {
        thisSeed = muonSeedCreate_->createSeed(3, protoTrack, layers, NShowers, NShowerSeg ); 
      }
    }
    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed); 
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( protoTrack.size() );

    // Marked segment as used
    usedDTlist2[index] = true;
  }


  // Loop over all possible MB3 segment to form seeds:
  index = -1;
  for (SegmentContainer::iterator it = DTlist3.begin(); it != DTlist3.end(); ++it ){

    index++;

    if (usedDTlist3[index] == true) continue;
    if ( int ((*it)->recHits().size()) < minDTHitsPerSegment ) continue;  
    if ((*it)->dimension() != 4) continue;

    //double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    //if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    bool showeringBefore = false; 
    NShowerSeg = 0; 
    if ( IdentifyShowering( DTlist3, usedDTlist3, eta_temp, phi_temp, -3, NShowerSeg)  ) showeringBefore = true ;
    int NShowers = 0; 
    if ( showeringBefore ) {
       // usedDTlist3[index] = true;
        NShowers++ ;
    }

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(-3);
 
    // Try adding segment from other stations
    if (foundMatchingSegment(3, protoTrack, DTlist4, usedDTlist4, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(-4);
    if ( showeringBefore )  NShowers++ ;

    // Check if in overlap region
    if (eta_temp > 0.8) {
      if (foundMatchingSegment(2, protoTrack, CSClist1F, usedCSClist1F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(1);
      if ( showeringBefore )  NShowers++ ;
      if (foundMatchingSegment(2, protoTrack, CSClist2F, usedCSClist2F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(2);
      if ( showeringBefore )  NShowers++ ;
      if (foundMatchingSegment(2, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(3);
      if ( showeringBefore )  NShowers++ ;
    }
    else if (eta_temp < -0.8) {
      if (foundMatchingSegment(2, protoTrack, CSClist1B, usedCSClist1B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(1);
      if ( showeringBefore )  NShowers++ ;
      if (foundMatchingSegment(2, protoTrack, CSClist2B, usedCSClist2B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(2);
      if ( showeringBefore )  NShowers++ ;
      if (foundMatchingSegment(2, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(3);
      if ( showeringBefore )  NShowers++ ;
    }
    
    unsigned nLayers = layers.size();
    // adding showering information   
    if ( nLayers < 2 && ShoweringSegments.size() > 0 ) {
       for (size_t i=0; i< ShoweringSegments.size(); i++) {
           protoTrack.push_back( ShoweringSegments[i] );
           layers.push_back( ShoweringLayers[i] );
       }
       ShoweringSegments.clear() ;  
       ShoweringLayers.clear() ;  
    }

    
    TrajectorySeed thisSeed;
    if ( nLayers < 2 ) {
        thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, NShowers, NShowerSeg );
    }else {
      if ( layers[nLayers-1] > 0 ) {
        thisSeed = muonSeedCreate_->createSeed(2, protoTrack, layers, NShowers, NShowerSeg );
      } else {
        thisSeed = muonSeedCreate_->createSeed(3, protoTrack, layers, NShowers, NShowerSeg ); 
      }
    }

    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed); 
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( protoTrack.size() );

    // Marked segment as used
    usedDTlist3[index] = true;
  }

  /* *********************************************************************************************************************
   * Form seeds from backward endcap
   *
   * Proceed from inside -> out
   * *********************************************************************************************************************/

  // Loop over all possible ME11 segment to form seeds:
  index = -1;

  for (SegmentContainer::iterator it = CSClist0B.begin(); it != CSClist0B.end(); ++it ){

    index++;

    if (usedCSClist0B[index] == true) continue;
    if ( int ((*it)->recHits().size()) < minCSCHitsPerSegment ) continue;  

    //double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    //if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    bool showeringBefore = false; 
    NShowerSeg = 0; 
    if ( IdentifyShowering( CSClist0B, usedCSClist0B, eta_temp, phi_temp, 0, NShowerSeg)  ) showeringBefore = true ;
    int NShowers = 0; 
    if ( showeringBefore ) {
       // usedCSClist0B[index] = true;
        NShowers++ ;
    }

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(0);

    // Try adding segment from other station
    if (foundMatchingSegment(1, protoTrack, CSClist1B, usedCSClist1B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(1);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(1, protoTrack, CSClist2B, usedCSClist2B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(2);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(1, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(3);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(1, protoTrack, CSClist4B, usedCSClist4B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(4);
    if ( showeringBefore )  NShowers++ ;

    unsigned nLayers = layers.size();

    // adding showering information   
    if ( nLayers < 2 && ShoweringSegments.size() > 0 ) {
       for (size_t i=0; i< ShoweringSegments.size(); i++) {
           protoTrack.push_back( ShoweringSegments[i] );
           layers.push_back( ShoweringLayers[i] );
       }
       ShoweringSegments.clear() ;  
       ShoweringLayers.clear() ;  
    }

    //if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed;
    if ( nLayers < 2 ) {
        thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, NShowers, NShowerSeg );
    }else {
      if ( fabs( gp.eta() ) > 1.7  ) {
        thisSeed = muonSeedCreate_->createSeed(5, protoTrack, layers, NShowers, NShowerSeg );
      } else {
        thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, NShowers, NShowerSeg );
      }
    }

    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( nLayers );

    // mark this segment as used
    usedCSClist0B[index] = true;
  }


  // Loop over all possible ME1/2 ME1/3 segment to form seeds:
  index = -1;
  for (SegmentContainer::iterator it = CSClist1B.begin(); it != CSClist1B.end(); ++it ){

    index++;

    if (usedCSClist1B[index] == true) continue;
    if ( int ((*it)->recHits().size()) < minCSCHitsPerSegment ) continue;  

    //double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    //if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    bool showeringBefore = false;
    NShowerSeg = 0; 
    if ( IdentifyShowering( CSClist1B, usedCSClist1B, eta_temp, phi_temp, 1, NShowerSeg)  ) showeringBefore = true ;
    int NShowers = 0; 
    if ( showeringBefore ) {
    //    usedCSClist1B[index] = true;
        NShowers++ ;
    }

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);
    
    std::vector<int> layers;
    layers.push_back(1);

    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist2B, usedCSClist2B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(2);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(1, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(3);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(1, protoTrack, CSClist4B, usedCSClist4B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(4);
    if ( showeringBefore )  NShowers++ ;

    unsigned nLayers = layers.size();

    // adding showering information   
    if ( nLayers < 2 && ShoweringSegments.size() > 0 ) {
       for (size_t i=0; i< ShoweringSegments.size(); i++) {
           protoTrack.push_back( ShoweringSegments[i] );
           layers.push_back( ShoweringLayers[i] );
       }
       ShoweringSegments.clear() ;  
       ShoweringLayers.clear() ;  
    }

    //if ( nLayers < 2 ) continue;

     TrajectorySeed thisSeed; 
     if ( nLayers < 2) {
       thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, NShowers, NShowerSeg );
     } else {
       thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, NShowers, NShowerSeg );
     }
     // Add the seeds to master collection
     rawSeeds.push_back(thisSeed);
     etaOfSeed.push_back(eta_temp);
     phiOfSeed.push_back(phi_temp);
     nSegOnSeed.push_back( nLayers );

     // mark this segment as used
     usedCSClist1B[index] = true;

  }


  // Loop over all possible ME2 segment to form seeds:
  index = -1;
  for (SegmentContainer::iterator it = CSClist2B.begin(); it != CSClist2B.end(); ++it ){

    index++;

    if (usedCSClist2B[index] == true) continue;
    if ( int ((*it)->recHits().size()) < minCSCHitsPerSegment ) continue;  

    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    bool showeringBefore = false; 
    NShowerSeg = 0; 
    if ( IdentifyShowering( CSClist2B, usedCSClist2B, eta_temp, phi_temp, 2, NShowerSeg)  ) showeringBefore = true ;
    int NShowers = 0; 
    if ( showeringBefore ) {
       // usedCSClist2B[index] = true;
        NShowers++ ;
    }

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(2);
    
    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(3);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(1, protoTrack, CSClist4B, usedCSClist4B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(4);
    if ( showeringBefore )  NShowers++ ;

    unsigned nLayers = layers.size();

    // adding showering information   
    if ( nLayers < 2 && ShoweringSegments.size() > 0 ) {
       for (size_t i=0; i< ShoweringSegments.size(); i++) {
           protoTrack.push_back( ShoweringSegments[i] );
           layers.push_back( ShoweringLayers[i] );
       }
       ShoweringSegments.clear() ;  
       ShoweringLayers.clear() ;  
    }

    //if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed; 
    if ( nLayers < 2) {
       thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, NShowers, NShowerSeg );
    } else {
       thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, NShowers, NShowerSeg );
    }

    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( nLayers );
    // mark this segment as used
    usedCSClist2B[index] = true;
  }

  // Loop over all possible ME3 segment to form seeds:
  index = -1;
  for (SegmentContainer::iterator it = CSClist3B.begin(); it != CSClist3B.end(); ++it ){

    index++;

    if (usedCSClist3B[index] == true) continue;
    if ( int ((*it)->recHits().size()) < minCSCHitsPerSegment ) continue;  

    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    bool showeringBefore = false; 
    NShowerSeg = 0; 
    if ( IdentifyShowering( CSClist3B, usedCSClist3B, eta_temp, phi_temp, 3, NShowerSeg)  ) showeringBefore = true ;
    int NShowers = 0; 
    if ( showeringBefore ) {
    //    usedCSClist3B[index] = true;
        NShowers++ ;
    }

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(2);
    
    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist4B, usedCSClist4B, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(4);
    if ( showeringBefore )  NShowers++ ;

    unsigned nLayers = layers.size();

    // adding showering information   
    if ( nLayers < 2 && ShoweringSegments.size() > 0 ) {
       for (size_t i=0; i< ShoweringSegments.size(); i++) {
           protoTrack.push_back( ShoweringSegments[i] );
           layers.push_back( ShoweringLayers[i] );
       }
       ShoweringSegments.clear() ;  
       ShoweringLayers.clear() ;  
    }

    // mark this segment as used
    usedCSClist3B[index] = true;

    if ( nLayers < 2 ) continue;
    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, NShowers, NShowerSeg ); 

    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( nLayers );
  }


  /* *****************************************************************************************************************
   * Form seeds from forward endcap
   *
   * Proceed from inside -> out
   * *****************************************************************************************************************/

  // Loop over all possible ME11 segment to form seeds:
  index = -1;
  for (SegmentContainer::iterator it = CSClist0F.begin(); it != CSClist0F.end(); ++it ){

    index++;
  
    if (usedCSClist0F[index] == true) continue;
    if ( int ((*it)->recHits().size()) < minCSCHitsPerSegment ) continue;
    
    //double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    //if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();  
    float phi_temp = gp.phi();      
    bool showeringBefore = false; 
    NShowerSeg = 0; 
    if ( IdentifyShowering( CSClist0F, usedCSClist0F, eta_temp, phi_temp, 0, NShowerSeg)  ) showeringBefore = true ;
    int NShowers = 0; 
    if ( showeringBefore ) {
       // usedCSClist0F[index] = true;
        NShowers++ ;
    }
    
    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(0);

    // Try adding segment from other station
    if (foundMatchingSegment(1, protoTrack, CSClist1F, usedCSClist1F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(1);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(1, protoTrack, CSClist2F, usedCSClist2F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(2);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(1, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(3);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(1, protoTrack, CSClist4F, usedCSClist4F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(4);
    if ( showeringBefore )  NShowers++ ;

    unsigned nLayers = layers.size();

    // adding showering information   
    if ( nLayers < 2 && ShoweringSegments.size() > 0 ) {
       for (size_t i=0; i< ShoweringSegments.size(); i++) {
           protoTrack.push_back( ShoweringSegments[i] );
           layers.push_back( ShoweringLayers[i] );
       }
       ShoweringSegments.clear() ;  
       ShoweringLayers.clear() ;  
    }

    //if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed;
    if ( nLayers < 2 ) {
        thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, NShowers, NShowerSeg );
    } else {
      if ( fabs( gp.eta() ) > 1.7  ) {
        thisSeed = muonSeedCreate_->createSeed(5, protoTrack, layers, NShowers, NShowerSeg );
      } else {
        thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, NShowers, NShowerSeg );
      }
    }
    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( nLayers );

    // mark this segment as used
    usedCSClist0F[index] = true;
  }
  

  // Loop over all possible ME1/2 ME1/3 segment to form seeds:
  index = -1;
  for (SegmentContainer::iterator it = CSClist1F.begin(); it != CSClist1F.end(); ++it ){
    
    index++;
    
    if (usedCSClist1F[index] == true) continue;
    if ( int ((*it)->recHits().size()) < minCSCHitsPerSegment ) continue;
  
    //double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    //if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    bool showeringBefore = false; 
    NShowerSeg = 0; 
    if ( IdentifyShowering( CSClist1F, usedCSClist1F, eta_temp, phi_temp, 1, NShowerSeg)  ) showeringBefore = true ;
    int NShowers = 0; 
    if ( showeringBefore ) {
     //   usedCSClist1F[index] = true;
        NShowers++ ;
    }
    
    SegmentContainer protoTrack;
    protoTrack.push_back(*it);
   
    std::vector<int> layers;
    layers.push_back(1);

    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist2F, usedCSClist2F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(2);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(1, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(3);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(1, protoTrack, CSClist4F, usedCSClist4F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(4);
    if ( showeringBefore )  NShowers++ ;

    unsigned nLayers = layers.size();
  
    // adding showering information   
    if ( nLayers < 2 && ShoweringSegments.size() > 0 ) {
       for (size_t i=0; i< ShoweringSegments.size(); i++) {
           protoTrack.push_back( ShoweringSegments[i] );
           layers.push_back( ShoweringLayers[i] );
       }
       ShoweringSegments.clear() ;  
       ShoweringLayers.clear() ;  
    }

    //if ( nLayers < 2 ) continue; 

    TrajectorySeed thisSeed; 
    if ( nLayers < 2) {
      thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, NShowers, NShowerSeg );
    } else {
      thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, NShowers, NShowerSeg );
    }
  
    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( nLayers );

    // mark this segment as used
    usedCSClist1F[index] = true;
  } 


  // Loop over all possible ME2 segment to form seeds:
  index = -1;
  for (SegmentContainer::iterator it = CSClist2F.begin(); it != CSClist2F.end(); ++it ){
  
    index++;

    if (usedCSClist2F[index] == true) continue;
    if ( int ((*it)->recHits().size()) < minCSCHitsPerSegment ) continue;
  
    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();  
    float phi_temp = gp.phi();   
    bool showeringBefore = false; 
    NShowerSeg = 0; 
    if ( IdentifyShowering( CSClist2F, usedCSClist2F, eta_temp, phi_temp, 2, NShowerSeg)  ) showeringBefore = true ;
    int NShowers = 0; 
    if ( showeringBefore ) {
    //    usedCSClist2F[index] = true;
        NShowers++ ;
    }
   
    SegmentContainer protoTrack;
    protoTrack.push_back(*it);
  
    std::vector<int> layers;
    layers.push_back(2);

    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(3);
    if ( showeringBefore )  NShowers++ ;
    if (foundMatchingSegment(1, protoTrack, CSClist4F, usedCSClist4F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(4);
    if ( showeringBefore )  NShowers++ ;
  
    unsigned nLayers = layers.size();

    // adding showering information   
    if ( nLayers < 2 && ShoweringSegments.size() > 0 ) {
       for (size_t i=0; i< ShoweringSegments.size(); i++) {
           protoTrack.push_back( ShoweringSegments[i] );
           layers.push_back( ShoweringLayers[i] );
       }
       ShoweringSegments.clear() ;  
       ShoweringLayers.clear() ;  
    }

    //if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed; 
    if ( nLayers < 2) {
      thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, NShowers, NShowerSeg );
    } else {
      thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, NShowers, NShowerSeg );
    }
      
    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);  
    etaOfSeed.push_back(eta_temp); 
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( nLayers );
    
    // mark this segment as used
    usedCSClist2F[index] = true;
  }

  // Loop over all possible ME3 segment to form seeds:
  index = -1;
  for (SegmentContainer::iterator it = CSClist3F.begin(); it != CSClist3F.end(); ++it ){
  
    index++;

    if (usedCSClist3F[index] == true) continue;
    if ( int ((*it)->recHits().size()) < minCSCHitsPerSegment ) continue;
  
    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();  
    float phi_temp = gp.phi();   
    bool showeringBefore = false; 
    NShowerSeg = 0; 
    if ( IdentifyShowering( CSClist3F, usedCSClist3F, eta_temp, phi_temp, 3, NShowerSeg)  ) showeringBefore = true ;
    int NShowers = 0; 
    if ( showeringBefore ) {
     //   usedCSClist3F[index] = true;
        NShowers++ ;
    }
   
    SegmentContainer protoTrack;
    protoTrack.push_back(*it);
  
    std::vector<int> layers;
    layers.push_back(2);

    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist4F, usedCSClist4F, eta_temp, phi_temp, layers[layers.size()-1], showeringBefore )) layers.push_back(4);
    if ( showeringBefore )  NShowers++ ;
  
    unsigned nLayers = layers.size();

    // adding showering information   
    if ( nLayers < 2 && ShoweringSegments.size() > 0 ) {
       for (size_t i=0; i< ShoweringSegments.size(); i++) {
           protoTrack.push_back( ShoweringSegments[i] );
           layers.push_back( ShoweringLayers[i] );
       }
       ShoweringSegments.clear() ;  
       ShoweringLayers.clear() ;  
    }

    // mark this segment as used
    usedCSClist3F[index] = true;

    if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, NShowers, NShowerSeg );
      
    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);  
    etaOfSeed.push_back(eta_temp); 
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( nLayers );
    
  }


  /* *********************************************************************************************************************
   * Clean up raw seed collection and pass to master collection
   * *********************************************************************************************************************/

  if (debug) std::cout << "*** CLEAN UP " << std::endl;
  if (debug) std::cout << "Number of seeds BEFORE " << rawSeeds.size() << std::endl;

  //std::cout << " === start cleaning ==== " << rawSeeds.size() << std::endl;

  int goodSeeds = 0;

  theSeeds = seedCleaner(eventSetup,rawSeeds);
  goodSeeds = theSeeds.size();

  if (debug) std::cout << "Number of seeds AFTER " << goodSeeds << std::endl;


  return goodSeeds;
}



/* *********************************************************************************************************************
 * Try to match segment from different station (layer)
 *
 * Note type = 1 --> endcap
 *           = 2 --> overlap
 *           = 3 --> barrel
 * *********************************************************************************************************************/

///                                                   segment for seeding         , segments collection     
bool MuonSeedBuilder::foundMatchingSegment( int type, SegmentContainer& protoTrack, SegmentContainer& segs, 
     BoolContainer& usedSeg, float& eta_last, float& phi_last, int& lastLayer, bool& showeringBefore  ) {

  bool ok = false;
  int scanlayer = (lastLayer < 0 ) ?  (lastLayer-1) : (lastLayer+1) ;

  if ( IdentifyShowering( segs, usedSeg, eta_last, phi_last, scanlayer, NShowerSeg )  ) {
     showeringBefore = true; 
     return ok ;
  }

  // Setup the searching cone-size
  // searching cone-size should changes with bending power
  double maxdEta;
  double maxdPhi;
  if ( type == 1 ) { 
    // CSC
    maxdEta = maxDeltaEtaCSC;
    if ( lastLayer == 0 || lastLayer == 1 ) {
       if ( fabs(eta_last) < 2.1 ) {
          maxdPhi = maxDeltaPhiCSC;
       } else {
          maxdPhi = 0.06;
       } 
    } else if (lastLayer== 2 ) {
       maxdPhi = 0.5*maxDeltaPhiCSC;
    } else  {
       maxdPhi = 0.2*maxDeltaPhiCSC;
    }
  } else if ( type == 2 ) { 
    // Overlap
    maxdEta = maxDeltaEtaOverlap;
    if ( lastLayer == -1 ) {
       maxdPhi = maxDeltaPhiDT;
    } else {
       maxdPhi = maxDeltaPhiOverlap;
    }
  } else {
    // DT
    maxdEta = maxDeltaEtaDT;
    if ( lastLayer == -1 ) {
       maxdPhi = maxDeltaPhiDT;
    } else if ( lastLayer == -2 ) {
       maxdPhi = 0.8*maxDeltaPhiDT;
    } else  {
       maxdPhi = 0.4*maxDeltaPhiDT;
    }
  }
  
  // if previous layer showers, limite the maxdPhi < 0.06
  if ( showeringBefore && maxdPhi > 0.03  ) maxdPhi = 0.03;
  // reset the showering flag
  showeringBefore = false ;

  // global phi/eta from previous segment 
  float eta_temp = eta_last;
  float phi_temp = phi_last;

  // Index counter to keep track of element used in segs 
  int          index = -1;
  int          best_match = index;
  float        best_R = sqrt( (maxdEta*maxdEta) + (maxdPhi*maxdPhi) );
  float        best_chi2 = 200;
  int          best_dimension = 2;
  int          best_nhits = minDTHitsPerSegment;
  if( type == 1 ) best_nhits = minCSCHitsPerSegment;
  // Loop over segments in other station (layer) and find candidate match 
  for (SegmentContainer::iterator it=segs.begin(); it!=segs.end(); ++it){

    index++;

    // Not to get confused:  eta_last is from the previous layer.
    // This is only to find the best set of segments by comparing at the distance layer-by-layer 
    GlobalPoint gp2 = (*it)->globalPosition(); 
    double dh = fabs( gp2.eta() - eta_temp ); 
    double df = fabs( gp2.phi() - phi_temp );
    double dR = sqrt( (dh*dh) + (df*df) );

    // dEta and dPhi should be within certain range
    bool case1 = (  dh  < maxdEta && df < maxdPhi ) ? true:false ;
    // for DT station 4 ; CSCSegment is always 4D 
    bool case2 = (  ((*it)->dimension()!= 4) && (dh< 0.5) && (df < maxdPhi) )  ? true:false ;
    if ( !case1 && !case2  ) continue;
     
    int NRechits = NRecHitsFromSegment( &*(*it) ) ;
    // crapy fucking way to get the fucking number of fucking rechits
    /*
    int NRechits = 0 ; 
    DetId geoId = (*it)->geographicalId();
    if ( geoId.subdetId() == MuonSubdetId::DT ) {
       DTChamberId DT_Id( geoId );
       std::vector<TrackingRecHit*> DThits = (*it)->recHits();
       int dt1DHits = 0;
       for (size_t j=0; j< DThits.size(); j++) {
           dt1DHits += (DThits[j]->recHits()).size();
       }
       NRechits = dt1DHits ;
    }
    if ( geoId.subdetId() == MuonSubdetId::CSC ) {
       NRechits = ((*it)->recHits()).size() ;
    }
    */
    if ( NRechits < best_nhits ) continue;
    best_nhits = NRechits ; 

    // reject 2D segments if 4D segments are available 
    if ( (*it)->dimension() < best_dimension ) continue;
    best_dimension = (*it)->dimension();

    // pick the segment with best chi2/dof within a fixed cone size
    if ( dR > best_R ) continue;

    // select smaller chi2/dof
    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    /// reject possible edge segments  
    if ((*it)->chi2()/dof < 0.001 && NRechits < 6 && type == 1) continue; 
    if ( (*it)->chi2()/dof > best_chi2 ) continue;
    best_chi2 = (*it)->chi2()/dof ;
    best_match = index;
    // propagate the eta and phi to next layer
    if ((*it)->dimension() != 4 ) {
       phi_last = phi_last;
       eta_last = eta_last;
    } else {
       phi_last = gp2.phi(); 
       eta_last = gp2.eta();
    }
  }   

  if (best_match < 0) return ok;

  index = -1;
   
  // Add best matching segment to protoTrack:
  for (SegmentContainer::iterator it=segs.begin(); it!=segs.end(); ++it){
      index++;
      if (index != best_match) continue;
      protoTrack.push_back(*it);
      usedSeg[best_match] = true;
      ok = true;     
  }  
  return ok; 
}

std::vector<TrajectorySeed> MuonSeedBuilder::seedCleaner(const edm::EventSetup& eventSetup, std::vector<TrajectorySeed>& seeds ) {

  theService->update(eventSetup);
  
  std::vector<TrajectorySeed> FinalSeeds;
  
  //std::cout<<" 1 **** SeedGrouping ***** "<<std::endl;
  // group the seeds
  std::vector<SeedContainer> theCollection = GroupSeeds(seeds);

  //std::cout<<" 2 **** SeedCleaning ***** "<<std::endl;

  // ckeck each group and pick the good one
  for (size_t i=0; i< theCollection.size(); i++ ) {

      // pick the seed w/ more than 1 segments and w/ 1st layer segment information
      SeedContainer goodSeeds   = SeedCandidates( theCollection[i], true );
      SeedContainer otherSeeds  = SeedCandidates( theCollection[i], false );
      if ( MomentumFilter( goodSeeds ) )  {
         //std::cout<<" == type1 "<<std::endl;
         std::vector<TrajectorySeed> betterSeeds = LengthFilter( goodSeeds ); 
	 TrajectorySeed bestSeed   = BetterChi2( betterSeeds );
	 //TrajectorySeed bestSeed   = BetterDirection( betterSeeds );
	 FinalSeeds.push_back( bestSeed );
      } 
      else if( MomentumFilter( otherSeeds ) ) {
         //std::cout<<" == type2 "<<std::endl;
         SeedContainer betterSeeds = LengthFilter( otherSeeds ); 
	 TrajectorySeed bestSeed   = BetterChi2( betterSeeds );
	 //TrajectorySeed bestSeed   = BetterDirection( betterSeeds );
	 FinalSeeds.push_back( bestSeed );
      } 
      else {
         //std::cout<<" == type3 "<<std::endl;
         SeedContainer betterSeeds = LengthFilter( theCollection[i] ); 
	 TrajectorySeed bestSeed   = BetterChi2( betterSeeds );
	 //TrajectorySeed bestSeed   = BetterDirection( betterSeeds );
	 FinalSeeds.push_back( bestSeed );
      }
      //std::cout<<"  **** Fine one seed ***** "<<std::endl;
  }  
  return FinalSeeds ; 

}


TrajectorySeed MuonSeedBuilder::BetterDirection(std::vector<TrajectorySeed>& seeds ) {

  float bestProjErr  = 9999.; 
  int winner = 0 ;
  AlgebraicSymMatrix mat(5,0) ; 
  for ( size_t i = 0; i < seeds.size(); i++ ) {

      edm::OwnVector<TrackingRecHit>::const_iterator r1 = seeds[i].recHits().first ;
      mat = r1->parametersError().similarityT( r1->projectionMatrix() );
      float ddx = mat[1][1]; 
      float ddy = mat[2][2]; 
      float dxx = mat[3][3]; 
      float dyy = mat[4][4];
      float projectErr = sqrt( (ddx*10000.) + (ddy*10000.) + dxx + dyy ) ;

      if ( projectErr > bestProjErr ) continue;
      winner = static_cast<int>(i) ;
      bestProjErr = projectErr ;
  }
  return seeds[winner];

}

TrajectorySeed MuonSeedBuilder::BetterChi2(std::vector<TrajectorySeed>& seeds ) {

  if ( seeds.size() == 1 ) return seeds[0];

  int winner = 0 ;
  std::vector<double> bestChi2(4,99999.);  
  std::vector<int>    moreHits(4,0);  
  for ( size_t i = 0; i < seeds.size(); i++ ) {

      // 1. fill out the Nchi2 of segments of the seed 
      //GlobalVector mom = SeedMomentum( seeds[i] ); // temporary use for debugging
      //double pt = sqrt( (mom.x()*mom.x()) + (mom.y()*mom.y()) );
      //std::cout<<" > SEED"<<i<<"  pt:"<<pt<< std::endl;

      int it = -1;
      std::vector<double> theChi2(4,99999.);  
      std::vector<int>    theHits(4,0);  
      for (edm::OwnVector<TrackingRecHit>::const_iterator r1 = seeds[i].recHits().first; r1 != seeds[i].recHits().second; r1++){
          it++;
          //std::cout<<"    segmet : "<<it <<std::endl; 
          if ( it > 3) break;
          theHits[it] = NRecHitsFromSegment( *r1 );
          theChi2[it] = NChi2OfSegment( *r1 );
      }
      //std::cout<<" -----  "<<std::endl;

      // 2. longer segment
      for (int j =0; j<4; j++) {

          if ( theHits[j] <  moreHits[j] ) break;

          if ( theHits[j] == moreHits[j] ) { 
            ///  compare the chi2 list
            bool decisionMade = false ;
            for (int k =0; k<4; k++) {
               if ( theChi2[k] >  bestChi2[k] ) break;
               if ( theChi2[k] == bestChi2[k] ) continue;
               if ( theChi2[k] <  bestChi2[k] ) {
                  bestChi2 = theChi2 ;
                  winner = static_cast<int>(i) ;
                  decisionMade = true;
                }
                break;
            }
            if ( decisionMade) break;
            if (!decisionMade) continue;
          }

          if ( theHits[j] >  moreHits[j] ) {
             moreHits = theHits ;
             winner = static_cast<int>(i) ;
          }
          break;
      }
  }
  //std::cout<<" Winner is "<< winner <<std::endl;
  return seeds[winner];
}

SeedContainer MuonSeedBuilder::LengthFilter(std::vector<TrajectorySeed>& seeds ) {
 
  SeedContainer longSeeds; 
  int NSegs = 0;
  for (size_t i = 0; i< seeds.size(); i++) {
    
      int theLength = static_cast<int>( seeds[i].nHits());
      if ( theLength > NSegs ) {
         NSegs = theLength ;
         longSeeds.clear();
         longSeeds.push_back( seeds[i] );
      } 
      else if ( theLength == NSegs ) {
         longSeeds.push_back( seeds[i] );
      } else {
         continue;
      } 
  }
  //std::cout<<" final Length :"<<NSegs<<std::endl;

  return longSeeds ; 
  
}

bool MuonSeedBuilder::MomentumFilter(std::vector<TrajectorySeed>& seeds ) {

  bool findgoodMomentum = false;
  SeedContainer goodMomentumSeeds = seeds;
  seeds.clear();
  for ( size_t i = 0; i < goodMomentumSeeds.size(); i++ ) {
       GlobalVector mom = SeedMomentum( goodMomentumSeeds[i] );
       double pt = sqrt( (mom.x()*mom.x()) + (mom.y()*mom.y()) );
       //if ( pt < 6.  || pt > 3000. ) continue;
       if ( pt < 6. ) continue;
       //std::cout<<" passed momentum :"<< pt <<std::endl;
       seeds.push_back( goodMomentumSeeds[i] );
       findgoodMomentum = true;  
  }
  if ( seeds.size() == 0 ) seeds = goodMomentumSeeds;

  return findgoodMomentum;
}

SeedContainer MuonSeedBuilder::SeedCandidates( std::vector<TrajectorySeed>& seeds, bool good ) {

  SeedContainer theCandidate;
  theCandidate.clear();

  bool longSeed = false;  
  bool withFirstLayer = false ;

  //std::cout<<"***** Seed Classification *****"<< seeds.size() <<std::endl;
  for ( size_t i = 0; i < seeds.size(); i++ ) {

      if (seeds[i].nHits() > 1 ) longSeed = true ;
      //std::cout<<"  Seed: "<<i<< std::endl;
      int idx = 0;
      for (edm::OwnVector<TrackingRecHit>::const_iterator r1 = seeds[i].recHits().first; r1 != seeds[i].recHits().second; r1++){

         idx++;
         const GeomDet* gdet = theService->trackingGeometry()->idToDet( (*r1).geographicalId() );
         DetId geoId = gdet->geographicalId();

         if ( geoId.subdetId() == MuonSubdetId::DT ) {
            DTChamberId DT_Id( (*r1).geographicalId() );
            //std::cout<<" ID:"<<DT_Id <<" pos:"<< r1->localPosition()  <<std::endl;
            if (DT_Id.station() != 1)  continue;
            withFirstLayer = true;
         }
         if ( geoId.subdetId() == MuonSubdetId::CSC ) {
            idx++;
            CSCDetId CSC_Id = CSCDetId( (*r1).geographicalId() );
            //std::cout<<" ID:"<<CSC_Id <<" pos:"<< r1->localPosition()  <<std::endl;
            if (CSC_Id.station() != 1)  continue;
            withFirstLayer = true;
         }
      }
      bool goodseed = (longSeed && withFirstLayer) ? true : false ;
  
      if ( goodseed == good )  theCandidate.push_back( seeds[i] );
  }
  return theCandidate;

}

std::vector<SeedContainer> MuonSeedBuilder::GroupSeeds( std::vector<TrajectorySeed>& seeds) {

  std::vector<SeedContainer> seedCollection;
  seedCollection.clear();
  std::vector<TrajectorySeed> theGroup ;
  std::vector<bool> usedSeed(seeds.size(),false);

  // categorize seeds by comparing overlapping segments or a certian eta-phi cone 
  for (unsigned int i= 0; i<seeds.size(); i++){
    
    if (usedSeed[i]) continue;
    theGroup.push_back( seeds[i] );
    usedSeed[i] = true ;

    GlobalPoint pos1 = SeedPosition( seeds[i]);

    for (unsigned int j= i+1; j<seeds.size(); j++){
 
       // seeds with overlaaping segments will be grouped together
       unsigned int overlapping = OverlapSegments(seeds[i], seeds[j]) ;
       if ( !usedSeed[j] && overlapping > 0 ) {
          // reject the identical seeds
          if ( seeds[i].nHits() == overlapping && seeds[j].nHits() == overlapping ) {
             usedSeed[j] = true ;
             continue;
          }
          theGroup.push_back( seeds[j] ); 
          usedSeed[j] = true ;
       }
       if (usedSeed[j]) continue;

       // seeds in a certain cone are grouped together  
       GlobalPoint pos2 = SeedPosition( seeds[j]);
       double dh = pos1.eta() - pos2.eta() ;
       double df = pos1.phi() - pos2.phi() ;
       double dR = sqrt( (dh*dh) + (df*df) );

       if ( dR > 0.3 && seeds[j].nHits() == 1) continue;
       if ( dR > 0.2 && seeds[j].nHits() >  1) continue;
       theGroup.push_back( seeds[j] ); 
       usedSeed[j] = true ;
    }
    sort(theGroup.begin(), theGroup.end(), lengthSorting ) ;
    seedCollection.push_back(theGroup);
    theGroup.clear(); 
  }
  return seedCollection;

}

unsigned int MuonSeedBuilder::OverlapSegments( TrajectorySeed seed1, TrajectorySeed seed2 ) {

  unsigned int overlapping = 0;
  for (edm::OwnVector<TrackingRecHit>::const_iterator r1 = seed1.recHits().first; r1 != seed1.recHits().second; r1++){
      
      DetId id1 = (*r1).geographicalId();
      const GeomDet* gdet1 = theService->trackingGeometry()->idToDet( id1 );
      GlobalPoint gp1 = gdet1->toGlobal( (*r1).localPosition() );

      for (edm::OwnVector<TrackingRecHit>::const_iterator r2 = seed2.recHits().first; r2 != seed2.recHits().second; r2++){

          DetId id2 = (*r2).geographicalId();
          if (id1 != id2 ) continue;

	  const GeomDet* gdet2 = theService->trackingGeometry()->idToDet( id2 );
	  GlobalPoint gp2 = gdet2->toGlobal( (*r2).localPosition() );

	  double dx = gp1.x() - gp2.x() ;
	  double dy = gp1.y() - gp2.y() ;
	  double dz = gp1.z() - gp2.z() ;
	  double dL = sqrt( dx*dx + dy*dy + dz*dz);

          if ( dL < 1. ) overlapping ++;
      
      }
  }
  return overlapping ;

}

GlobalPoint MuonSeedBuilder::SeedPosition( TrajectorySeed seed ) {

  TrajectoryStateTransform tsTransform;

  PTrajectoryStateOnDet pTSOD = seed.startingState();
  DetId SeedDetId(pTSOD.detId());
  const GeomDet* geoDet = theService->trackingGeometry()->idToDet( SeedDetId );
  TrajectoryStateOnSurface SeedTSOS = tsTransform.transientState(pTSOD, &(geoDet->surface()), &*theService->magneticField());
  GlobalPoint  pos  = SeedTSOS.globalPosition();

  return pos ;

}

GlobalVector MuonSeedBuilder::SeedMomentum( TrajectorySeed seed ) {

  TrajectoryStateTransform tsTransform;

  PTrajectoryStateOnDet pTSOD = seed.startingState();
  DetId SeedDetId(pTSOD.detId());
  const GeomDet* geoDet = theService->trackingGeometry()->idToDet( SeedDetId );
  TrajectoryStateOnSurface SeedTSOS = tsTransform.transientState(pTSOD, &(geoDet->surface()), &*theService->magneticField());
  GlobalVector  mom  = SeedTSOS.globalMomentum();

  return mom ;

}

int MuonSeedBuilder::NRecHitsFromSegment( const TrackingRecHit& rhit ) {

      int NRechits = 0 ; 
      const GeomDet* gdet = theService->trackingGeometry()->idToDet( rhit.geographicalId() );
      MuonTransientTrackingRecHit::MuonRecHitPointer theSeg = 
      MuonTransientTrackingRecHit::specificBuild(gdet, rhit.clone() );

      DetId geoId = gdet->geographicalId();
      if ( geoId.subdetId() == MuonSubdetId::DT ) {
         DTChamberId DT_Id( rhit.geographicalId() );
	 std::vector<TrackingRecHit*> DThits = theSeg->recHits();
	 int dt1DHits = 0;
	 for (size_t j=0; j< DThits.size(); j++) {
             dt1DHits += (DThits[j]->recHits()).size();
         }
         NRechits = dt1DHits ;
         //std::cout<<" D_rh("<< dt1DHits  <<") " ;
      }

      if ( geoId.subdetId() == MuonSubdetId::CSC ) {
         NRechits = (theSeg->recHits()).size() ;
         //std::cout<<" C_rh("<< NRechits <<") " ;
      }
      return NRechits ;
}

int MuonSeedBuilder::NRecHitsFromSegment( MuonTransientTrackingRecHit *rhit ) {

    int NRechits = 0 ; 
    DetId geoId = rhit->geographicalId();
    if ( geoId.subdetId() == MuonSubdetId::DT ) {
       DTChamberId DT_Id( geoId );
       std::vector<TrackingRecHit*> DThits = rhit->recHits();
       int dt1DHits = 0;
       for (size_t j=0; j< DThits.size(); j++) {
           dt1DHits += (DThits[j]->recHits()).size();
       }
       NRechits = dt1DHits ;
       //std::cout<<" D_rh("<< dt1DHits  <<") " ;
    }
    if ( geoId.subdetId() == MuonSubdetId::CSC ) {
       NRechits = (rhit->recHits()).size() ;
       //std::cout<<" C_rh("<<(rhit->recHits()).size() <<") " ;
    }
    return NRechits;

}

double MuonSeedBuilder::NChi2OfSegment( const TrackingRecHit& rhit ) {

      double NChi2 = 999999. ; 
      const GeomDet* gdet = theService->trackingGeometry()->idToDet( rhit.geographicalId() );
      MuonTransientTrackingRecHit::MuonRecHitPointer theSeg = 
      MuonTransientTrackingRecHit::specificBuild(gdet, rhit.clone() );

      double dof = static_cast<double>( theSeg->degreesOfFreedom() );
      NChi2 = theSeg->chi2() / dof ;
      //std::cout<<" Chi2 = "<< NChi2  <<" |" ;

      return NChi2 ;
}

bool MuonSeedBuilder::IdentifyShowering( SegmentContainer& segs, BoolContainer& usedSeg, float& eta_last, float& phi_last, int layer, int& NShoweringSegments ) {

  bool showering  = false ;  

  int  nSeg   = 0 ;
  int  nRhits = 0 ;
  double nChi2  = 9999. ;
  int theOrigin = -1;
  std::vector<int> badtag;
  int    index = -1;
  double aveEta = 0.0;
  for (SegmentContainer::iterator it = segs.begin(); it != segs.end(); ++it){

      index++;
      GlobalPoint gp = (*it)->globalPosition(); 
      double dh = gp.eta() - eta_last ;
      double df = gp.phi() - phi_last ;
      double dR = sqrt( (dh*dh) + (df*df) ) ;

      double dof = static_cast<double>( (*it)->degreesOfFreedom() );
      double nX2 = (*it)->chi2() / dof ;

      bool isDT = false ; 
      DetId geoId = (*it)->geographicalId();
      if ( geoId.subdetId() == MuonSubdetId::DT ) isDT = true;

      if (dR < 0.3 ) {
         nSeg++ ;
         badtag.push_back( index ) ;
         aveEta += fabs( gp.eta() ) ; 
         // pick up the best segment from showering chamber 
         int rh = NRecHitsFromSegment( &*(*it) );
         if (rh < 6 && !isDT) continue;
         if (rh < 12 && isDT) continue;
         if ( rh > nRhits ) { 
            nRhits = rh ;
            if ( nX2 > nChi2 ) continue ;
            if (layer != 0 && layer != 1 && layer != -1 ) {
               theOrigin = index ;
            }
         }
      }

  }
  aveEta =  aveEta/static_cast<double>(nSeg) ;
  bool isME11A = (aveEta >= 2.1 &&  layer == 0) ? true : false ;
  bool isME12  = (aveEta >  1.2 && aveEta <= 1.65 && layer == 1) ? true : false ;
  bool isME11  = (aveEta >  1.65 && aveEta <= 2.1 && layer == 0) ? true : false ;
  bool is1stLayer = (layer == -1 || layer == 0 || isME12 || isME11 || isME11A) ? true : false ;

  NShoweringSegments += nSeg;

  if ( nSeg  > 3 && !isME11A ) showering = true ;
  if ( nSeg  > 6 &&  isME11A ) showering = true ;

  // if showering, flag all segments in order to skip this layer for pt estimation except 1st layer
  //std::cout<<" from Showering "<<std::endl;
  if (showering && !is1stLayer ) {
     for (std::vector<int>::iterator it = badtag.begin(); it != badtag.end(); ++it ) { 
         usedSeg[*it] = true;              
         if ( (*it) != theOrigin ) continue; 
	 ShoweringSegments.push_back( segs[*it] );
	 ShoweringLayers.push_back( layer );
     }
  }
  return showering ;

}

double MuonSeedBuilder::etaError(const GlobalPoint gp, double rErr) {

  double dHdTheta = 0.0;
  double dThetadR = 0.0;
  double etaErr = 1.0;

  if (gp.perp() != 0) {

     dHdTheta = ( gp.mag()+gp.z() )/gp.perp();
     dThetadR = gp.z() / gp.perp2() ;
     etaErr =  0.25 * (dHdTheta * dThetadR) * (dHdTheta * dThetadR) * rErr ;
  }
 
  return etaErr;
}

