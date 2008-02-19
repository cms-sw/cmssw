/**
 *  See header file for a description of this class.
 *  
 *  \author Dominique Fortin - UCR
 */


#include <RecoMuon/MuonSeedGenerator/src/MuonSeedBuilder.h>
#include <RecoMuon/MuonSeedGenerator/src/MuonSeedCreator.h>

// Data Formats 
#include <DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h>
#include <DataFormats/TrajectorySeed/interface/TrajectorySeed.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
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

typedef std::pair<double, TrajectorySeed> seedpr ;
static bool ptDecreasing(const seedpr s1, const seedpr s2) { return ( s1.first > s2.first ); }


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

  // Minimum seed Pt
  theMinMomentum       = pset.getParameter<double>("minimumSeedPt");


  // muon service
  // edm::ParameterSet serviceParameters = pset.getParameter<edm::ParameterSet>("ServiceParameters");
  // theService        = new MuonServiceProxy(serviceParameters);

  // Class for forming muon seeds
  muonSeedCreate_      = new MuonSeedCreator( pset ); 

}

/*
 * Destructor
 */
MuonSeedBuilder::~MuonSeedBuilder(){

  delete muonSeedCreate_;

  // if (theService) delete theService;
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
  std::vector<float> ptOfSeed;
  std::vector<int>   nSegOnSeed;


 // Instantiate the accessor (get the segments: DT + CSC but not RPC=false)
  MuonDetLayerMeasurements muonMeasurements(theDTSegmentLabel,theCSCSegmentLabel,edm::InputTag(),
					    enableDTMeasurement,enableCSCMeasurement,false);
  
  
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

  // mark the showering layer
  badSeedLayer.clear();

  if (debug) {
    std::cout << "*** Number of CSC segments is " << CSClist4F.size()+CSClist3F.size()+CSClist2F.size()+CSClist1F.size()+CSClist0F.size()+CSClist4B.size()+CSClist3B.size()+CSClist2B.size()+CSClist1B.size()+CSClist0B.size()<< std::endl;
    std::cout << "In ME11: " << CSClist0F.size()+CSClist0B.size() << std::endl;
    std::cout << "In ME12: " << CSClist1F.size()+CSClist1B.size() << std::endl;
    std::cout << "In ME2 : " << CSClist2F.size()+CSClist2B.size() << std::endl;
    std::cout << "In ME3 : " << CSClist3F.size()+CSClist3B.size() << std::endl;
    std::cout << "In ME4 : " << CSClist4F.size()+CSClist4B.size() << std::endl;
  }


  /* *********************************************************************************************************************
   * Form seeds in barrel region
   *
   * Proceed from inside -> out
   * *********************************************************************************************************************/

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
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(-1);

    // Try adding segment from other stations
    if (foundMatchingSegment(3, protoTrack, DTlist2, usedDTlist2, eta_temp, phi_temp)) layers.push_back(-2);
    if (foundMatchingSegment(3, protoTrack, DTlist3, usedDTlist3, eta_temp, phi_temp)) layers.push_back(-3);
    if (foundMatchingSegment(3, protoTrack, DTlist4, usedDTlist4, eta_temp, phi_temp)) layers.push_back(-4);

    // Check if in overlap region
    if (eta_temp > 0.8) {
      if (foundMatchingSegment(2, protoTrack, CSClist1F, usedCSClist1F, eta_temp, phi_temp)) layers.push_back(1);
      if (foundMatchingSegment(2, protoTrack, CSClist2F, usedCSClist2F, eta_temp, phi_temp)) layers.push_back(2);
      if (foundMatchingSegment(2, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp)) layers.push_back(3);
    } 
    else if (eta_temp < -0.8) {
      if (foundMatchingSegment(2, protoTrack, CSClist1B, usedCSClist1B, eta_temp, phi_temp)) layers.push_back(1);
      if (foundMatchingSegment(2, protoTrack, CSClist2B, usedCSClist2B, eta_temp, phi_temp)) layers.push_back(2);
      if (foundMatchingSegment(2, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp)) layers.push_back(3);
    }
 
    unsigned nLayers = layers.size();
   
    if ( nLayers < 2 ) continue; 

    TrajectorySeed thisSeed;

    if ( layers[nLayers-1] > 0 ) {
      thisSeed = muonSeedCreate_->createSeed(2, protoTrack, layers, badSeedLayer, seedPt);  // overlap
    } else {
      thisSeed = muonSeedCreate_->createSeed(3, protoTrack, layers, badSeedLayer, seedPt);  // DT only
    }

    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed); 
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
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
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(-2);

 
    // Try adding segment from other stations
    if (foundMatchingSegment(3, protoTrack, DTlist3, usedDTlist3, eta_temp, phi_temp)) layers.push_back(-3);
    if (foundMatchingSegment(3, protoTrack, DTlist4, usedDTlist4, eta_temp, phi_temp)) layers.push_back(-4);

    // Check if in overlap region
    if (eta_temp > 0.8) {
      if (foundMatchingSegment(2, protoTrack, CSClist1F, usedCSClist1F, eta_temp, phi_temp)) layers.push_back(1);
      if (foundMatchingSegment(2, protoTrack, CSClist2F, usedCSClist2F, eta_temp, phi_temp)) layers.push_back(2);
      if (foundMatchingSegment(2, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp)) layers.push_back(3);
    }
    else if (eta_temp < -0.8) {
      if (foundMatchingSegment(2, protoTrack, CSClist1B, usedCSClist1B, eta_temp, phi_temp)) layers.push_back(1);
      if (foundMatchingSegment(2, protoTrack, CSClist2B, usedCSClist2B, eta_temp, phi_temp)) layers.push_back(2);
      if (foundMatchingSegment(2, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp)) layers.push_back(3);
    }

    unsigned nLayers = layers.size();
  
    if ( nLayers < 2 ) continue;
    
    TrajectorySeed thisSeed;
  
    if ( layers[nLayers-1] > 0 ) {
      thisSeed = muonSeedCreate_->createSeed(2, protoTrack, layers, badSeedLayer, seedPt);  // overlap
    } else {
      thisSeed = muonSeedCreate_->createSeed(3, protoTrack, layers, badSeedLayer, seedPt);  // DT only
    }

    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed); 
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
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
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(-3);
 
    // Try adding segment from other stations
    if (foundMatchingSegment(3, protoTrack, DTlist4, usedDTlist4, eta_temp, phi_temp)) layers.push_back(-4);

    // Check if in overlap region
    if (eta_temp > 0.8) {
      if (foundMatchingSegment(2, protoTrack, CSClist1F, usedCSClist1F, eta_temp, phi_temp)) layers.push_back(1);
      if (foundMatchingSegment(2, protoTrack, CSClist2F, usedCSClist2F, eta_temp, phi_temp)) layers.push_back(2);
      if (foundMatchingSegment(2, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp)) layers.push_back(3);
    }
    else if (eta_temp < -0.8) {
      if (foundMatchingSegment(2, protoTrack, CSClist1B, usedCSClist1B, eta_temp, phi_temp)) layers.push_back(1);
      if (foundMatchingSegment(2, protoTrack, CSClist2B, usedCSClist2B, eta_temp, phi_temp)) layers.push_back(2);
      if (foundMatchingSegment(2, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp)) layers.push_back(3);
    }
    
    unsigned nLayers = layers.size();
  
    if ( nLayers < 2 ) continue;
    
    TrajectorySeed thisSeed;
  
    if ( layers[nLayers-1] > 0 ) {
      thisSeed = muonSeedCreate_->createSeed(2, protoTrack, layers, badSeedLayer, seedPt);  // overlap
    } else {
      thisSeed = muonSeedCreate_->createSeed(3, protoTrack, layers, badSeedLayer, seedPt);  // DT only
    }

    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed); 
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
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

    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(0);

    // Try adding segment from other station
    if (foundMatchingSegment(1, protoTrack, CSClist1B, usedCSClist1B, eta_temp, phi_temp)) layers.push_back(1);
    if (foundMatchingSegment(1, protoTrack, CSClist2B, usedCSClist2B, eta_temp, phi_temp)) layers.push_back(2);
    if (foundMatchingSegment(1, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp)) layers.push_back(3);
    if (foundMatchingSegment(1, protoTrack, CSClist4B, usedCSClist4B, eta_temp, phi_temp)) layers.push_back(4);

    unsigned nLayers = layers.size();

    if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed;
    if ( fabs( gp.eta() ) > 2.1  ) {
      thisSeed = muonSeedCreate_->createSeed(5, protoTrack, layers, badSeedLayer, seedPt);
      rawSeeds.push_back(thisSeed);
          //thisSeed = muonSeedCreate_->createSeed(6, protoTrack, layers, seedPt);
          //rawSeeds.push_back(thisSeed);
    } else {
      thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer, seedPt);
      rawSeeds.push_back(thisSeed);
    }

    // Add the seeds to master collection
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
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

    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();

    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);
    
    std::vector<int> layers;
    layers.push_back(1);

    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist2B, usedCSClist2B, eta_temp, phi_temp)) layers.push_back(2);
    if (foundMatchingSegment(1, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp)) layers.push_back(3);
    if (foundMatchingSegment(1, protoTrack, CSClist4B, usedCSClist4B, eta_temp, phi_temp)) layers.push_back(4);

    unsigned nLayers = layers.size();

    if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer, seedPt);

    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
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
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(2);
    
    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp)) layers.push_back(3);
    if (foundMatchingSegment(1, protoTrack, CSClist4B, usedCSClist4B, eta_temp, phi_temp)) layers.push_back(4);

    unsigned nLayers = layers.size();

    if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer, seedPt);

    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
    nSegOnSeed.push_back( nLayers );
    // mark this segment as used
    usedCSClist2B[index] = true;
  }


  /* *********************************************************************************************************************
   * Form seeds from forward endcap
   *
   * Proceed from inside -> out
   * *********************************************************************************************************************/

  // Loop over all possible ME11 segment to form seeds:
  index = -1;

  for (SegmentContainer::iterator it = CSClist0F.begin(); it != CSClist0F.end(); ++it ){

    index++;
  
    if (usedCSClist0F[index] == true) continue;
    if ( int ((*it)->recHits().size()) < minCSCHitsPerSegment ) continue;
    
    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();  
    float phi_temp = gp.phi();      
    float seedPt = 0.;
    
    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(0);

    // Try adding segment from other station
    if (foundMatchingSegment(1, protoTrack, CSClist1F, usedCSClist1F, eta_temp, phi_temp)) layers.push_back(1);
    if (foundMatchingSegment(1, protoTrack, CSClist2F, usedCSClist2F, eta_temp, phi_temp)) layers.push_back(2);
    if (foundMatchingSegment(1, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp)) layers.push_back(3);
    if (foundMatchingSegment(1, protoTrack, CSClist4F, usedCSClist4F, eta_temp, phi_temp)) layers.push_back(4);

    unsigned nLayers = layers.size();

    if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed;

    if ( fabs( gp.eta() ) > 2.1  ) {
      thisSeed = muonSeedCreate_->createSeed(5, protoTrack, layers, badSeedLayer, seedPt);
      rawSeeds.push_back(thisSeed);
          //thisSeed = muonSeedCreate_->createSeed(6, protoTrack, layers, seedPt);
          //rawSeeds.push_back(thisSeed);
    } else {
      thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer, seedPt);
      rawSeeds.push_back(thisSeed);
    }
 
    // Add the seeds to master collection
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
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
  
    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();

    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    float seedPt = 0.;
    
    SegmentContainer protoTrack;
    protoTrack.push_back(*it);
   
    std::vector<int> layers;
    layers.push_back(1);

    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist2F, usedCSClist2F, eta_temp, phi_temp)) layers.push_back(2);
    if (foundMatchingSegment(1, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp)) layers.push_back(3);
    if (foundMatchingSegment(1, protoTrack, CSClist4F, usedCSClist4F, eta_temp, phi_temp)) layers.push_back(4);

    unsigned nLayers = layers.size();
  
    if ( nLayers < 2 ) continue; 
  
    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer, seedPt);

    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
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
    float seedPt = 0.;
   
    SegmentContainer protoTrack;
    protoTrack.push_back(*it);
  
    std::vector<int> layers;
    layers.push_back(2);

    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp)) layers.push_back(3);
    if (foundMatchingSegment(1, protoTrack, CSClist4F, usedCSClist4F, eta_temp, phi_temp)) layers.push_back(4);
  
    unsigned nLayers = layers.size();

    if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer, seedPt);
      
    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);  
    etaOfSeed.push_back(eta_temp); 
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
    nSegOnSeed.push_back( nLayers );
    
    // mark this segment as used
    usedCSClist2F[index] = true;
  }

  /* *********************************************************************************************************************
   *      Form Seed for those single segment events
   *
   * *********************************************************************************************************************/

  index = -1;
  for (SegmentContainer::iterator it = DTlist1.begin(); it != DTlist1.end(); ++it ){
    index++;
    if (usedDTlist1[index] == true) continue;
    //if ((*it)->dimension() != 4) continue;

    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(-1);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer, seedPt);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
    nSegOnSeed.push_back( 1 );
    usedDTlist1[index] = true;
  }
  index = -1;
  for (SegmentContainer::iterator it = DTlist2.begin(); it != DTlist2.end(); ++it ){
    index++;
    if (usedDTlist2[index] == true) continue;
    //if ((*it)->dimension() != 4) continue;

    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(-2);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer, seedPt);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
    nSegOnSeed.push_back( 1 );
    usedDTlist2[index] = true;
  }

  index = -1;
  for (SegmentContainer::iterator it = DTlist3.begin(); it != DTlist3.end(); ++it ){
    index++;
    if (usedDTlist3[index] == true) continue;
    //if ((*it)->dimension() != 4) continue;

    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(-3);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer, seedPt);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
    nSegOnSeed.push_back( 1 );
    usedDTlist3[index] = true;
  }

  index = -1;
  for (SegmentContainer::iterator it = CSClist0F.begin(); it != CSClist0F.end(); ++it ){
    index++;
    if (usedCSClist0F[index] == true) continue;

    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    float seedPt = 0.;
    if ( fabs(eta_temp) > 2.08 ) continue;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(0);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer, seedPt);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( 1 );
    ptOfSeed.push_back(seedPt);
    usedCSClist0F[index] = true;
  }

  index = -1;
  for (SegmentContainer::iterator it = CSClist0B.begin(); it != CSClist0B.end(); ++it ){
    index++;
    if (usedCSClist0B[index] == true) continue;

    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    float seedPt = 0.;
    if ( fabs(eta_temp) > 2.08 ) continue;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(0);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer, seedPt);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
    nSegOnSeed.push_back( 1 );
    usedCSClist0B[index] = true;
  }

  index = -1;
  for (SegmentContainer::iterator it = CSClist1F.begin(); it != CSClist1F.end(); ++it ){
    index++;
    if (usedCSClist1F[index] == true) continue;

    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(1);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer, seedPt);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
    nSegOnSeed.push_back( 1 );
    usedCSClist1F[index] = true;
  }

  index = -1;
  for (SegmentContainer::iterator it = CSClist1B.begin(); it != CSClist1B.end(); ++it ){
    index++;
    if (usedCSClist1B[index] == true) continue;

    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(1);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer, seedPt);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
    nSegOnSeed.push_back( 1 );
    usedCSClist1B[index] = true;
  }

  index = -1;
  for (SegmentContainer::iterator it = CSClist2F.begin(); it != CSClist2F.end(); ++it ){
    index++;
    if (usedCSClist2F[index] == true) continue;

    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(2);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer, seedPt);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
    nSegOnSeed.push_back( 1 );
    usedCSClist2F[index] = true;
  }

  index = -1;
  for (SegmentContainer::iterator it = CSClist2B.begin(); it != CSClist2B.end(); ++it ){
    index++;
    if (usedCSClist2B[index] == true) continue;

    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    float seedPt = 0.;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(2);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer, seedPt);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    ptOfSeed.push_back(seedPt);
    nSegOnSeed.push_back( 1 );
    usedCSClist2B[index] = true;
  }



  /* *********************************************************************************************************************
   * Clean up raw seed collection and pass to master collection
   * *********************************************************************************************************************/

  if (debug) std::cout << "*** CLEAN UP " << std::endl;
  if (debug) std::cout << "Number of seeds BEFORE " << rawSeeds.size() << std::endl;

  int goodSeeds = 0;
  float MaxEta;
  float MaxPhi;
  std::vector<seedpr> SeedwPt;


  for ( unsigned i = 0; i < rawSeeds.size(); i++ ) {
    bool ok = true;
    if ( fabs(etaOfSeed[i]) < 1. ) {
      MaxEta = maxEtaResolutionDT;
      MaxPhi = maxPhiResolutionDT;
    } else {
      MaxEta = maxEtaResolutionCSC;
      MaxPhi = maxPhiResolutionCSC;
    }

    // Test if 2 seeds represent the same track
    bool closeMatch = false;
    bool bad_estimation2 = false;
    float pt2 = 0.;
    float size2 = 0.;
    double deltaR2 = 999.;

    for ( unsigned j = 0; j < rawSeeds.size(); j++ ) {      
      // same seed, skip
      if ( i == j ) continue;

      double deltaR2_temp = ( etaOfSeed[i] - etaOfSeed[j] ) * ( etaOfSeed[i] - etaOfSeed[j] );
      deltaR2_temp       += ( phiOfSeed[i] - phiOfSeed[j] ) * ( phiOfSeed[i] - phiOfSeed[j] );

      // This shouldn't happen, but just in case...
      if (deltaR2_temp < 0.) deltaR2_temp = -1. * deltaR2_temp;
      deltaR2_temp = sqrt(deltaR2_temp);

      if ( fabs( etaOfSeed[i] - etaOfSeed[j] ) < MaxEta && 
           fabs( phiOfSeed[i] - phiOfSeed[j] ) < MaxPhi ) {

        if ( deltaR2_temp < deltaR2 ) {
          closeMatch = true;
          deltaR2    = deltaR2_temp;
          pt2        = ptOfSeed[j];
          size2      = nSegOnSeed[j];
          if ( fabs(ptOfSeed[j]) <= theMinMomentum+0.1 ) {
            bad_estimation2 = true;
          }
          else {
            bad_estimation2 = false;
          } 
        }
        // If deltaR is same, take seed with higher Pt
        else if ( deltaR2_temp          == deltaR2 &&
                  nSegOnSeed[j]         >= size2 &&
                  fabs(ptOfSeed[j]/pt2)  > 1. ) {
          closeMatch = true;
          deltaR2    = deltaR2_temp;
          pt2        = ptOfSeed[j];
          size2      = nSegOnSeed[j];
          if ( fabs(ptOfSeed[j]) <= theMinMomentum+0.1 ) {
            bad_estimation2 = true;
          }
          else {
            bad_estimation2 = false;
          } 
        }
      }
    }

    bool bad_estimation = ( fabs(ptOfSeed[i]) <= theMinMomentum+0.1 ) ? true:false; 

    // Remove seeds in ME1/a where we have pt = ptMin for those seeds with multiple segments
    if ( bad_estimation  && fabs(etaOfSeed[i]) > 2.1 ) ok = false;


/*    if ( closeMatch ) {
 *
 *     // If 2 seeds close to one another, but not same:
 *     if ( bad_estimation && !bad_estimation2 ) {
 *       ok = false;   // keep the seed which was formed with more than 1 segment, if pt > ptMinimum
 *     }
 *     else if ( nSegOnSeed[i] < 2 && !bad_estimation2 ) {
 *       ok = false;   // keep the seed which has more than 1 segment and lead to a pt estimate > ptMinimum 
 *     }
 *     else if ( nSegOnSeed[i] <= size2 && fabs(ptOfSeed[i]/pt2) < 1. ) {
 *       ok = false;   // keep the seed with largest pt estimate
 *     }
 *   } 
 */  
    
    if ( ok ) {
      seedpr pr1(ptOfSeed[i], rawSeeds[i]);
      SeedwPt.push_back(pr1);
      goodSeeds++;

      if ( debug ) std::cout << "Seed "          << i
                             << " pt estimate: " << ptOfSeed[i] 
                             << " eta: "         << etaOfSeed[i] 
                             << " phi: "         << phiOfSeed[i] 
                             << std::endl;
    }
  }  


  // Sort the seeds by decreasing pT  
  // This is to help reconstruction out of STA for seeds sharing hits
  sort( SeedwPt.begin(), SeedwPt.end(), ptDecreasing );

  // Fill seed collections to be sent to output
  for(std::vector<seedpr>::iterator i1 = SeedwPt.begin(); i1!= SeedwPt.end(); ++i1 ) {
     theSeeds.push_back( (*i1).second );
  }


  return goodSeeds;

}



/* *********************************************************************************************************************
 * Try to match segment from different station (layer)
 *
 * Note type = 1 --> endcap
 *           = 2 --> overlap
 *           = 3 --> barrel
 * *********************************************************************************************************************/

bool MuonSeedBuilder::foundMatchingSegment( int type, SegmentContainer& protoTrack, SegmentContainer& segs, 
					    BoolContainer& usedSeg, float& eta_last, float& phi_last ) {

  bool ok = false;

  double maxdEta;
  double maxdPhi;
  if (type == 1 ) { 
    // CSC
    maxdEta = maxDeltaEtaCSC;
    maxdPhi = maxDeltaPhiCSC;
  } else if (type == 2 ) { 
    // Overlap
    maxdEta = maxDeltaEtaOverlap;
    maxdPhi = maxDeltaPhiOverlap;
  } else {
    // DT
    maxdEta = maxDeltaEtaDT;
    maxdPhi = maxDeltaPhiDT;
  }

  // look up the showering chamber
  int last_chamber=0;
  int same_cb =0 ;
  int showerLayer = 9;
  for (SegmentContainer::iterator it=segs.begin(); it!=segs.end(); ++it){

      DetId pdid = (*it)->geographicalId();
      if ( pdid.subdetId() == MuonSubdetId::DT) {
         DTChamberId MB_Id = DTChamberId( pdid );
         showerLayer = MB_Id.station();
         if ( MB_Id.sector() == last_chamber ) {
            same_cb++;
         } else {
            last_chamber = MB_Id.sector();
         }
      }
      if ( pdid.subdetId() == MuonSubdetId::CSC) {
         CSCDetId ME_Id = CSCDetId( pdid );
         showerLayer = ME_Id.station();
         if ( ME_Id.chamber() == last_chamber ) {
            same_cb++;
         } else {
            last_chamber = ME_Id.chamber();
         }
      }
  }
  if (same_cb > 1)  badSeedLayer.push_back( showerLayer ); 

  // Global phi/eta from previous segment 
  float eta_temp = eta_last;
  float phi_temp = phi_last;

  // Index counter to keep track of element used in segs 
  int          index = -1;
  int          best_match = index;
  unsigned int best_nCSChits = minCSCHitsPerSegment;
  float        best_R = sqrt( maxdEta*maxdEta + maxdPhi*maxdPhi );
  float        best_chi2 = 20000;

  // Loop over segments in other station (layer) and find candidate match 
  for (SegmentContainer::iterator it=segs.begin(); it!=segs.end(); ++it){

    index++;
    //if (usedSeg[index]) continue;
    if ( (type == 1) && ( int ((*it)->recHits().size()) < minCSCHitsPerSegment ) ) continue;  

    // chi2 cut !
    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    GlobalPoint gp2 = (*it)->globalPosition(); 

    // Not to get confused:  eta_last is from the previous layer.
    // This is only to find the best set of segments by comparing at the distance layer-by-layer 
    double deltaEtaTest = gp2.eta() - eta_temp; 
    double deltaPhiTest = gp2.phi() - phi_temp;

    if (deltaEtaTest < 0.) deltaEtaTest = -deltaEtaTest;
    if (deltaPhiTest < 0.) deltaPhiTest = -deltaPhiTest;

    if ( deltaEtaTest > maxdEta || deltaPhiTest > maxdPhi ) continue;

    float R = sqrt( deltaEtaTest*deltaEtaTest + deltaPhiTest*deltaPhiTest );

    // DF: We should consider using R instead, which seems more natural to me
    //     although the resolution in phi is much better than in eta...
    bool case1 = ( fabs(deltaEtaTest) < maxdEta && fabs(deltaPhiTest)< maxdPhi ) ? true:false ;
    bool case2 = ((*it)->dimension()!= 4) && (fabs(deltaEtaTest)< 0.6) && (fabs(deltaPhiTest)< maxdPhi)? true:false ;
    bool closed = ( fabs( R - best_R ) < 0.01  && index!=1 ) ? true:false;

    if ( !case1 && !case2  ) continue;
    if (R < best_R || closed) {

      //if (deltaPhiTest < best_Dphi) {
      if ((type == 1) && closed && ((*it)->recHits().size()) < best_nCSChits) continue;
      if (type ==1) { best_nCSChits = ((*it)->recHits().size()); }
      // select smaller chi2/dof
      if ( (*it)->chi2()/dof < best_chi2 ) {
         best_chi2 = (*it)->chi2()/dof ;
         best_R = R;
         best_match = index;
      }

      if ((*it)->dimension() != 4 && type==3  ) {
         phi_last = phi_last;
         eta_last = eta_last;
      } else {
         phi_last = gp2.phi(); 
         eta_last = gp2.eta();
      }
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
