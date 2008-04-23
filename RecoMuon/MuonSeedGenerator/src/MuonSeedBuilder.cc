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
typedef std::pair<int, TrajectorySeed> seedpr ;
static bool lDecreasing(const seedpr s1, const seedpr s2) { return ( s1.first > s2.first ); }


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
      thisSeed = muonSeedCreate_->createSeed(2, protoTrack, layers, badSeedLayer);  // overlap
    } else {
      thisSeed = muonSeedCreate_->createSeed(3, protoTrack, layers, badSeedLayer);  // DT only
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
      thisSeed = muonSeedCreate_->createSeed(2, protoTrack, layers, badSeedLayer);  // overlap
    } else {
      thisSeed = muonSeedCreate_->createSeed(3, protoTrack, layers, badSeedLayer);  // DT only
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
      thisSeed = muonSeedCreate_->createSeed(2, protoTrack, layers, badSeedLayer);  // overlap
    } else {
      thisSeed = muonSeedCreate_->createSeed(3, protoTrack, layers, badSeedLayer);  // DT only
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

    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();

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
          thisSeed = muonSeedCreate_->createSeed(5, protoTrack, layers, badSeedLayer);
          rawSeeds.push_back(thisSeed);
          //thisSeed = muonSeedCreate_->createSeed(6, protoTrack, layers);
          //rawSeeds.push_back(thisSeed);
       } else {
          thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer);
          rawSeeds.push_back(thisSeed);
       }

       // Add the seeds to master collection
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

    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();

    float eta_temp = gp.eta();
    float phi_temp = gp.phi();

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

       TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer);

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

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(2);
    
    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist3B, usedCSClist3B, eta_temp, phi_temp)) layers.push_back(3);
    if (foundMatchingSegment(1, protoTrack, CSClist4B, usedCSClist4B, eta_temp, phi_temp)) layers.push_back(4);

    unsigned nLayers = layers.size();

    if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer);

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

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(2);
    
    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist4B, usedCSClist4B, eta_temp, phi_temp)) layers.push_back(4);

    unsigned nLayers = layers.size();

    if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer);

    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( nLayers );
    // mark this segment as used
    usedCSClist3B[index] = true;
  }

  // Loop over all possible ME2 segment to form seeds:
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

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(2);
    
    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist4B, usedCSClist4B, eta_temp, phi_temp)) layers.push_back(4);

    unsigned nLayers = layers.size();

    if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer);

    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( nLayers );
    // mark this segment as used
    usedCSClist3B[index] = true;
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
          thisSeed = muonSeedCreate_->createSeed(5, protoTrack, layers, badSeedLayer);
          rawSeeds.push_back(thisSeed);
          //thisSeed = muonSeedCreate_->createSeed(6, protoTrack, layers);
          //rawSeeds.push_back(thisSeed);
       } else {
          thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer);
          rawSeeds.push_back(thisSeed);
       }
 
       // Add the seeds to master collection
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
  
    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    if ( ((*it)->chi2()/dof) > 20000.0 ) continue;

    // Global position of starting point for protoTrack
    GlobalPoint gp = (*it)->globalPosition();

    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    
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
  
       TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer);

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
   
    SegmentContainer protoTrack;
    protoTrack.push_back(*it);
  
    std::vector<int> layers;
    layers.push_back(2);

    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist3F, usedCSClist3F, eta_temp, phi_temp)) layers.push_back(3);
    if (foundMatchingSegment(1, protoTrack, CSClist4F, usedCSClist4F, eta_temp, phi_temp)) layers.push_back(4);
  
    unsigned nLayers = layers.size();

    if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer);
      
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
   
    SegmentContainer protoTrack;
    protoTrack.push_back(*it);
  
    std::vector<int> layers;
    layers.push_back(2);

    // Try adding segment from other stations
    if (foundMatchingSegment(1, protoTrack, CSClist4F, usedCSClist4F, eta_temp, phi_temp)) layers.push_back(4);
  
    unsigned nLayers = layers.size();

    if ( nLayers < 2 ) continue;

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(1, protoTrack, layers, badSeedLayer);
      
    // Add the seeds to master collection
    rawSeeds.push_back(thisSeed);  
    etaOfSeed.push_back(eta_temp); 
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( nLayers );
    
    // mark this segment as used
    usedCSClist3F[index] = true;
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

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(-1);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
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

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(-2);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
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

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(-3);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
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
    if ( fabs(eta_temp) > 2.08 ) continue;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(0);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( 1 );
    usedCSClist0F[index] = true;
  }

  index = -1;
  for (SegmentContainer::iterator it = CSClist0B.begin(); it != CSClist0B.end(); ++it ){
    index++;
    if (usedCSClist0B[index] == true) continue;

    GlobalPoint gp = (*it)->globalPosition();
    float eta_temp = gp.eta();
    float phi_temp = gp.phi();
    if ( fabs(eta_temp) > 2.08 ) continue;

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(0);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
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

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(1);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
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

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(1);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
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

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(2);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
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

    SegmentContainer protoTrack;
    protoTrack.push_back(*it);

    std::vector<int> layers;
    layers.push_back(2);

    TrajectorySeed thisSeed = muonSeedCreate_->createSeed(4, protoTrack, layers, badSeedLayer);
    rawSeeds.push_back(thisSeed);
    etaOfSeed.push_back(eta_temp);
    phiOfSeed.push_back(phi_temp);
    nSegOnSeed.push_back( 1 );
    usedCSClist2B[index] = true;
  }



  /* *********************************************************************************************************************
   * Clean up raw seed collection and pass to master collection
   * *********************************************************************************************************************/

  if (debug) std::cout << "*** CLEAN UP " << std::endl;
  if (debug) std::cout << "Number of seeds BEFORE " << rawSeeds.size() << std::endl;


  int goodSeeds = 0;
  std::vector<seedpr> SeedwL ;

  for ( unsigned i = 0; i < rawSeeds.size(); i++ ) {
    seedpr pr1(rawSeeds[i].nHits(), rawSeeds[i]);
    SeedwL.push_back(pr1);
  }
  
  // sort the seeds by # of own segments 
  sort(SeedwL.begin(), SeedwL.end(), lDecreasing ) ;
  std::vector<TrajectorySeed> sortedSeeds;
  sortedSeeds.clear();
  for(std::vector<seedpr>::iterator i1 = SeedwL.begin(); i1!= SeedwL.end(); ++i1 ) {
     sortedSeeds.push_back( (*i1).second ) ;
  }

  // clean the seeds 
  //std::cout<<" the original seed size is "<< sortedSeeds.size() <<std::endl;
  seedCleaner(eventSetup,sortedSeeds);
  theSeeds = sortedSeeds;
  goodSeeds = sortedSeeds.size();

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

  // showering test
  int showerIdx =0;
  double aveIdx = 0.;
  double aveEta = 0.;
  double avePhi = 0.;
  int showerLayer = 9;
  for (SegmentContainer::iterator it=segs.begin(); it!=segs.end(); ++it){

      DetId pdid = (*it)->geographicalId();
      if ( pdid.subdetId() == MuonSubdetId::DT ) {
         DTChamberId MB_Id = DTChamberId( pdid );
         showerLayer = -1*MB_Id.station();
      }
      if ( pdid.subdetId() == MuonSubdetId::CSC ) {
         CSCDetId ME_Id = CSCDetId( pdid );
         if (ME_Id.station()==1 && ME_Id.ring()==1 ) {
           showerLayer = 0;     
         } else {
           showerLayer = ME_Id.station();
         }
      }

      GlobalPoint gp1 = (*it)->globalPosition(); 
      double dh = fabs( gp1.eta() - eta_last ); 
      double df = fabs( gp1.phi() - phi_last );

      if ( dh > maxdEta || df > maxdPhi ) continue;
      showerIdx++;
      // don't count the MB4 
      if ( (*it)->dimension() != 4  ) continue;
      aveIdx += 1.0;
      aveEta += gp1.eta(); 
      avePhi += gp1.phi(); 
  }

  if (showerIdx > 2) {
     aveEta = aveEta/aveIdx ;
     avePhi = avePhi/aveIdx ;
     badSeedLayer.push_back( showerLayer ); 
  } else {
     aveEta = eta_last;
     avePhi = phi_last;
  }


  // global phi/eta from previous segment 
  float eta_temp = aveEta;
  float phi_temp = avePhi;

  // Index counter to keep track of element used in segs 
  int          index = -1;
  int          best_match = index;
  unsigned int best_nCSChits = minCSCHitsPerSegment;
  float        best_R = sqrt( maxdEta*maxdEta + maxdPhi*maxdPhi );
  float        best_chi2 = 200;

  // Loop over segments in other station (layer) and find candidate match 
  for (SegmentContainer::iterator it=segs.begin(); it!=segs.end(); ++it){

    index++;
    //if (usedSeg[index]) continue;
    if ( (type == 1) && ( int ((*it)->recHits().size()) < minCSCHitsPerSegment ) ) continue;  

    // chi2 cut !
    double dof = static_cast<double>( (*it)->degreesOfFreedom() ) ;
    //if ( ((*it)->chi2()/dof) > best_chi2 ) continue;

    GlobalPoint gp2 = (*it)->globalPosition(); 
    //GlobalError ge2 = (*it)->globalDirectionError();
    //GlobalError ge2 = (*it)->globalPositionError();
    //double rerr  = ge2.rerr(gp2) ;

    // Not to get confused:  eta_last is from the previous layer.
    // This is only to find the best set of segments by comparing at the distance layer-by-layer 
    double deltaEtaTest = fabs( gp2.eta() - eta_temp ); 
    double deltaPhiTest = fabs( gp2.phi() - phi_temp );


    float R = sqrt( deltaEtaTest*deltaEtaTest + deltaPhiTest*deltaPhiTest );

    bool case1 = ( fabs(deltaEtaTest) < maxdEta && fabs(deltaPhiTest)< maxdPhi ) ? true:false ;
    // for DT station 4
    bool case2 = ((*it)->dimension()!= 4) && (fabs(deltaEtaTest)< 0.6) && (fabs(deltaPhiTest)< maxdPhi)? true:false ;

    bool closed = ( fabs( R - best_R ) < 0.01  && index!=0 ) ? true:false;

    // reject showering segments which are closed enough but not long enough in CSC...
    if ((type == 1) && closed && ((*it)->recHits().size()) < best_nCSChits) continue;

    if ( !case1 && !case2  ) continue;
    // pick the segment which is closest to ave/last position 
    // if they are about the same close, pick the one with best chi2/dof
    if (R < best_R || closed) {

      if (type ==1) { best_nCSChits = ((*it)->recHits().size()); }
      // select smaller chi2/dof
      if ( (*it)->chi2()/dof < best_chi2 ) {
         best_chi2 = (*it)->chi2()/dof ;
         best_R = R;
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

    } 
  }   

  if (best_match < 0) return ok;

  index = -1;
 
   
  if (showerIdx <= 2) { 
    // Add best matching segment to protoTrack:
    for (SegmentContainer::iterator it=segs.begin(); it!=segs.end(); ++it){
      index++;
      if (index != best_match) continue;
      protoTrack.push_back(*it);
      usedSeg[best_match] = true;
      ok = true;     
    }  
    return ok; 
  } else {
     ok = false;
     return ok;
  }

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

void MuonSeedBuilder::seedCleaner(const edm::EventSetup& eventSetup, std::vector<TrajectorySeed>& seeds ) {

  theService->update(eventSetup);
  TrajectoryStateTransform tsTransform;

  // categorize seeds by comparing overlapping segments
  std::vector<bool> usedSeed(seeds.size(),false);
  std::vector<int>  seedGrp(seeds.size(), 0);
  std::vector<int>  grpSize;
  grpSize.clear();
  grpSize.push_back(0) ;
  int grp = 0; 

  for (unsigned int i= 0; i<seeds.size(); i++){
    
    if (usedSeed[i]) continue;
    grp++;
    usedSeed[i] = true; 
    seedGrp[i]  = grp;
    grpSize.push_back(1);

    for (unsigned int j= i+1; j<seeds.size(); j++){
     
      int sameseg=0;
      // compare the segments between 2 seeds
      if(usedSeed[j]) continue;
      for (edm::OwnVector<TrackingRecHit>::const_iterator rt1 = seeds[i].recHits().first; rt1 != seeds[i].recHits().second; rt1++){

        const GeomDet* gdet1 = theService->trackingGeometry()->idToDet( (*rt1).geographicalId() );
        GlobalPoint gp1 = gdet1->toGlobal( (*rt1).localPosition() );
        if (gdet1->subDetector()== MuonSubdetId::DT ) {
           DTChamberId MB_Id = DTChamberId( gdet1->geographicalId() );
           if (MB_Id.station() == 4)  continue;
        }
        for (edm::OwnVector<TrackingRecHit>::const_iterator rt2 = seeds[j].recHits().first; rt2 != seeds[j].recHits().second; rt2++){

            const GeomDet* gdet2 = theService->trackingGeometry()->idToDet( (*rt2).geographicalId() );
            GlobalPoint gp2 = gdet2->toGlobal( (*rt2).localPosition() );

            double dx = gp1.x() - gp2.x() ;
            double dy = gp1.y() - gp2.y() ;
            double dz = gp1.z() - gp2.z() ;
            double dR = sqrt( dx*dx + dy*dy + dz*dz);

            if( dR < 15.0 ) sameseg++;
        }
      }
      //flag and group the used seeds
      if (sameseg > 0) {
         usedSeed[j]=true;
         seedGrp[j]=grp;
         grpSize[grp]+=1; 
      }
    }
  }

  // cleaning the seeds w/ overlapping segments
  std::vector<TrajectorySeed> goodSeeds;
  goodSeeds.clear();

  for (int i=1; i<(grp+1); i++) {
    unsigned int segSize=0;
    double dRErr  = -1.0;
    double drelE  = -1.0;

    int bestseed = -1;
    int grpleader=0;
    bool keep_all = false;
    //std::cout<<" grp"<<i<<" size= "<< grpSize[i] << std::endl;
    //double pt1 = -1.0; 
    //double h1 = -99.0;   

    for (unsigned int j= 0; j<seeds.size(); j++){
 
      if ( seedGrp[j]==i ) {
        
        grpleader++;
        if (grpleader==1 && seeds[j].nHits()<3 )  keep_all=true;

        PTrajectoryStateOnDet pTSOD = (seeds[j]).startingState();
	DetId SeedDetId(pTSOD.detId());
	const GeomDet* geoDet = theService->trackingGeometry()->idToDet( SeedDetId );
	TrajectoryStateOnSurface SeedTSOS = tsTransform.transientState(pTSOD, &(geoDet->surface()), &*theService->magneticField());
	GlobalVector seed_m   = SeedTSOS.globalMomentum();
        GlobalPoint  seed_xyz = SeedTSOS.globalPosition();
	double seed_mt = sqrt ( (seed_m.x()*seed_m.x()) + (seed_m.y()*seed_m.y()) );
        std::vector<float> err_mx = pTSOD.errorMatrix();
        double ddx = err_mx[2]; 
        double ddy = err_mx[5]; 
        double dxx = err_mx[9]; 
        double dyy = err_mx[14];
        
        double dRR = sqrt (ddx*36. + ddy*36 + dxx + dyy ) ; 
        double relErr = fabs(sqrt(err_mx[0]) / pTSOD.parameters().signedInverseMomentum()) ; 
        //std::cout<<"     seeds "<<j<<" dRErr: "<<dRR<<"  pt= "<<seed_mt<<" dPt:"<<relErr <<" with "<<seeds[j].nHits()<<" segs "<<std::endl;

        // this cut only apply for Endcap muon system 
	if ( (seed_mt <= 5.0 || seed_mt > 5000.0 ) && ( grpSize[i] > 1 ) ) continue;

        if ( keep_all ) {
           goodSeeds.push_back(seeds[j]);
        } else {
          // pick the one associated with more segments
          if ( seeds[j].nHits() > segSize ) {
             bestseed = j;
             segSize = seeds[j].nHits();
             dRErr = sqrt (ddx*36. + ddy*36 + dxx + dyy ) ;
             drelE = relErr ;
          } 
          // or pick the one with better relative error
          else if ( (seeds[j].nHits() == segSize) && (drelE > relErr) ){
             bestseed = j;
             dRErr = sqrt (ddx*36. + ddy*36 + dxx + dyy);
             drelE = relErr;
          } 
          else if ( (seeds[j].nHits() == segSize) && (drelE == relErr) && (dRErr > dRR) ){
             bestseed =j;
             dRErr = sqrt (ddx*36. + ddy*36 + dxx + dyy);
             drelE = relErr;
          } else {
            if (debug) std::cout<<"skip this seed "<<std::endl; 
          }
          //----------------------------------------------------
        }

      }  
    }
    //std::cout<<"best seeds = "<< bestseed <<std::endl;
    if ( bestseed > -1 ) {
       goodSeeds.push_back( seeds[bestseed] );
       //std::cout<<"seeds "<<bestseed<<" dRErr: "<<dRErr<<"  pt= "<<pt1<<" dPt:"<<drelE <<" with "<<seeds[bestseed].nHits()<<" segs @h: "<<h1 <<std::endl;
    }    
  }
  //std::cout<<" goodSeed size= "<<goodSeeds.size()<<std::endl;
  seeds.clear();
  seeds = goodSeeds ;
  //std::cout<<" ------------ finished 1 seeds set----------------- " <<std::endl;

}

