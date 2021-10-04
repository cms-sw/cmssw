#include "DataFormats/CSCRecHit/interface/CSCJetCandidate.h"

/*
 * Create LeafCandidate with (eta,phi) only as CSC rechits has no energy/momentum measurement
 * Pt is set to 1.0 as a place holder, mass is set at 0.
 * Vertex associated with the CSC rechit is set to the origin.
 *
 */
reco::CSCJetCandidate::CSCJetCandidate(double phi, double eta, float x,float y, float z,float tPeak, float tWire,
                                       int quality, int chamber, int station, int nStrips, int hitWire, int wgroupsBX, int nWireGroups):
             LeafCandidate(0,LorentzVector(math::PtEtaPhiMLorentzVector(1.0,eta,phi,0)),Point(0,0,0)) {
    
    x_ = x;
    y_ = y;
    z_ = z;
    tPeak_ = tPeak;
    tWire_ = tWire;
    quality_      =    quality;     
    chamber_      =    chamber; 
    station_      =    station; 
    nStrips_      =    nStrips;
    hitWire_      =    hitWire;
    wgroupsBX_    =    wgroupsBX; 
    nWireGroups_  =    nWireGroups;

}

reco::CSCJetCandidate::~CSCJetCandidate() {

}
