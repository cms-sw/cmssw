#include "DataFormats/CSCRecHit/interface/CSCJetCandidate.h"

/*
 * Create LeafCandidate with (eta,phi) only as CSC rechits has no energy/momentum measurement
 * Energy is set to 1.0 as a place holder, mass is set at 0.
 * Vertex associated with the CSC rechit is set to the origin.
 *
 */
reco::CSCJetCandidate::CSCJetCandidate(double phi, double eta, double x,double y, double z,double tPeak, double tWire):
             LeafCandidate(0,LorentzVector(math::PtEtaPhiMLorentzVector(1.0,eta,phi,0)),Point(0,0,0)) {
    
    x_ = x;
    y_ = y;
    z_ = z;
    tPeak_ = tPeak;
    tWire_ = tWire;
}

reco::CSCJetCandidate::~CSCJetCandidate() {

}
