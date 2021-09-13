#include "DataFormats/CSCRecHit/interface/CSCJetCandidate.h"


reco::CSCJetCandidate:CSCJetCandidate(double phi, double eta, double x,double y, double z,double t);
             LeafCandidate(0,LorentzVector(math::PtEtaPhiMLorentzVector(1.0,eta(),phi(),0)),Point(x,y,z)) {

    phi_ = phi;
    eta_ = eta;
    x_   = x;
    y_ = y;
    z_ = z;
}

reco::CSCJetCandidate::~CSCJetCandidate() {

}
