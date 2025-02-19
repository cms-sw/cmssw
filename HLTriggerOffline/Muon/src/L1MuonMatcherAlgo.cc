#include "HLTriggerOffline/Muon/interface/L1MuonMatcherAlgo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

L1MuonMatcherAlgo::L1MuonMatcherAlgo(const edm::ParameterSet & iConfig) :
    prop_(iConfig),
    preselectionCut_(iConfig.existsAs<std::string>("preselection") ? iConfig.getParameter<std::string>("preselection") : ""),
    deltaR2_(std::pow(iConfig.getParameter<double>("maxDeltaR"),2)),
    deltaPhi_(iConfig.existsAs<double>("maxDeltaPhi") ? iConfig.getParameter<double>("maxDeltaPhi") : 10),
    sortByDeltaPhi_(iConfig.existsAs<bool>("sortByDeltaPhi") ? iConfig.getParameter<bool>("sortByDeltaPhi") : false)
{
}

L1MuonMatcherAlgo::~L1MuonMatcherAlgo() {}

void
L1MuonMatcherAlgo::init(const edm::EventSetup & iSetup) {
    prop_.init(iSetup);
}

bool
L1MuonMatcherAlgo::match(TrajectoryStateOnSurface & propagated, const l1extra::L1MuonParticle &l1, float &deltaR, float &deltaPhi) const {
    if (preselectionCut_(l1)) {
        GlobalPoint pos = propagated.globalPosition();
        double thisDeltaPhi = ::deltaPhi(double(pos.phi()),  l1.phi());
        double thisDeltaR2  = ::deltaR2(double(pos.eta()), double(pos.phi()), l1.eta(), l1.phi());
        if ((fabs(thisDeltaPhi) < deltaPhi_) && (thisDeltaR2 < deltaR2_)) {
            deltaR   = std::sqrt(thisDeltaR2);
            deltaPhi = thisDeltaPhi;
            return true;
        }
    }
    return false;
}

int
L1MuonMatcherAlgo::match(TrajectoryStateOnSurface & propagated, const std::vector<l1extra::L1MuonParticle> &l1s, float &deltaR, float &deltaPhi) const {
    return matchGeneric(propagated, l1s, preselectionCut_, deltaR, deltaPhi);
/*
    int match = -1;
    double minDeltaPhi = deltaPhi_;
    double minDeltaR2  = deltaR2_;
    GlobalPoint pos = propagated.globalPosition();
    for (int i = 0, n = l1s.size(); i < n; ++i) {
        const l1extra::L1MuonParticle &l1 = l1s[i];
        if (preselectionCut_(l1)) {
            double thisDeltaPhi = ::deltaPhi(double(pos.phi()),  l1.phi());
            double thisDeltaR2  = ::deltaR2(double(pos.eta()), double(pos.phi()), l1.eta(), l1.phi());
            if ((fabs(thisDeltaPhi) < deltaPhi_) && (thisDeltaR2 < deltaR2_)) { // check both
                if (sortByDeltaPhi_ ? (fabs(thisDeltaPhi) < fabs(minDeltaPhi)) : (thisDeltaR2 < minDeltaR2)) { // sort on one
                    match = i;
                    deltaR   = std::sqrt(thisDeltaR2);
                    deltaPhi = thisDeltaPhi;
                    if (sortByDeltaPhi_) minDeltaPhi = thisDeltaPhi; else minDeltaR2 = thisDeltaR2;
                }
            }
        }
    }
    return match;
*/
}


