#include "MuonAnalysis/MuonAssociators/interface/L1MuonMatcherAlgo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

L1MuonMatcherAlgo::L1MuonMatcherAlgo(const edm::ParameterSet & iConfig) :
    prop_(iConfig),
    preselectionCut_(iConfig.existsAs<std::string>("preselection") ? iConfig.getParameter<std::string>("preselection") : ""),
    deltaR2_(std::pow(iConfig.getParameter<double>("maxDeltaR"),2)),
    deltaPhi_(iConfig.existsAs<double>("maxDeltaPhi") ? iConfig.getParameter<double>("maxDeltaPhi") : 10),
    deltaEta_(iConfig.existsAs<double>("maxDeltaEta") ? iConfig.getParameter<double>("maxDeltaEta") : 10),
    sortByDeltaPhi_(iConfig.existsAs<bool>("sortByDeltaPhi") ? iConfig.getParameter<bool>("sortByDeltaPhi") : false),
    sortByDeltaEta_(iConfig.existsAs<bool>("sortByDeltaEta") ? iConfig.getParameter<bool>("sortByDeltaEta") : false),
    l1PhiOffset_(iConfig.existsAs<double>("l1PhiOffset") ? iConfig.getParameter<double>("l1PhiOffset") : 0)
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
        double thisDeltaPhi = ::deltaPhi(double(pos.phi()),  l1.phi()+l1PhiOffset_);
        double thisDeltaEta = pos.eta() - l1.eta();
        double thisDeltaR2  = ::deltaR2(double(pos.eta()), double(pos.phi()), l1.eta(), l1.phi()+l1PhiOffset_);
        if ((fabs(thisDeltaPhi) < deltaPhi_) && (fabs(thisDeltaEta) < deltaEta_) && (thisDeltaR2 < deltaR2_)) {
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
}


