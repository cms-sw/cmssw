#include "PhysicsTools/JetCharge/interface/JetCharge.h"

double JetCharge::charge(const LorentzVector &lv, const reco::CandidateCollection &vec) const {
    return chargeFromVal<reco::Candidate,reco::CandidateCollection>(lv, vec);
}

double JetCharge::charge(const LorentzVector &lv, const reco::TrackCollection &vec) const {
    return chargeFromVal<reco::Track,reco::TrackCollection>(lv, vec);
}

double JetCharge::charge(const LorentzVector &lv, const reco::TrackRefVector &vec) const {
    return chargeFromRef<reco::TrackRef,reco::TrackRefVector>(lv, vec);
}

double JetCharge::charge(const reco::Candidate &parent) const {
    return chargeFromValIterator<reco::Candidate,reco::Candidate::const_iterator>(parent.p4(),parent.begin(),parent.end()); 
}

JetCharge::JetCharge(const edm::ParameterSet &iCfg) :
var_(Pt),
exp_(iCfg.getParameter<double>("exp")) {
	std::string var = iCfg.getParameter<std::string>("var");
	if (var == "Pt") {
		var_ = Pt;
	} else if (var == "RelPt") {
		var_ = RelPt;
	} else if (var == "RelEta") {
		var_ = RelEta;
	} else if (var == "DeltaR") {
		var_ = DeltaR;
	} else if (var == "Unit") {
		var_ = Unit;
	} else  {
		throw cms::Exception("Configuration error") << "Unknown variable " 
                        << var.c_str() << " for computing jet charge";
	}
}

