#include "PhysicsTools/IsolationAlgos/interface/IsoDepositVetoFactory.h"

#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "PhysicsTools/IsolationAlgos/interface/EventDependentAbsVetos.h"
#include <boost/regex.hpp>

// ---------- FIRST DEFINE NEW VETOS ------------
namespace reco { namespace isodeposit {
    class SwitchingEcalVeto : public AbsVeto {
        public:
            // creates SwitchingEcalVeto from another AbsVeto (which becomes owned by this veto) 
            SwitchingEcalVeto(AbsVeto *veto, bool isBarrel) :
                veto_(veto), barrel_(isBarrel) {}
            virtual bool veto(double eta, double phi, float value) const {
                return (fabs(eta) < 1.479) == (barrel_) ? veto_->veto(eta,phi,value) : false;
            }
            virtual void centerOn(double eta, double phi) {
	      veto_->centerOn(eta,phi);
	    }
        private:
            std::auto_ptr<AbsVeto> veto_;
            bool barrel_;   
    };
} }

// ---------- THEN THE ACTUAL FACTORY CODE ------------
reco::isodeposit::AbsVeto *
IsoDepositVetoFactory::make(const char *string) {
    reco::isodeposit::EventDependentAbsVeto * evdep = 0;
    std::auto_ptr<reco::isodeposit::AbsVeto> ret(make(string,evdep));
    if (evdep != 0) {
        throw cms::Exception("Configuration") << "The resulting AbsVeto depends on the edm::Event.\n" 
                                              << "Please use the two-arguments IsoDepositVetoFactory::make.\n";
    }
    return ret.release();
}

reco::isodeposit::AbsVeto *
IsoDepositVetoFactory::make(const char *string, reco::isodeposit::EventDependentAbsVeto *&evdep) {
    using namespace reco::isodeposit;
    static boost::regex 
        ecalSwitch("^Ecal(Barrel|Endcaps):(.*)"),
        threshold("Threshold\\((\\d+\\.\\d+)\\)"),
        thresholdtransverse("ThresholdFromTransverse\\((\\d+\\.\\d+)\\)"),
        absthreshold("AbsThreshold\\((\\d+\\.\\d+)\\)"),
        absthresholdtransverse("AbsThresholdFromTransverse\\((\\d+\\.\\d+)\\)"),
        cone("ConeVeto\\((\\d+\\.\\d+)\\)"),
        angleCone("AngleCone\\((\\d+\\.\\d+)\\)"),
        angleVeto("AngleVeto\\((\\d+\\.\\d+)\\)"),
        rectangularEtaPhiVeto("RectangularEtaPhiVeto\\(([+-]?\\d+\\.\\d+),([+-]?\\d+\\.\\d+),([+-]?\\d+\\.\\d+),([+-]?\\d+\\.\\d+)\\)"),
        otherCandidates("OtherCandidatesByDR\\((\\w+:?\\w*:?\\w*),\\s*(\\d+\\.?|\\d*\\.\\d*)\\)"),
        number("^(\\d+\\.?|\\d*\\.\\d*)$");
    boost::cmatch match;
    
    evdep = 0; // by default it does not depend on this
    if (regex_match(string, match, ecalSwitch)) {
        return new SwitchingEcalVeto(make(match[2].first), (match[1] == "Barrel") );
    } else if (regex_match(string, match, threshold)) {
        return new ThresholdVeto(atof(match[1].first));
    } else if (regex_match(string, match, thresholdtransverse)) {
        return new ThresholdVetoFromTransverse(atof(((std::string)match[1]).c_str()));
    } else if (regex_match(string, match, absthreshold)) {
        return new AbsThresholdVeto(atof(match[1].first));
    } else if (regex_match(string, match, absthresholdtransverse)) {
        return new AbsThresholdVetoFromTransverse(atof(((std::string)match[1]).c_str()));
    } else if (regex_match(string, match, cone)) {
        return new ConeVeto(Direction(), atof(match[1].first));
    } else if (regex_match(string, match, number)) {
        return new ConeVeto(Direction(), atof(match[1].first));
    } else if (regex_match(string, match, angleCone)) {
        return new AngleCone(Direction(), atof(match[1].first));
    } else if (regex_match(string, match, angleVeto)) {
        return new AngleConeVeto(Direction(), atof(match[1].first));
    } else if (regex_match(string, match, rectangularEtaPhiVeto)) {
        return new RectangularEtaPhiVeto(reco::isodeposit::Direction(), 
                    atof(match[1].first), atof(match[2].first), 
                    atof(match[3].first), atof(match[4].first));
    } else if (regex_match(string, match, otherCandidates)) {
        OtherCandidatesDeltaRVeto *ret = new OtherCandidatesDeltaRVeto(edm::InputTag(match[1]), 
                                                                        atof(match[2].first));
        evdep = ret;
        return ret;
    } else {
        throw cms::Exception("Not Implemented") << "Veto " << string << " not implemented yet...";
    }
}
