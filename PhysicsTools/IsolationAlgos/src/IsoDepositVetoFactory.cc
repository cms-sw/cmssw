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
            virtual bool veto(double eta, double phi, float value) const override {
                return (fabs(eta) < 1.479) == (barrel_) ? veto_->veto(eta,phi,value) : false;
            }
            virtual void centerOn(double eta, double phi) override {
	      veto_->centerOn(eta,phi);
	    }
        private:
            std::auto_ptr<AbsVeto> veto_;
            bool barrel_;
    };

    class NumCrystalVeto : public AbsVeto {
        public:
            NumCrystalVeto(Direction dir, double iR) : vetoDir_(dir), iR_(iR) {}
            virtual bool veto(double eta, double phi, float value) const override {
                if( fabs(vetoDir_.eta()) < 1.479) {
                    return ( vetoDir_.deltaR(Direction(eta,phi)) < 0.0174*iR_ );
                } else {
                    return ( vetoDir_.deltaR(Direction(eta,phi)) < 0.00864*fabs(sinh(eta))*iR_ );
                }
            }
            virtual void centerOn(double eta, double phi) override { vetoDir_ = Direction(eta,phi); }
        private:
            Direction vetoDir_; float iR_;
    };

    class NumCrystalEtaPhiVeto : public AbsVeto {
        public:
            NumCrystalEtaPhiVeto(const math::XYZVectorD& dir, double iEta, double iPhi) :
                vetoDir_(dir.eta(),dir.phi()),
                iEta_(iEta),
                iPhi_(iPhi) {}
            NumCrystalEtaPhiVeto(Direction dir, double iEta, double iPhi) :
                vetoDir_(dir.eta(),dir.phi()),
                iEta_(iEta),
                iPhi_(iPhi) {}
            virtual bool veto(double eta, double phi, float value) const override {
                double dPhi = phi - vetoDir_.phi();
                double dEta = eta - vetoDir_.eta();
                while( dPhi < -M_PI )   dPhi += 2*M_PI;
                while( dPhi >= M_PI )   dPhi -= 2*M_PI;
                if( fabs(vetoDir_.eta()) < 1.479) {
                    return ( (fabs(dEta) < 0.0174*iEta_) && (fabs(dPhi) < 0.0174*iPhi_) );
                } else {
                    return ( (fabs(dEta) < 0.00864*fabs(sinh(eta))*iEta_) &&
                             (fabs(dPhi) < 0.00864*fabs(sinh(eta))*iPhi_) );
                }
            }
            virtual void centerOn(double eta, double phi) override { vetoDir_ = Direction(eta,phi); }
        private:
            Direction vetoDir_;
            double iEta_, iPhi_;
    };

} }

// ---------- THEN THE ACTUAL FACTORY CODE ------------
reco::isodeposit::AbsVeto *
IsoDepositVetoFactory::make(const char *string, edm::ConsumesCollector& iC) {
    reco::isodeposit::EventDependentAbsVeto * evdep = 0;
    std::auto_ptr<reco::isodeposit::AbsVeto> ret(make(string,evdep, iC));
    if (evdep != 0) {
        throw cms::Exception("Configuration") << "The resulting AbsVeto depends on the edm::Event.\n"
                                              << "Please use the two-arguments IsoDepositVetoFactory::make.\n";
    }
    return ret.release();
}

reco::isodeposit::AbsVeto *
IsoDepositVetoFactory::make(const char *string, reco::isodeposit::EventDependentAbsVeto *&evdep, edm::ConsumesCollector& iC) {
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
        numCrystal("NumCrystalVeto\\((\\d+\\.\\d+)\\)"),
        numCrystalEtaPhi("NumCrystalEtaPhiVeto\\((\\d+\\.\\d+),(\\d+\\.\\d+)\\)"),
        otherCandidatesDR("OtherCandidatesByDR\\((\\w+:?\\w*:?\\w*),\\s*(\\d+\\.?|\\d*\\.\\d*)\\)"),
        otherJetConstituentsDR("OtherJetConstituentsDeltaRVeto\\((\\w+:?\\w*:?\\w*),\\s*(\\d+\\.?|\\d*\\.\\d*),\\s*(\\w+:?\\w*:?\\w*),\\s*(\\d+\\.?|\\d*\\.\\d*)\\)"),
        otherCand("^(.*?):(.*)"),
        number("^(\\d+\\.?|\\d*\\.\\d*)$");
    boost::cmatch match;

    //std::cout << "<IsoDepositVetoFactory::make>:" << std::endl;
    //std::cout << " string = " << string << std::endl;

    evdep = 0; // by default it does not depend on this
    if (regex_match(string, match, ecalSwitch)) {
        return new SwitchingEcalVeto(make(match[2].first, iC), (match[1] == "Barrel") );
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
        return new RectangularEtaPhiVeto(Direction(),
                    atof(match[1].first), atof(match[2].first),
                    atof(match[3].first), atof(match[4].first));
    } else if (regex_match(string, match, numCrystal)) {
        return new NumCrystalVeto(Direction(), atof(match[1].first));
    } else if (regex_match(string, match, numCrystalEtaPhi)) {
        return new NumCrystalEtaPhiVeto(Direction(),atof(match[1].first),atof(match[2].first));
    } else if (regex_match(string, match, otherCandidatesDR)) {
        OtherCandidatesDeltaRVeto *ret = new OtherCandidatesDeltaRVeto(edm::InputTag(match[1]),
                                                                        atof(match[2].first),
                                                                        iC);
        evdep = ret;
        return ret;
    } else if (regex_match(string, match, otherJetConstituentsDR)) {
        OtherJetConstituentsDeltaRVeto *ret = new OtherJetConstituentsDeltaRVeto(Direction(),
						   edm::InputTag(match[1]), atof(match[2].first),
						   edm::InputTag(match[3]), atof(match[4].first),
						   iC);
        evdep = ret;
        return ret;
    } else if (regex_match(string, match, otherCand)) {
        OtherCandVeto *ret = new OtherCandVeto(edm::InputTag(match[1]),
                                                           make(match[2].first, iC), iC);
        evdep = ret;
        return ret;
    } else {
        throw cms::Exception("Not Implemented") << "Veto " << string << " not implemented yet...";
    }
}
