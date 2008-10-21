#include "PhysicsTools/IsolationAlgos/interface/IsoDepositVetoFactory.h"

#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include <boost/regex.hpp>

// ---------- FIRST DEFINE NEW VETOS ------------
namespace reco { namespace isodeposit {
    class SwitchingEcalVeto : public AbsVeto {
        public:
            // creates SwitchingEcalVeto from another AbsVeto (which becomes owned by this veto) 
            SwitchingEcalVeto(AbsVeto *veto, bool isBarrel) :
                veto_(veto), barrel_(isBarrel) {}
            virtual bool veto(double eta, double phi, float value) const {
                return on_ ? veto_->veto(eta,phi,value) : false;
            }
            virtual void centerOn(double eta, double phi) {
                if ( (fabs(eta) < 1.479) == (barrel_) ) {
                    on_ = true;
                    veto_->centerOn(eta,phi);
                } else {
                    on_ = false;
                }
            }
        private:
            std::auto_ptr<AbsVeto> veto_;
            bool barrel_, on_;   
    };
} }

// ---------- THEN THE ACTUAL FACTORY CODE ------------
reco::isodeposit::AbsVeto *
IsoDepositVetoFactory::make(const char *string) {
    using namespace reco::isodeposit;

    static boost::regex 
        ecalSwitch("^Ecal(Barrel|Endcaps):(.*)"),
        threshold("Threshold\\((\\d+\\.\\d+)\\)"),
        cone("ConeVeto\\((\\d+\\.\\d+)\\)"),
        angleCone("AngleCone\\((\\d+\\.\\d+)\\)"),
        angleVeto("AngleVeto\\((\\d+\\.\\d+)\\)"),
        rectangularEtaPhiVeto("RectangularEtaPhiVeto\\(([+-]?\\d+\\.\\d+),([+-]?\\d+\\.\\d+),([+-]?\\d+\\.\\d+),([+-]?\\d+\\.\\d+)\\)"),
        number("^(\\d+\\.?|\\d*\\.\\d*)$");
    boost::cmatch match;
    
    if (regex_match(string, match, ecalSwitch)) {
        return new SwitchingEcalVeto(make(match[2].first), (match[1].first == "Barrel") );
    } else if (regex_match(string, match, threshold)) {
        return new ThresholdVeto(atof(match[1].first));
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
    } else {
        throw cms::Exception("Not Implemented") << "Veto " << string << " not implemented yet...";
    }
}
