#ifndef RecoJets_JetCharge_JetCharge_H
#define RecoJets_JetCharge_JetCharge_H

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector.h"
//#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <Math/VectorUtil.h>
#include <Math/Rotation3D.h>
#include <Math/RotationZ.h>
#include <Math/RotationY.h>
#include <cmath>

class JetCharge {
public:
	enum Variable { Pt, RelPt, RelEta, DeltaR, Unit };
	typedef reco::Particle::LorentzVector  LorentzVector;
	typedef reco::Particle::Vector         Vector;

	JetCharge(Variable var, double exponent=1.0) : var_(var), exp_(exponent) { }
	JetCharge(const edm::ParameterSet &iCfg) ;

	double charge(const LorentzVector &lv, const reco::TrackCollection &vec) const ;
	double charge(const LorentzVector &lv, const reco::TrackRefVector &vec) const ;
	double charge(const LorentzVector &lv, const reco::CandidateCollection &vec) const ;
	double charge(const reco::Candidate &parent) const ;
	//double charge(const LorentzVector &lv, const reco::CandidateRefVector &vec) const ;

private:
	// functions
        template<typename T, typename C>
	double chargeFromRef(const LorentzVector &lv, const C &vec) const ;
        template<typename T, typename C>
	double chargeFromVal(const LorentzVector &lv, const C &vec) const ;
        template<typename T, typename IT>
        double chargeFromValIterator(const LorentzVector &lv, const IT &begin, const IT &end) const ;

        template <class T>
	double getWeight(const LorentzVector &lv, const T& obj) const ;

	// data members
	Variable var_; double exp_;


};
template<typename T, typename IT>
double JetCharge::chargeFromValIterator(const LorentzVector &lv, const IT &begin, const IT &end) const {
    double num = 0.0, den = 0.0;
    for (IT it = begin; it != end ; ++it) {
        const T & obj = *it;
        double w = getWeight(lv, obj);
        den += w;
        num += w * obj.charge(); 
    }
    return (den > 0.0 ? num/den : 0.0);
}

template<typename T, typename C>
double JetCharge::chargeFromVal(const LorentzVector &lv, const C &vec) const {
    typedef typename C::const_iterator IT;
    return JetCharge::chargeFromValIterator<T,IT>(lv, vec.begin(), vec.end());
}

template<typename T, typename C>
double JetCharge::chargeFromRef(const LorentzVector &lv, const C &vec) const {
    typedef typename C::const_iterator IT;
    double num = 0.0, den = 0.0;
    for (IT it = vec.begin(), end = vec.end(); it != end ; ++it) {
        const T & obj = *it;
        double w = getWeight(lv, *obj);
        den += w;
        num += w * obj->charge();
    }
    return (den > 0.0 ? num/den : 0.0);
}

template <class T>
double JetCharge::getWeight(const LorentzVector &lv, const T& obj) const { 
    double ret;
    switch (var_) {
        case Pt: 
            ret = obj.pt(); 
            break;
        case DeltaR: 
            ret = ROOT::Math::VectorUtil::DeltaR(lv.Vect(), obj.momentum());
            break;
        case RelPt: 
        case RelEta: 
            ret =  lv.Vect().Dot(obj.momentum())/(lv.P() * obj.p()); // cos(theta)
            ret =  (var_ == RelPt ? 
                std::sqrt(1 - ret*ret) * obj.p() :    // p * sin(theta) = pt
            - 0.5 * std::log((1-ret)/(1+ret)));   // = - log tan theta/2 = eta
            break;
        case Unit:
        default:
            ret = 1.0;
    }
    return (exp_ == 1.0 ? ret :  (ret > 0 ? 
                                std::pow(ret,exp_) : 
                                - std::pow(std::abs(ret), exp_)));
}


#endif
