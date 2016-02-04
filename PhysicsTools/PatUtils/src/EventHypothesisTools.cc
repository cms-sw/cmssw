#include "PhysicsTools/PatUtils/interface/EventHypothesisTools.h"

using namespace pat::eventhypothesis;

AndFilter::AndFilter(ParticleFilter *f1, ParticleFilter *f2) :
    filters_(2) 
{
    filters_.push_back(f1); filters_.push_back(f2);
}

bool AndFilter::operator()(const CandRefType &cand, const std::string &role) const {
    for (boost::ptr_vector<ParticleFilter>::const_iterator it = filters_.begin(); it != filters_.end(); ++it) {
        if (! (*it)(cand, role) ) return false;
    }
    return true;
}

OrFilter::OrFilter(ParticleFilter *f1, ParticleFilter *f2) :
    filters_(2) 
{
    filters_.push_back(f1); filters_.push_back(f2);
}

bool OrFilter::operator()(const CandRefType &cand, const std::string &role) const {
    for (boost::ptr_vector<ParticleFilter>::const_iterator it = filters_.begin(); it != filters_.end(); ++it) {
        if ( (*it)(cand, role) ) return true;
    }
    return false;
}

ByPdgId::ByPdgId(int32_t pdgCode, bool alsoAntiparticle) :
    pdgCode_(alsoAntiparticle ? std::abs(pdgCode) : pdgCode),
    antiparticle_(alsoAntiparticle)
{
}

bool ByPdgId::operator()(const CandRefType &cand, const std::string &role) const {
    return antiparticle_ ? 
              (std::abs(cand->pdgId()) == pdgCode_) :
              (cand->pdgId() == pdgCode_);
}

ByString::ByString(const std::string &cut) : 
    sel_(cut)
{
}

bool ByString::operator()(const CandRefType &cand, const std::string &role) const {
    return sel_(*cand);
}
