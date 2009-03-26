#ifndef PhysicsTools_PatUtils_interface_EventHypothesisTools_h
#define PhysicsTools_PatUtils_interface_EventHypothesisTools_h

#include "boost/ptr_container/ptr_vector.hpp"
#include "DataFormats/PatCandidates/interface/EventHypothesis.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

namespace pat { namespace eventhypothesis {

            /** Does the AND of some filters. OWNS the pointers to the filters */
            class AndFilter : public ParticleFilter {
                public:
                    AndFilter() : filters_(2) {}
                    AndFilter(ParticleFilter *f1, ParticleFilter *f2) ;
                    virtual ~AndFilter() {}
                    AndFilter & operator&=(ParticleFilter *filter) { filters_.push_back(filter); return *this; }
                    virtual bool operator()(const CandRefType &cand, const std::string &role) const ;
                private:
                    boost::ptr_vector<ParticleFilter> filters_;
            };

            /** Does the OR of some filters. OWNS the pointers to the filters */
            class OrFilter : public ParticleFilter {
                public:
                    OrFilter() : filters_(2) {}
                    OrFilter(ParticleFilter *f1, ParticleFilter *f2) ;
                    virtual ~OrFilter() {}
                    OrFilter & operator&=(ParticleFilter *filter) { filters_.push_back(filter); return *this; }
                    virtual bool operator()(const CandRefType &cand, const std::string &role) const ;
                private:
                    boost::ptr_vector<ParticleFilter> filters_;
            };

            class ByPdgId : public ParticleFilter {
                public:
                    explicit ByPdgId(int32_t pdgCode, bool alsoAntiparticle=true) ;
                    virtual ~ByPdgId() {}
                    virtual bool operator()(const CandRefType &cand, const std::string &role) const ;
                private:
                    int32_t pdgCode_;
                    bool    antiparticle_;
            };

            class ByString : public ParticleFilter {
                public:
                    ByString(const std::string &cut) ; // not putting the explicit on purpose, I want to see what happens
                    virtual ~ByString() {}
                    virtual bool operator()(const CandRefType &cand, const std::string &role) const ;
                private:
                    StringCutObjectSelector<reco::Candidate> sel_;
                    
            };



} }

#endif
