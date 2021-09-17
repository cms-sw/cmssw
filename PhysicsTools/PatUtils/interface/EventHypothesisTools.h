#ifndef PhysicsTools_PatUtils_interface_EventHypothesisTools_h
#define PhysicsTools_PatUtils_interface_EventHypothesisTools_h

#include "DataFormats/PatCandidates/interface/EventHypothesis.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <memory>
#include <vector>

namespace pat {
  namespace eventhypothesis {

    /** Does the AND of some filters. OWNS the pointers to the filters */
    class AndFilter : public ParticleFilter {
    public:
      AndFilter() : filters_(2) {}
      AndFilter(ParticleFilter *f1, ParticleFilter *f2);
      ~AndFilter() override {}
      AndFilter &operator&=(ParticleFilter *filter) {
        filters_.emplace_back(filter);
        return *this;
      }
      bool operator()(const CandRefType &cand, const std::string &role) const override;

    private:
      std::vector<std::unique_ptr<ParticleFilter>> filters_;
    };

    /** Does the OR of some filters. OWNS the pointers to the filters */
    class OrFilter : public ParticleFilter {
    public:
      OrFilter() : filters_(2) {}
      OrFilter(ParticleFilter *f1, ParticleFilter *f2);
      ~OrFilter() override {}
      OrFilter &operator&=(ParticleFilter *filter) {
        filters_.emplace_back(filter);
        return *this;
      }
      bool operator()(const CandRefType &cand, const std::string &role) const override;

    private:
      std::vector<std::unique_ptr<ParticleFilter>> filters_;
    };

    class ByPdgId : public ParticleFilter {
    public:
      explicit ByPdgId(int32_t pdgCode, bool alsoAntiparticle = true);
      ~ByPdgId() override {}
      bool operator()(const CandRefType &cand, const std::string &role) const override;

    private:
      int32_t pdgCode_;
      bool antiparticle_;
    };

    class ByString : public ParticleFilter {
    public:
      ByString(const std::string &cut);  // not putting the explicit on purpose, I want to see what happens
      ~ByString() override {}
      bool operator()(const CandRefType &cand, const std::string &role) const override;

    private:
      StringCutObjectSelector<reco::Candidate> sel_;
    };

  }  // namespace eventhypothesis
}  // namespace pat

#endif
