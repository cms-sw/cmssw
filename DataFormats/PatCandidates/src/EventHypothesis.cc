#include "DataFormats/PatCandidates/interface/EventHypothesis.h"
#include "DataFormats/PatCandidates/interface/EventHypothesisLooper.h"

char *
pat::EventHypothesis::getDemangledSymbol(const char* mangledSymbol) const {
    int status;
    char *demangledSymbol = abi::__cxa_demangle(mangledSymbol, nullptr, nullptr, &status);
    return (status == 0) ? demangledSymbol : nullptr;
}

void pat::EventHypothesis::add(const CandRefType &ref, const std::string &role) {
    particles_.push_back(value_type(role,ref));
}

const pat::EventHypothesis::CandRefType & 
pat::EventHypothesis::get(const std::string &role, int index) const 
{
    if (index >= 0) {
        const_iterator it = realGet(begin(), end(), ByRole(role), index);
        if (it == end()) { throw cms::Exception("Index not found") << "Can't find a particle with role " << role << " and index " << index << "\n"; }
        return it->second;
    } else {
        const_reverse_iterator it = realGet(rbegin(), rend(), ByRole(role), -index);
        if (it == rend()) { throw cms::Exception("Index not found") << "Can't find a particle with role " << role << " and index " << index << "\n"; }
        return it->second;
    }
}

const pat::EventHypothesis::CandRefType &
pat::EventHypothesis::get(const ParticleFilter &filter, int index) const 
{
    if (index >= 0) {
        const_iterator it = realGet(begin(), end(), filter, index);
        if (it == end()) { throw cms::Exception("Index not found") << "Can't find a particle matching filter with index " << index << "\n"; }
        return it->second;
    } else {
        const_reverse_iterator it = realGet(rbegin(), rend(), filter, -index);
        if (it == rend()) { throw cms::Exception("Index not found") << "Can't find a particle matching filter with index " << index << "\n"; }
        return it->second;
    }
}


std::vector<pat::EventHypothesis::CandRefType> 
pat::EventHypothesis::all(const std::string &roleRegexp) const 
{
    return all(pat::eventhypothesis::RoleRegexpFilter(roleRegexp));
}

std::vector<pat::EventHypothesis::CandRefType> 
pat::EventHypothesis::all(const ParticleFilter &filter) const 
{
    std::vector<pat::EventHypothesis::CandRefType> ret;
    for (const_iterator it = begin(); it != end(); ++it) {
        if (filter(*it)) ret.push_back(it->second);
    }
    return ret;
}

size_t
pat::EventHypothesis::count(const std::string &roleRegexp) const 
{
    return count(pat::eventhypothesis::RoleRegexpFilter(roleRegexp));
}

size_t
pat::EventHypothesis::count(const ParticleFilter &role) const 
{
    size_t n = 0;
    for (const_iterator it = begin(); it != end(); ++it) {
        if (role(*it)) ++n;
    }
    return n;
}

pat::EventHypothesis::CandLooper
pat::EventHypothesis::loop() const 
{
    return loop(pat::eventhypothesis::AcceptAllFilter::get());
}

pat::EventHypothesis::CandLooper
pat::EventHypothesis::loop(const std::string &roleRegexp) const
{
    return loop(new pat::eventhypothesis::RoleRegexpFilter(roleRegexp));
}

pat::EventHypothesis::CandLooper
pat::EventHypothesis::loop(const ParticleFilter &role) const 
{
    return CandLooper(*this, role); 
}

pat::EventHypothesis::CandLooper
pat::EventHypothesis::loop(const ParticleFilter *role) const 
{
    return CandLooper(*this, role); 
}

pat::EventHypothesis::CandLooper
pat::EventHypothesis::loop(const ParticleFilterPtr &role) const 
{
    return CandLooper(*this, role); 
}

const pat::eventhypothesis::AcceptAllFilter pat::eventhypothesis::AcceptAllFilter::s_dummyFilter;
