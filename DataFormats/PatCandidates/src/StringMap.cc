#include "DataFormats/PatCandidates/interface/StringMap.h"

void StringMap::add(const std::string &string, int32_t value) {
    entries_.push_back(value_type(string,value));
}

void StringMap::sort() {
    std::sort(entries_.begin(), entries_.end());
}

void StringMap::clear() {
    entries_.clear();
}

int32_t StringMap::operator[](const std::string &string) const {
    vector_type::const_iterator match =  std::lower_bound(entries_.begin(), entries_.end(), string, MatchByString());
    return (match == end() ? -1 : match->second);
}

const std::string & StringMap::operator[](int32_t number) const {
    static const std::string empty_;
    vector_type::const_iterator match = find(number);
    return (match == end() ? empty_ : match->first);
}

StringMap::const_iterator StringMap::find(const std::string &string) const {
    vector_type::const_iterator match =  std::lower_bound(entries_.begin(), entries_.end(), string, MatchByString());
    return (match->first == string ? match : end());
}

StringMap::const_iterator StringMap::find(int32_t number) const {
    return std::find_if(entries_.begin(), entries_.end(), MatchByNumber(number)); 
}


