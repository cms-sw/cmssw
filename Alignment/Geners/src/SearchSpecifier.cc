#include "Alignment/Geners/interface/SearchSpecifier.hh"

namespace gs {
    bool SearchSpecifier::matches(const std::string& sentence) const
    {
        if (useRegex_)
#ifdef CPP11_STD_AVAILABLE
            return std::regex_match(sentence.begin(), sentence.end(), regex_);
#else
            return regex_.matches(sentence);
#endif
        else
            return sentence == tag_;
    }
}
