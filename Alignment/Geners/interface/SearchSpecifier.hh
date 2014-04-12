#ifndef GENERS_SEARCHSPECIFIER_HH_
#define GENERS_SEARCHSPECIFIER_HH_

#include <string>

#include "Alignment/Geners/interface/Regex.hh"

namespace gs {
    class SearchSpecifier
    {
    public:
        inline SearchSpecifier(const char* exact)
            : tag_(exact ? exact : ""), useRegex_(false) {}

        inline SearchSpecifier(const std::string& exact)
            : tag_(exact), useRegex_(false) {}

        inline SearchSpecifier(const Regex& regex)
            : regex_(regex), useRegex_(true) {}

        // Note that the C++11 regex object does not specify a way to
        // extract the regular expression from it. Sometimes, however,
        // it is useful to have a way to print the expression searched.
        // The following special constructor helps: it can be used in
        // case one wants to use regex but also wants to remember the
        // regular expression itself. This constructor can also be useful
        // in case the code gets a command line switch on whether to use
        // regular expressions or not. Note that, for C++11 regex, the
        // regex flavor can no longer be specified (default is used).
        //
        inline SearchSpecifier(const std::string& expr, const bool useRegex)
            : tag_(expr), regex_(useRegex ? expr : std::string()),
              useRegex_(useRegex) {}

        inline bool useRegex() const {return useRegex_;}
        inline const std::string& pattern() const {return tag_;}

        bool matches(const std::string& sentence) const;

    private:
        SearchSpecifier();

        std::string tag_;
        Regex regex_;
        bool useRegex_;
    };
}

#endif // GENERS_SEARCHSPECIFIER_HH_

