// Utilities.h

#ifndef L1Trigger_L1TMuonEndCap_emtf_Utilities
#define L1Trigger_L1TMuonEndCap_emtf_Utilities

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <utility>
#include "TMath.h"

namespace emtf {

//////////////////////////////////////////////////////////////////////////
// ------------------Some Helpful Things----------------------------------
//////////////////////////////////////////////////////////////////////////

// Array of counts for error calculation.
extern const Double_t twoJets_scale[16];
extern const std::vector<Double_t> twoJetsScale;

// Array of GeV values for error calculation.
extern const Double_t ptscale[31];
extern const std::vector<Double_t> ptScale;

template<class bidiiter>
bidiiter shuffle(bidiiter begin, bidiiter end, size_t num_random)
{
// We will end up with the same elements in the collection except that
// the first num_random elements will be randomized.

    size_t left = std::distance(begin, end);
    while (num_random--) {
        bidiiter r = begin;
        std::advance(r, rand()%left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

template <typename T>
std::string numToStr( T num )
{
// Convert a number to a string.
    std::stringstream ss;
    ss << num;
    std::string s = ss.str();
    return s;
};

float processPrediction(float BDTPt, int Quality, float PrelimFit);

void mergeNtuples(const char* ntuplename, const char* filestomerge, const char* outputfile);

void sortNtupleByEvent(const char* ntuplename, const char* filenametosort, const char* outputfile);

} // end of emtf namespace

#endif
