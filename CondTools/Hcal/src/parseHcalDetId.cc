#include <cstdlib>
#include <cerrno>
#include <cassert>
#include <vector>
#include <sstream>
#include <cstring>
#include <iterator>

#include "CondTools/Hcal/interface/parseHcalDetId.h"

static const char* subdetNames[] = {
    "",
    "HB",
    "HE",
    "HO",
    "HF",
};
static const unsigned nSsubdetNames = sizeof(subdetNames)/sizeof(subdetNames[0]);

static bool parseSubdetector(const char *c, HcalSubdetector* result)
{
    assert(c);
    assert(result);
    for (unsigned i=1; i<nSsubdetNames; ++i)
        if (strcmp(c, subdetNames[i]) == 0)
        {
            *result = static_cast<HcalSubdetector>(i);
            return true;
        }
    return false;
}

static bool parse_int(const char *c, int *result)
{
    assert(c);
    assert(result);
    char *endptr;
    errno = 0;
    *result = strtol(c, &endptr, 0);
    return !errno && *endptr == '\0';
}

const char* hcalSubdetectorName(HcalSubdetector subdet)
{
    const unsigned ind = static_cast<unsigned>(subdet);
    assert(ind < nSsubdetNames);
    return subdetNames[ind];
}

HcalDetId parseHcalDetId(const std::string& s)
{
    using namespace std;

    // Expected string contents:
    //
    //   ieta  iphi  depth  subdetector
    //
    // subdetector is one of "HB", "HE", "HF", or "HO"
    //
    HcalDetId result;
    istringstream iss(s);
    vector<string> tokens(istream_iterator<string>{iss},
                          istream_iterator<string>{});
    if (tokens.size() == 4)
    {
        HcalSubdetector subdet;
        int ieta, iphi, depth;
        if (parse_int(tokens[0].c_str(), &ieta)
            && parse_int(tokens[1].c_str(), &iphi)
            && parse_int(tokens[2].c_str(), &depth)
            && parseSubdetector(tokens[3].c_str(), &subdet)
            )
            result = HcalDetId(subdet, ieta, iphi, depth);
    }
    return result;
}
