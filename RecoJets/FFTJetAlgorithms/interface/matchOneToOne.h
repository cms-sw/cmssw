//
// Utility function for matching elements of one vector to another
// using best matching distance. All pairwise distances are calculated
// and then sorted; the best match corresponds to the smallest
// distance. Both matched elements are removed from the pool,
// then the best match is found among the remaining elements, etc.
//
// I. Volobouev
// April 2008

#ifndef RecoJets_FFTJetAlgorithms_matchOneToOne_h
#define RecoJets_FFTJetAlgorithms_matchOneToOne_h

#include <vector>
#include <algorithm>

namespace fftjetcms {
    namespace Private {
        struct matchOneToOne_MatchInfo
        {
            double distance;
            unsigned i1;
            unsigned i2;

            inline bool operator<(const matchOneToOne_MatchInfo& r) const
                {return distance < r.distance;}

            inline bool operator>(const matchOneToOne_MatchInfo& r) const
                {return distance > r.distance;}
        };
    }

    //
    // (*matchFrom1To2)[idx] will be set to the index of the element in v2
    // which corresponds to the element at index "idx" in v1. If no match
    // to the element at index "idx" in v1 is found, (*matchFrom1To2)[idx]
    // is set to -1. All non-negative (*matchFrom1To2)[idx] values will be
    // unique. The function returns the total number of matches made.
    //
    template <class T1, class T2, class DistanceCalculator>
    unsigned matchOneToOne(const std::vector<T1>& v1, const std::vector<T2>& v2,
                           const DistanceCalculator& calc,
                           std::vector<int>* matchFrom1To2,
                           const double maxMatchingDistance = 1.0e300)
    {
        unsigned nused = 0;
        matchFrom1To2->clear();

        const unsigned n1 = v1.size();
        if (n1)
        {
            matchFrom1To2->reserve(n1);
            for (unsigned i1=0; i1<n1; ++i1)
                matchFrom1To2->push_back(-1);

            const unsigned n2 = v2.size();
            if (n2)
            {
                const unsigned nmatches = n1*n2;
                std::vector<Private::matchOneToOne_MatchInfo> distanceTable(nmatches);
                std::vector<int> taken2(n2);

                for (unsigned i2=0; i2<n2; ++i2)
                    taken2[i2] = 0;

                Private::matchOneToOne_MatchInfo* m;
                for (unsigned i1=0; i1<n1; ++i1)
                    for (unsigned i2=0; i2<n2; ++i2)
                    {
                        m = &distanceTable[i1*n2+i2];
                        m->distance = calc(v1[i1], v2[i2]);
                        m->i1 = i1;
                        m->i2 = i2;
                    }

                std::sort(distanceTable.begin(), distanceTable.end());
                for (unsigned i=0; i<nmatches && nused<n1 && nused<n2; ++i)
                {
                    m = &distanceTable[i];
                    if (m->distance > maxMatchingDistance)
                        break;
                    if ((*matchFrom1To2)[m->i1] < 0 && !taken2[m->i2])
                    {
                        (*matchFrom1To2)[m->i1] = static_cast<int>(m->i2);
                        taken2[m->i2] = 1;
                        ++nused;
                    }
                }
            }
        }

        return nused;
    }
}

#endif // RecoJets_FFTJetAlgorithms_matchOneToOne_h
