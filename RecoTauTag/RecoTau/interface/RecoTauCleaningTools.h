#ifndef RecoTauTag_RecoTau_RecoTauCleaningTools_h
#define RecoTauTag_RecoTau_RecoTauCleaningTools_h

#include <algorithm>
#include <functional>

namespace reco { namespace tau {

template<typename RankingList, typename Type>
  class RecoTauLexicographicalRanking :
      public std::binary_function<Type, Type, bool> {
    public:
      // Store our list of ranking functions and intialize the vectors
      // that hold the comparison result
      explicit RecoTauLexicographicalRanking(const RankingList& rankers):
        rankers_(rankers) {}
      // Predicate to compare a and b
      bool operator()(const Type& a, const Type& b) const {
        typename RankingList::const_iterator ranker = rankers_.begin();
        while (ranker != rankers_.end()) {
          double aResult = (*ranker)(a);
          double bResult = (*ranker)(b);
          if (aResult != bResult)
            return (aResult < bResult);
          ++ranker;
        }
        // If all aare equal return false
        return false;
      }
    private:
      const RankingList& rankers_;
  };

template<typename Container, class OverlapFunction>
Container cleanOverlaps(const Container& dirty) {
  typedef typename Container::const_iterator Iterator;
  // Output container of clean objects
  Container clean;
  OverlapFunction overlapChecker;
  for (Iterator candidate = dirty.begin(); candidate != dirty.end();
       ++candidate) {
    // Check if this overlaps with a pizero already in the clean list
    bool overlaps = false;
    for (Iterator cleaned = clean.begin();
         cleaned != clean.end() && !overlaps; ++cleaned) {
      overlaps = overlapChecker(*candidate, *cleaned);
    }
    // If it didn't overlap with anything clean, add it to the clean list
    if (!overlaps)
      clean.insert(clean.end(), *candidate);
  }
  return clean;
}

template<typename T>
class SortByDescendingPt {
  public:
    bool operator()(const T& a, const T& b) const {
      return a.pt() > b.pt();
    }
};

}}  // end reco::tau namespace

#endif
