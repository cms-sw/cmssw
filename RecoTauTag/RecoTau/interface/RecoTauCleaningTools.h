#ifndef RecoTauTag_RecoTau_RecoTauCleaningTools_h
#define RecoTauTag_RecoTau_RecoTauCleaningTools_h

#include <algorithm>

namespace reco::tau {

  template <typename RankingList, typename Type>
  class RecoTauLexicographicalRanking {
  public:
    // Store our list of ranking functions and intialize the vectors
    // that hold the comparison result
    explicit RecoTauLexicographicalRanking(const RankingList& rankers) : rankers_(rankers) {}
    // Predicate to compare a and b
    bool operator()(const Type& a, const Type& b) const {
      for (auto const& ranker : rankers_) {
        double aResult = (*ranker)(a);
        double bResult = (*ranker)(b);
        if (aResult != bResult)
          return (aResult < bResult);
      }
      // If all aare equal return false
      return false;
    }

  private:
    const RankingList& rankers_;
  };

  template <typename Container, class OverlapFunction>
  Container cleanOverlaps(const Container& dirty) {
    // Output container of clean objects
    Container clean;
    OverlapFunction overlapChecker;
    for (auto const& candidate : dirty) {
      // Check if this overlaps with a pizero already in the clean list
      bool overlaps = false;
      for (auto cleaned = clean.begin(); cleaned != clean.end() && !overlaps; ++cleaned) {
        overlaps = overlapChecker(candidate, *cleaned);
      }
      // If it didn't overlap with anything clean, add it to the clean list
      if (!overlaps)
        clean.insert(clean.end(), candidate);
    }
    return clean;
  }

}  // namespace reco::tau

#endif
