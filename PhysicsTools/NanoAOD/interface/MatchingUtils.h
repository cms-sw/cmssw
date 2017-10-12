#ifndef NanoAOD_MatchingUtils_h
#define NanoAOD_MatchingUtils_h

/*#include <utility>
#include <vector>
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
*/
template <class C1, class C2>
bool matchByCommonSourceCandidatePtr(const C1 & c1, const C2 & c2) {
    for(unsigned int i1 = 0 ; i1 < c1.numberOfSourceCandidatePtrs();i1++){
        auto  c1s=c1.sourceCandidatePtr(i1);
            for(unsigned int i2 = 0 ; i2 < c2.numberOfSourceCandidatePtrs();i2++) {
                if(c2.sourceCandidatePtr(i2)==c1s) return true;
            }
    }
    return false;
}

template <class C1, class C2>
bool matchByCommonParentSuperClusterRef(const C1 & c1, const C2  & c2) {
  auto c1s = c1.parentSuperCluster();
  auto c2s = c2.parentSuperCluster();
  return (c1s == c2s);
}

/*
template<typename I>
std::pair<const I &,float> bestMatch(auto item, auto targetColl,const StringCutObjectSelector<I> & cut="1") {
    float deltaR2Min = 9e99;
    const I & bm;
    for(const auto & t : targetColl){
	if(cut(t)) {
            float dR2 = deltaR2(item,t);
            if(dR2 < deltaR2Min){
                deltaR2Min = dR2;
                 bm = t;
   	    }
        }
    }
    return std::pair<const I &,float>(bm, deltaR2Min);
}


template<typename I>
std::vector<std::pair<const I &,float>> matchCollections(auto coll, auto targetColl,const StringCutObjectSelector<I> & cut="1") {
    std::vector<std::pair<I,float>> pairs;
    if(coll.empty()) return pairs;
    for(auto & p : coll){
	pairs.push_back(bestMatch(p,targetColl,cut));
    }
    return pairs;
}

*/

#endif
