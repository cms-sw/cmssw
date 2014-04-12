#include "PhysicsTools/PatAlgos/interface/OverlapTest.h"

#include <algorithm>
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"

using namespace pat::helper;

void
BasicOverlapTest::readInput(const edm::Event & iEvent, const edm::EventSetup &iSetup)
{
    iEvent.getByToken(srcToken_, candidates_);
    isPreselected_.resize(candidates_->size());
    size_t idx = 0;
    for (reco::CandidateView::const_iterator it = candidates_->begin(); it != candidates_->end(); ++it, ++idx) {
        isPreselected_[idx] = presel_(*it);
    }
    // Yes, I could use std::transform. But would people like it?
    // http://www.sgi.com/tech/stl/transform.html
}

bool
BasicOverlapTest::fillOverlapsForItem(const reco::Candidate &item, reco::CandidatePtrVector &overlapsToFill) const
{
    size_t idx = 0;
    std::vector<std::pair<float,size_t> > matches;
    for (reco::CandidateView::const_iterator it = candidates_->begin(); it != candidates_->end(); ++it, ++idx) {
        if (!isPreselected_[idx]) continue;
        double dr = reco::deltaR(item, *it);
        if (dr < deltaR_) {
            if (checkRecoComponents_) {
                OverlapChecker overlaps;
                if (!overlaps(item, *it)) continue;
            }
            if (!pairCut_(pat::DiObjectProxy(item,*it))) continue;
            matches.push_back(std::make_pair(dr, idx));
        }
    }
    // see if we matched anything
    if (matches.empty()) return false;

    // sort matches
    std::sort(matches.begin(), matches.end());
    // fill ptr vector
    for (std::vector<std::pair<float,size_t> >::const_iterator it = matches.begin(); it != matches.end(); ++it) {
        overlapsToFill.push_back(candidates_->ptrAt(it->second));
    }
    return true;
}

bool
OverlapBySuperClusterSeed::fillOverlapsForItem(const reco::Candidate &item, reco::CandidatePtrVector &overlapsToFill) const
{
    const reco::RecoCandidate * input = dynamic_cast<const reco::RecoCandidate *>(&item);
    if (input == 0) throw cms::Exception("Type Error") << "Input to OverlapBySuperClusterSeed is not a RecoCandidate. "
                                                       << "It's a " << typeid(item).name() << "\n";
    reco::SuperClusterRef mySC = input->superCluster();
    if (mySC.isNull() || !mySC.isAvailable()) {
        throw cms::Exception("Bad Reference") << "Input to OverlapBySuperClusterSeed has a null or dangling superCluster reference\n";
    }
    const reco::CaloClusterPtr & mySeed = mySC->seed();
    if (mySeed.isNull()) {
        throw cms::Exception("Bad Reference") << "Input to OverlapBySuperClusterSeed has a null superCluster seed reference\n";
    }
    bool hasOverlaps = false;
    size_t idx = 0;
//     for (edm::View<reco::RecoCandidate>::const_iterator it = others_->begin(); it != others_->end(); ++it, ++idx) {
    for (reco::CandidateView::const_iterator it = others_->begin(); it != others_->end(); ++it, ++idx) {
        const reco::RecoCandidate * other = dynamic_cast<const reco::RecoCandidate *>(&*it);
        reco::SuperClusterRef otherSc = other->superCluster();
        if (otherSc.isNull() || !otherSc.isAvailable()) {
            throw cms::Exception("Bad Reference") << "One item in the OverlapBySuperClusterSeed input list has a null or dangling superCluster reference\n";
        }
        if (mySeed == otherSc->seed()) {
            overlapsToFill.push_back(others_->ptrAt(idx));
            hasOverlaps = true;
        }
    }
    return hasOverlaps;
}
