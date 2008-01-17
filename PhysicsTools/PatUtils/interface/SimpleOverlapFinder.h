#ifndef PhysicsTools_PatUtils_SimpleOverlapFinder_h
#define PhysicsTools_PatUtils_SimpleOverlapFinder_h

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/Math/interface/deltaR.h"

/**
  \brief Helper class to check for overlaps (only uses deltaR)
  Given two lists of candidates (e.g. jets and electrons) provides the list of indices of items in the first list that fall within a specific deltaR from at least one item in the second.
*/
#include <memory>
#include <vector>

namespace pat { 
    
    class SimpleOverlapFinder {
        public:
            typedef std::pair<size_t, const reco::Candidate *> Overlap;
            typedef std::vector<Overlap> Overlaps;

            SimpleOverlapFinder() : deltaR2_(-1.0) { }
            explicit SimpleOverlapFinder(double deltaR) : deltaR2_(deltaR * deltaR) { }
            ~SimpleOverlapFinder() { }

            std::auto_ptr< Overlaps > find (
                const std::vector< const reco::Candidate * > &toClean,
                const std::vector< const reco::Candidate * > &theOthers) const ;
            
             std::auto_ptr< Overlaps > find (
                const std::vector< const reco::Candidate * > &toClean,
                const edm::View<reco::Candidate> &theOthers) const ;
     private:
            double deltaR2_;
            
    };
}

#endif
