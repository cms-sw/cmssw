#include "PhysicsTools/PatAlgos/interface/OverlapHelper.h"

using pat::helper::OverlapHelper;
using pat::SimpleOverlapFinder;

OverlapHelper::OverlapHelper(const std::vector<edm::ParameterSet> &psets) {
    typedef std::vector<edm::ParameterSet>::const_iterator VPI;
    tests_.reserve(psets.size());
    for (VPI it = psets.begin(), ed = psets.end(); it != ed; ++it) {
       tests_.push_back( Test(  it->getParameter<edm::InputTag>("collection"),
                                pat::SimpleOverlapFinder(it->getParameter<double>("deltaR")) 
                              )
                        );
    }
}


std::auto_ptr<OverlapHelper::Result>
OverlapHelper::test( const edm::Event &iEvent, const std::vector<const reco::Candidate *> &in  ) const {
    using namespace std;
    using namespace edm;

    size_t size = in.size();
    auto_ptr<Result> ret(new Result(size, 0));
    Result &result = *ret; // pointers to vectors are more ugly work with than refs to vectors

    int testBit = 1;
    for (vector<Test>::const_iterator itt = tests_.begin(), edt = tests_.end(); itt != edt; ++itt, testBit <<= 1) {
        // read data
        Handle< View<reco::Candidate> > hView;
        iEvent.getByLabel(itt->src, hView);
        // test overlaps
        auto_ptr<SimpleOverlapFinder::Overlaps> overlaps = itt->finder.find(in, *hView);
        // mark items
        for (SimpleOverlapFinder::Overlaps::const_iterator ito = overlaps->begin(), edo = overlaps->end(); ito != edo; ++ito) {
            result[ito->first] += testBit;
        }
    }

    return ret;
}

