//#include "PhysicsTools/FWLite/interface/FWLiteCandEvaluator.h"

#include "CommonTools/Utils/src/ExpressionPtr.h"
#include "CommonTools/Utils/src/ExpressionBase.h"
#include "CommonTools/Utils/src/SelectorPtr.h"
#include "CommonTools/Utils/src/SelectorBase.h"
#include <vector>
#include "PhysicsTools/FWLite/interface/ScannerHelpers.h"

// typedefs are useful, sometimes Root has problems with default template arguments (e.g. the allocator)
namespace reco { namespace parser {
    typedef std::vector<reco::parser::ExpressionPtr> ExpressionPtrs;
    typedef std::vector<reco::parser::SelectorPtr> SelectorPtrs;
} }


namespace {
 struct dictionary {
    // all these are templates, so we need to instantiate them
    reco::parser::ExpressionPtr eptr;
    std::vector<reco::parser::ExpressionPtr> eptrs;
    reco::parser::SelectorPtr sptr;
    std::vector<reco::parser::SelectorPtr> sptrs;
 };
}
