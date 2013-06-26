#ifndef CommonTools_Utils_Comparison_h
#define CommonTools_Utils_Comparison_h
/* \class reco::parser::Comparison
 *
 * Comparison template
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include "CommonTools/Utils/src/ComparisonBase.h"

namespace reco {
  namespace parser {
    template<class CompT>
    struct Comparison : public ComparisonBase {
      virtual bool compare(double lhs, double rhs) const { return comp(lhs, rhs); }
    private:
      CompT comp;
    };
  }
}

#endif
