#ifndef CommonTools_Utils_Comparison_h
#define CommonTools_Utils_Comparison_h
/* \class reco::parser::Comparison
 *
 * Comparison template
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/ComparisonBase.h"

namespace reco {
  namespace parser {
    template <class CompT>
    struct Comparison : public ComparisonBase {
      bool compare(double lhs, double rhs) const override { return comp(lhs, rhs); }

    private:
      CompT comp;
    };
  }  // namespace parser
}  // namespace reco

#endif
