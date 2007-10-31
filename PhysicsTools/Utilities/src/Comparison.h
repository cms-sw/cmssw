#ifndef Utilities_Comparison_h
#define Utilities_Comparison_h
/* \class reco::parser::Comparison
 *
 * Comparison template
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/ComparisonBase.h"

namespace reco {
  namespace parser {
    template<class CompT>
    struct Comparison : public ComparisonBase {
      virtual bool compare( double lhs, double rhs ) const { return comp( lhs, rhs ); }
    private:
      CompT comp;
    };
  }
}

#endif
