#ifndef CommonTools_Utils_IntSetter_h
#define CommonTools_Utils_IntSetter_h
/* \class reco::parser::IntSetter
 *
 * Integer setter
 *
 * \author  Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "CommonTools/Utils/src/IntStack.h"

namespace reco {
  namespace parser {
    struct IntSetter {
      IntSetter( IntStack & stack ) : stack_( stack ) { }
      void operator()(int n) const {
	stack_.push_back(n);
      }
    private:
      IntStack & stack_;
    };
  }
}

#endif
