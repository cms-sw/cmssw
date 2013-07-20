#ifndef CommonTools_Utils_UnaryCutSetter_h
#define CommonTools_Utils_UnaryCutSetter_h
/* \class reco::parser::UnaryCutSetter
 *
 * Cut setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include "CommonTools/Utils/src/SelectorStack.h"
#include "CommonTools/Utils/src/LogicalUnaryOperator.h"

namespace reco {
  namespace parser {    
    template<typename Op>
    struct UnaryCutSetter {
      UnaryCutSetter(SelectorStack & selStack) :
	selStack_(selStack) { }
      void operator()(const char*, const char*) const {
	selStack_.push_back(SelectorPtr(new LogicalUnaryOperator<Op>(selStack_)));
      }     
      void operator()(const char&) const {
	const char * c;
	operator()(c, c);
      }     
      SelectorStack & selStack_;
    };
  }
 }

#endif
