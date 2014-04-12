#ifndef CommonTools_Utils_TrinarySelectorSetter_h
#define CommonTools_Utils_TrinarySelectorSetter_h
/* \class reco::parser::TrinarySelectorSetter
 *
 * Trinary selector setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/src/SelectorStack.h"
#include "CommonTools/Utils/src/ComparisonStack.h"
#include "CommonTools/Utils/src/ExpressionStack.h"
#include "CommonTools/Utils/src/TrinarySelector.h"
#include <boost/shared_ptr.hpp>

namespace reco {
  namespace parser {    
    class TrinarySelectorSetter {
    public:
      TrinarySelectorSetter(SelectorStack& selStack,
			     ComparisonStack& cmpStack, 
			     ExpressionStack& expStack) : 
	selStack_(selStack), cmpStack_(cmpStack), expStack_(expStack) { }
      
      void operator()(const char *, const char *) const {
	boost::shared_ptr<ExpressionBase> rhs = expStack_.back(); expStack_.pop_back();
	boost::shared_ptr<ExpressionBase> mid = expStack_.back(); expStack_.pop_back();
	boost::shared_ptr<ExpressionBase> lhs = expStack_.back(); expStack_.pop_back();
	boost::shared_ptr<ComparisonBase> comp2 = cmpStack_.back(); cmpStack_.pop_back();
	boost::shared_ptr<ComparisonBase> comp1 = cmpStack_.back(); cmpStack_.pop_back();
#ifdef BOOST_SPIRIT_DEBUG 
	BOOST_SPIRIT_DEBUG_OUT << "pushing trinary selector" << std::endl;
#endif
	selStack_.push_back(SelectorPtr(new TrinarySelector(lhs, comp1, mid, comp2, rhs)));
      }
    private:
      SelectorStack& selStack_;
      ComparisonStack& cmpStack_;
      ExpressionStack& expStack_;
    };
  }
}

#endif
