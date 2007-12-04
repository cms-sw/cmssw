#ifndef Utilities_BinarySelectorSetter_h
#define Utilities_BinarySelectorSetter_h
/* \class reco::parser::BinarySelectorSetter
 *
 * Binary selector setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/SelectorStack.h"
#include "PhysicsTools/Utilities/src/ComparisonStack.h"
#include "PhysicsTools/Utilities/src/ExpressionStack.h"
#include "PhysicsTools/Utilities/src/BinarySelector.h"
#include <boost/shared_ptr.hpp>

namespace reco {
  namespace parser {    
    class BinarySelectorSetter {
    public:
      BinarySelectorSetter( SelectorStack& selStack,
			    ComparisonStack& cmpStack, ExpressionStack& expStack ) : 
	selStack_( selStack ), cmpStack_( cmpStack ), expStack_( expStack ) { }
      
      void operator()( const char * , const char * ) const {
	boost::shared_ptr<ExpressionBase> rhs = expStack_.back(); expStack_.pop_back();
	boost::shared_ptr<ExpressionBase> lhs = expStack_.back(); expStack_.pop_back();
	boost::shared_ptr<ComparisonBase> comp = cmpStack_.back(); cmpStack_.pop_back();
#ifdef BOOST_SPIRIT_DEBUG 
	BOOST_SPIRIT_DEBUG_OUT << "pushing binary selector" << std::endl;
#endif
	selStack_.push_back( SelectorPtr( new BinarySelector( lhs, comp, rhs ) ) );
      }
    private:
      SelectorStack & selStack_;
      ComparisonStack & cmpStack_;
      ExpressionStack & expStack_;
    };
  }
}

#endif
