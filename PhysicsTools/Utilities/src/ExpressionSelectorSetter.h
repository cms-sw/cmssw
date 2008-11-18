#ifndef Utilities_ExpressionSelectorSetter_h
#define Utilities_ExpressionSelectorSetter_h
/* \class reco::parser::ExpressionSelectorSetter
 *
 * Creates an implicit Binary selector setter by comparing an expression to 0
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "PhysicsTools/Utilities/src/SelectorStack.h"
#include "PhysicsTools/Utilities/src/ExpressionStack.h"
#include "PhysicsTools/Utilities/src/BinarySelector.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/Utilities/src/ExpressionNumber.h"
#include "PhysicsTools/Utilities/src/Comparison.h"
#include <boost/shared_ptr.hpp>
#include <functional>

namespace reco {
  namespace parser {    
    class ExpressionSelectorSetter {
    public:
      ExpressionSelectorSetter( SelectorStack& selStack, ExpressionStack& expStack ) : 
	selStack_( selStack ), expStack_( expStack ) { }
      
      void operator()(const char * , const char *) const {
	if(expStack_.empty())
	  throw edm::Exception(edm::errors::Configuration)
	    << "Parse error: empty expression stack." << "\"\n";
	boost::shared_ptr<ExpressionBase> rhs = expStack_.back(); expStack_.pop_back();
	boost::shared_ptr<ExpressionBase> lhs( new ExpressionNumber(0.0));
	boost::shared_ptr<ComparisonBase> comp( new Comparison<std::not_equal_to<double> >() );
#ifdef BOOST_SPIRIT_DEBUG 
	BOOST_SPIRIT_DEBUG_OUT << "pushing expression selector" << std::endl;
#endif
	selStack_.push_back( SelectorPtr( new BinarySelector( lhs, comp, rhs ) ) );
      }
    private:
      SelectorStack & selStack_;
      ExpressionStack & expStack_;
    };
  }
}

#endif
