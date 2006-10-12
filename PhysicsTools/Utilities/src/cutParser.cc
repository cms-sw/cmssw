#include "PhysicsTools/Utilities/interface/cutParser.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <Reflex/Member.h>
#include <Reflex/Object.h>
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>
#include <vector>
using namespace ROOT::Reflex;
using namespace std;

namespace reco {
  namespace parser {
    struct ExpressionBase {
      virtual ~ExpressionBase() {};
      virtual double value( const Object & ) const = 0;
    };
    
    struct ExpressionNumber : public ExpressionBase {
      virtual double value( const Object& ) const { return value_; }
      ExpressionNumber( double value ) : value_( value ) { }
    private:
      double value_;
    };
    
    struct ExpressionVar : public ExpressionBase {
      ExpressionVar( MethodMap::method_t m ): method_( m ) { }
      virtual double value( const Object & o ) const {
	using namespace method;
	Object ro = method_.first.Invoke( o );
	void * addr = ro.Address();
	double ret = 0;
	switch( method_.second ) {
	case( doubleType ) : ret = * static_cast<double         *>( addr ); break;
	case( floatType  ) : ret = * static_cast<float          *>( addr ); break;
	case( intType    ) : ret = * static_cast<int            *>( addr ); break;
	case( uIntType   ) : ret = * static_cast<unsigned int   *>( addr ); break;
	case( shortType  ) : ret = * static_cast<short          *>( addr ); break;
	case( uShortType ) : ret = * static_cast<unsigned short *>( addr ); break;
	case( longType   ) : ret = * static_cast<long           *>( addr ); break;
	case( uLongType  ) : ret = * static_cast<unsigned long  *>( addr ); break;
	case( charType   ) : ret = * static_cast<char           *>( addr ); break;
	case( uCharType  ) : ret = * static_cast<unsigned char  *>( addr ); break;
	case( boolType   ) : ret = * static_cast<bool           *>( addr ); break;
	};
	return ret;
      }
      MethodMap::method_t method_;
    };
    
    typedef vector<boost::shared_ptr<ExpressionBase> > ExpressionStack;
    
    struct ExpressionNumberSetter {
      ExpressionNumberSetter( ExpressionStack & stack ) : stack_( stack ) { }
      void operator()( double n ) const {
	stack_.push_back( boost::shared_ptr<ExpressionBase>( new ExpressionNumber( n ) ) );
      }
    private:
      ExpressionStack & stack_;
    };
    
    struct ExpressionVarSetter {
      ExpressionVarSetter( ExpressionStack & stack, const MethodMap & methods ) : 
	stack_( stack ), methods_( methods ){ }
      
      void operator()( const char * begin, const char* end ) const {
	string methodName( begin, end );
	string::size_type endOfExpr = methodName.find_last_of(' ');
	if( endOfExpr != string::npos )
	  methodName.erase( endOfExpr, methodName.size() );
	MethodMap::const_iterator m = methods_.find( methodName );
	if( m == methods_.end() ) {
	  throw edm::Exception( edm::errors::Configuration, 
				string( "unknown method name \""+ methodName + "\"" ) );
	}
	stack_.push_back( boost::shared_ptr<ExpressionBase>( new ExpressionVar( m->second ) ) );
    }
    private:
      ExpressionStack & stack_;
      const MethodMap & methods_;
    };
    
    struct ComparisonBase {
      virtual bool compare( double, double ) const = 0;
    };
    
    template<class CompT>
    struct Comparison : public ComparisonBase {
      virtual bool compare( double lhs, double rhs ) const { return comp( lhs, rhs ); }
    private:
      CompT comp;
    };
    
    typedef vector<boost::shared_ptr<ComparisonBase> > ComparisonStack;
    template<class CompT>
    struct ComparisonSetter {
      ComparisonSetter( ComparisonStack& stack ) : stack_( stack ) { }
      void operator()( const char ) const {
	stack_.push_back( boost::shared_ptr<ComparisonBase>( new Comparison<CompT>() ) );
      }
    private:
      ComparisonStack & stack_;
    };
    
    typedef vector<selector_ptr> SelectorStack;
  
    struct BinarySelector : public ReflexSelector {
      BinarySelector( boost::shared_ptr<ExpressionBase> lhs,
		      boost::shared_ptr<ComparisonBase> cmp,
		      boost::shared_ptr<ExpressionBase> rhs ) :
	lhs_( lhs ), cmp_( cmp ), rhs_( rhs ) {}
      virtual bool operator()( const Object& o ) const {
	return cmp_->compare( lhs_->value(o), rhs_->value(o) );
      }
      boost::shared_ptr<ExpressionBase> lhs_;
      boost::shared_ptr<ComparisonBase> cmp_;
      boost::shared_ptr<ExpressionBase> rhs_;
    };
    
    class BinarySelectorSetter {
    public:
      BinarySelectorSetter( SelectorStack& selStack,
			    ComparisonStack& cmpStack, ExpressionStack& expStack ) : 
	selStack_( selStack ), cmpStack_( cmpStack ), expStack_( expStack ) { }
      
      void operator()( const char* iVarStart, const char* iVarEnd) const {
	boost::shared_ptr<ExpressionBase> rhs = expStack_.back(); expStack_.pop_back();
	boost::shared_ptr<ExpressionBase> lhs = expStack_.back(); expStack_.pop_back();
	boost::shared_ptr<ComparisonBase> comp = cmpStack_.back(); cmpStack_.pop_back();
	selStack_.push_back( selector_ptr( new BinarySelector( lhs, comp, rhs ) ) );
      }
    private:
      SelectorStack & selStack_;
      ComparisonStack & cmpStack_;
      ExpressionStack & expStack_;
    };
    
    struct TrinarySelector : public ReflexSelector {
      TrinarySelector( boost::shared_ptr<ExpressionBase> lhs,
		       boost::shared_ptr<ComparisonBase> cmp1,
		       boost::shared_ptr<ExpressionBase> mid,
		       boost::shared_ptr<ComparisonBase> cmp2,
		       boost::shared_ptr<ExpressionBase> rhs ) :
	lhs_( lhs ), cmp1_( cmp1 ), mid_( mid ), cmp2_( cmp2 ),rhs_( rhs ) {}
      virtual bool operator()( const Object& o ) const {
	return 
	  cmp1_->compare( lhs_->value( o ), mid_->value( o ) ) &&
	  cmp2_->compare( mid_->value( o ), rhs_->value( o ) );
      }
      boost::shared_ptr<ExpressionBase> lhs_;
      boost::shared_ptr<ComparisonBase> cmp1_;
      boost::shared_ptr<ExpressionBase> mid_;
      boost::shared_ptr<ComparisonBase> cmp2_;
      boost::shared_ptr<ExpressionBase> rhs_;
    };
    
    class TrinarySelectorSetter {
    public:
      TrinarySelectorSetter( SelectorStack& selStack,
			     ComparisonStack& cmpStack, ExpressionStack& expStack ) : 
	selStack_( selStack ), cmpStack_( cmpStack ), expStack_( expStack ) { }
      
      void operator()( const char* begin, const char* end ) const {
	boost::shared_ptr<ExpressionBase> rhs = expStack_.back(); expStack_.pop_back();
	boost::shared_ptr<ExpressionBase> mid = expStack_.back(); expStack_.pop_back();
	boost::shared_ptr<ExpressionBase> lhs = expStack_.back(); expStack_.pop_back();
	boost::shared_ptr<ComparisonBase> comp2 = cmpStack_.back(); cmpStack_.pop_back();
	boost::shared_ptr<ComparisonBase> comp1 = cmpStack_.back(); cmpStack_.pop_back();
	selStack_.push_back( selector_ptr( new TrinarySelector( lhs, comp1, mid, comp2, rhs ) ) );
      }
    private:
      SelectorStack& selStack_;
      ComparisonStack& cmpStack_;
      ExpressionStack& expStack_;
    };
    
    enum Combiner { kAnd, kOr };
    
    typedef vector<Combiner> CombinerStack;
    
    struct AndCombiner : public ReflexSelector {
      AndCombiner( selector_ptr lhs, selector_ptr rhs ) :
	lhs_( lhs ), rhs_( rhs ) { }
      virtual bool operator()( const Object& o ) const {
	return (*lhs_)( o ) && (*rhs_)( o );
      }
    private:
      selector_ptr lhs_, rhs_;
    };
    
    struct OrCombiner : public ReflexSelector {
      OrCombiner( selector_ptr lhs, selector_ptr rhs ) :
	lhs_( lhs ), rhs_( rhs ) {}
      virtual bool operator()( const Object& o ) const {
	return (*lhs_)( o ) || (*rhs_)( o );
      }
    private:
      selector_ptr lhs_, rhs_;
    };
    
    struct CombinerSetter {
      CombinerSetter( Combiner comb, CombinerStack& stack ):
	comb_( comb ), stack_( stack ) {}
      
      void operator()(const char ) const { stack_.push_back( comb_ ); }
    private:
      Combiner comb_;
      CombinerStack & stack_;
    };
    
    struct CutSetter {
      CutSetter( selector_ptr& cut, SelectorStack& selStack, CombinerStack& cmbStack ) :
	cut_( cut ), selStack_( selStack ), cmbStack_( cmbStack ) { }
      
      void operator()( const char*, const char* ) const {
	if( 0 == cut_.get() ) {
	  cut_ = selStack_.back();
	  selStack_.pop_back();
	} else {
	  switch ( cmbStack_.back() ) {
	  case ( kAnd ) : {
	    selector_ptr lhs = cut_;
	    cut_ = selector_ptr( new AndCombiner( lhs, selStack_.back() ) );
	    break;
	  }
	  case ( kOr ) : {
	    selector_ptr lhs = cut_;
	    cut_ = selector_ptr( new OrCombiner( lhs, selStack_.back() ) );
	    break;
	  }
	  };
	  cmbStack_.pop_back();
	  selStack_.pop_back();
	}
      }
      selector_ptr & cut_;
      SelectorStack & selStack_;
      CombinerStack & cmbStack_;
    };
    
    bool cutParser( const string& value, const MethodMap& methods, selector_ptr & sel ) {
      using namespace boost::spirit;
      typedef rule<scanner<const char*, scanner_policies_t> > Rule_t;

      ExpressionStack expressionStack;
      ComparisonStack cmpStack;
      SelectorStack selStack;
      CombinerStack cmbStack;
      
      Rule_t number = real_p [ ExpressionNumberSetter( expressionStack ) ];
      Rule_t var = ( alpha_p >> * alnum_p ) [ ExpressionVarSetter( expressionStack, methods ) ];
      
      Rule_t comparison_op = 
	ch_p('<')                [ ComparisonSetter<less<double> >         ( cmpStack ) ]   | 
	( ch_p('<') >> ch_p('=') [ ComparisonSetter<less_equal<double> >   ( cmpStack ) ] ) | 
	ch_p('=')                [ ComparisonSetter<equal_to<double> >     ( cmpStack ) ]   | 
	( ch_p('>') >> ch_p('=') [ ComparisonSetter<greater_equal<double> >( cmpStack ) ] ) | 
	ch_p('>')                [ ComparisonSetter<greater<double> >      ( cmpStack ) ]   | 
	( ch_p('!') >> ch_p('=') [ ComparisonSetter<not_equal_to<double> > ( cmpStack ) ] );
      Rule_t expression = 
	var | number;
      Rule_t binary_comp = 
	expression >> comparison_op >> expression;
      Rule_t trinary_comp = 
	expression >> comparison_op >> var >> comparison_op >> expression;
      Rule_t logical_combiner = 
	ch_p('&') [ CombinerSetter( kAnd, cmbStack )] | 
	ch_p('|') [ CombinerSetter(kOr, cmbStack) ];
      Rule_t cut = 
	( trinary_comp [ TrinarySelectorSetter( selStack, cmpStack, expressionStack ) ] | 
	  binary_comp  [ BinarySelectorSetter ( selStack, cmpStack, expressionStack ) ] ) [ CutSetter( sel, selStack, cmbStack ) ] 
        % logical_combiner;

      return parse<>( value.c_str(), cut, space_p ).full;
    }
  }
}
