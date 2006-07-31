#include "PhysicsTools/UtilAlgos/interface/cutParser.h"
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>
#include "FWCore/Utilities/interface/EDMException.h"
#include <Reflex/Member.h>
#include <Reflex/Object.h>
#include <vector>
using namespace ROOT::Reflex;
using namespace boost::spirit;
using namespace std;

namespace reco {
  namespace parser {
    typedef scanner<const char*, scanner_policies_t> ScannerUsed;
    typedef rule<ScannerUsed> Rule_t;
    
    struct ExpressionBase {
      virtual ~ExpressionBase() {};
      virtual double value( const Object & ) const = 0;
    };
    
    struct ExpressionNumber : public ExpressionBase {
      virtual double value( const Object& ) const { return m_value; }
      double m_value;
      ExpressionNumber( double iValue ) : m_value(iValue) { }
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
      ExpressionStack& stack_;
    };
    
    struct ExpressionVarSetter {
      ExpressionVarSetter( ExpressionStack & stack, const MethodMap & methods ) : 
	stack_( stack ), methods_( methods ){ }
      
      void operator()( const char * iVarStart, const char* iVarEnd ) const {
	string methodName( iVarStart, iVarEnd );
	string::size_type endOfExpr = methodName.find_last_of(' ');
	if( endOfExpr != string::npos )
	  methodName.erase( endOfExpr, methodName.size() );
	MethodMap::const_iterator itMethod = methods_.find( methodName );
	if( itMethod == methods_.end() ) {
	  throw edm::Exception( edm::errors::Configuration, 
				string( "unknown method name \""+ methodName + "\"" ) );
	}
	stack_.push_back( boost::shared_ptr<ExpressionBase>( new ExpressionVar( itMethod->second ) ) );
    }
    private:
      ExpressionStack& stack_;
      const MethodMap & methods_;
    };
    
    struct ComparisonBase {
      virtual bool compare( double, double ) const = 0;
    };
    
    template<class CompT>
    struct Comparison : public ComparisonBase {
      virtual bool compare( double iLHS, double iRHS ) const {
	CompT comp;
	return comp( iLHS, iRHS );
      }
    };
    
    typedef vector<boost::shared_ptr<ComparisonBase> > ComparisonStack;
    template<class CompT>
    class ComparisonSetter {
    public:
      ComparisonSetter( ComparisonStack& iStack) : stack_(iStack){}
      
      void operator()( const char ) const {
	stack_.push_back( boost::shared_ptr<ComparisonBase>( new Comparison<CompT>() ) );
      }
    private:
      ComparisonStack& stack_;
    };
    
    typedef vector<selector_ptr> SelectorStack;
  
    struct BinarySelector : public ReflexSelector {
      BinarySelector( boost::shared_ptr<ExpressionBase> iLHS,
		      boost::shared_ptr<ComparisonBase> iComp,
		      boost::shared_ptr<ExpressionBase> iRHS ) :
	lhs_( iLHS), comp_(iComp), rhs_( iRHS ) {}
      virtual bool operator()(const Object& o) const {
	return comp_->compare( lhs_->value(o), rhs_->value(o) );
      }
      boost::shared_ptr<ExpressionBase> lhs_;
      boost::shared_ptr<ComparisonBase> comp_;
      boost::shared_ptr<ExpressionBase> rhs_;
    };
    
    class BinarySelectorSetter {
    public:
      BinarySelectorSetter( SelectorStack& iSelStack,
			    ComparisonStack& iCompStack, ExpressionStack& iExpStack) : 
	sStack_(iSelStack), cStack_(iCompStack), eStack_(iExpStack){}
      
      void operator()( const char* iVarStart, const char* iVarEnd) const {
	boost::shared_ptr<ExpressionBase> rhs = eStack_.back(); eStack_.pop_back();
	boost::shared_ptr<ExpressionBase> lhs = eStack_.back(); eStack_.pop_back();
	boost::shared_ptr<ComparisonBase> comp = cStack_.back(); cStack_.pop_back();
	sStack_.push_back( selector_ptr( new BinarySelector( lhs, comp, rhs ) ) );
      }
    private:
      SelectorStack& sStack_;
      ComparisonStack& cStack_;
      ExpressionStack& eStack_;
    };
    
    struct TrinarySelector : public ReflexSelector {
      TrinarySelector( boost::shared_ptr<ExpressionBase> iLHS,
		       boost::shared_ptr<ComparisonBase> iComp1,
		       boost::shared_ptr<ExpressionBase> iMid,
		       boost::shared_ptr<ComparisonBase> iComp2,
		       boost::shared_ptr<ExpressionBase> iRHS) :
	lhs_( iLHS), comp1_(iComp1), mid_(iMid), comp2_(iComp2),rhs_( iRHS ) {}
      virtual bool operator()(const Object& o) const {
	return comp1_->compare( lhs_->value(o), mid_->value(o) ) &&
	  comp2_->compare( mid_->value(o), rhs_->value(o) );
      }
      boost::shared_ptr<ExpressionBase> lhs_;
      boost::shared_ptr<ComparisonBase> comp1_;
      boost::shared_ptr<ExpressionBase> mid_;
      boost::shared_ptr<ComparisonBase> comp2_;
      boost::shared_ptr<ExpressionBase> rhs_;
    };
    
    class TrinarySelectorSetter {
    public:
      TrinarySelectorSetter( SelectorStack& iSelStack,
			     ComparisonStack& iCompStack, ExpressionStack& iExpStack ) : 
	sStack_(iSelStack), cStack_(iCompStack), eStack_(iExpStack){}
      
      void operator()( const char* iVarStart, const char* iVarEnd) const {
	boost::shared_ptr<ExpressionBase> rhs = eStack_.back(); eStack_.pop_back();
	boost::shared_ptr<ExpressionBase> mid = eStack_.back();eStack_.pop_back();
	boost::shared_ptr<ExpressionBase> lhs = eStack_.back();eStack_.pop_back();
	boost::shared_ptr<ComparisonBase> comp2 = cStack_.back();cStack_.pop_back();
	boost::shared_ptr<ComparisonBase> comp1 = cStack_.back();cStack_.pop_back();
	sStack_.push_back( selector_ptr( new TrinarySelector( lhs, comp1, mid, comp2, rhs ) ) );
      }
    private:
      SelectorStack& sStack_;
      ComparisonStack& cStack_;
      ExpressionStack& eStack_;
    };
    
    enum Combiner { kAnd, kOr };
    
    typedef vector< Combiner > CombinerStack;
    
    struct AndCombiner : public ReflexSelector {
      AndCombiner( selector_ptr iLHS,
		   selector_ptr iRHS ) :
	lhs_( iLHS), rhs_( iRHS ) {}
      virtual bool operator()(const Object& o) const {
	return (*lhs_)(o) && (*rhs_)(o);
      }
      selector_ptr lhs_;
      selector_ptr rhs_;
    };
    
    struct OrCombiner : public ReflexSelector {
      OrCombiner( selector_ptr iLHS,
		  selector_ptr iRHS ) :
	lhs_( iLHS), rhs_( iRHS ) {}
      virtual bool operator()(const Object& o) const {
	return (*lhs_)(o) || (*rhs_)(o);
      }
      selector_ptr lhs_;
      selector_ptr rhs_;
    };
    
    struct CombinerSetter {
      CombinerSetter( Combiner iCombiner, CombinerStack& iStack ):
	comb_( iCombiner), stack_(iStack) {}
      
      void operator()(const char ) const {
	stack_.push_back( comb_ );
      }
      Combiner comb_;
      CombinerStack& stack_;
    };
    
    struct CutSetter {
      CutSetter( selector_ptr& iCut,
		 SelectorStack& iSelStack,
		 CombinerStack& iCombStack) :
	cut_( iCut ), sStack_(iSelStack), cStack_(iCombStack) {}
      
      void operator()(const char*, const char* ) const {
	if( 0 == cut_.get() ) {
	  cut_ = sStack_.back();
	  sStack_.pop_back();
	} else {
	  if( cStack_.back() == kAnd ) {
	    selector_ptr lhs = cut_;
	    cut_ = selector_ptr(new AndCombiner(lhs, sStack_.back() ) );
	  } else {
	    selector_ptr lhs = cut_;
	    cut_ = selector_ptr(new OrCombiner(lhs, sStack_.back() ) );
	  }
	  cStack_.pop_back();
	  sStack_.pop_back();
	}
      }
      selector_ptr& cut_;
      SelectorStack& sStack_;
      CombinerStack& cStack_;
    };
    
    bool cutParser( const string& value, const MethodMap& methods, selector_ptr& sel ) {
      
      using namespace boost::spirit;
      ExpressionStack expressionStack;
      ComparisonStack cStack;
      SelectorStack sStack;
      CombinerStack combStack;
      
      Rule_t number = real_p                       [ExpressionNumberSetter(expressionStack)];
      Rule_t var = (alpha_p >> *alnum_p)           [ExpressionVarSetter(expressionStack, methods)];
      
      Rule_t comparison_op = ch_p('<')             [ComparisonSetter<less<double> >(cStack)]           | 
	(ch_p('<') >> ch_p('=')[ComparisonSetter<less_equal<double> >(cStack)])    | 
	ch_p('=')              [ComparisonSetter<equal_to<double> >(cStack)]       | 
	(ch_p('>') >> ch_p('=')[ComparisonSetter<greater_equal<double> >(cStack)]) | 
	ch_p('>')              [ComparisonSetter<greater<double> >(cStack)]        | 
	(ch_p('!') >> ch_p('=')[ComparisonSetter<not_equal_to<double> >(cStack)]);
      
      Rule_t expression = var | number;
      Rule_t binary_comp = expression >> comparison_op >> expression;
      Rule_t trinary_comp = expression >> comparison_op >> var >> comparison_op >> expression;
      Rule_t logical_combiner = ch_p('&')[CombinerSetter(kAnd, combStack)] | ch_p('|')[CombinerSetter(kOr, combStack)];
      Rule_t cut = (trinary_comp [TrinarySelectorSetter(sStack, cStack, expressionStack)] | 
		    binary_comp  [BinarySelectorSetter(sStack, cStack, expressionStack)]) [CutSetter(sel, sStack, combStack)] % logical_combiner;
      return parse<>( value.c_str(),
		      (
		       cut
		       )
		      ,
		      space_p).full;
    }
  }
}
