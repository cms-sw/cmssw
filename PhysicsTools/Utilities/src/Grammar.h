#ifndef Utilities_Grammar_h
#define Utilities_Grammar_h
/* \class reco::parser::Grammar
 *
 * parser grammar
 *
 * \author original version: Chris Jones, Cornell, 
 *         extended by Luca Lista, INFN
 *
 * \version $Revision: 1.9 $
 *
 */
#include "boost/spirit/core.hpp"
#include "boost/spirit/utility/grammar_def.hpp"
#include <functional>
#include "PhysicsTools/Utilities/src/ExpressionNumberSetter.h"
#include "PhysicsTools/Utilities/src/ExpressionVarSetter.h"
#include "PhysicsTools/Utilities/src/ExpressionFunctionSetter.h"
#include "PhysicsTools/Utilities/src/ComparisonSetter.h"
#include "PhysicsTools/Utilities/src/BinarySelectorSetter.h"
#include "PhysicsTools/Utilities/src/TrinarySelectorSetter.h"
#include "PhysicsTools/Utilities/src/IntSetter.h"
#include "PhysicsTools/Utilities/src/CombinerStack.h"
#include "PhysicsTools/Utilities/src/MethodStack.h"
#include "PhysicsTools/Utilities/src/TypeStack.h"
#include "PhysicsTools/Utilities/src/IntStack.h"
#include "PhysicsTools/Utilities/src/CombinerSetter.h"
#include "PhysicsTools/Utilities/src/FunctionSetter.h"
#include "PhysicsTools/Utilities/src/CutSetter.h"
#include "PhysicsTools/Utilities/src/ExpressionSetter.h"
#include "PhysicsTools/Utilities/src/ExpressionBinaryOperatorSetter.h"
#include "PhysicsTools/Utilities/src/ExpressionUnaryOperatorSetter.h"
#include "PhysicsTools/Utilities/src/MethodSetter.h"
// #include "PhysicsTools/Utilities/src/Abort.h"

namespace reco {
  namespace parser {    
    struct Grammar : public boost::spirit::grammar<Grammar> {
      SelectorPtr dummySel_;
      ExpressionPtr dummyExpr_;
      SelectorPtr * sel_; 
      ExpressionPtr * expr_;
      mutable ExpressionStack exprStack;
      mutable ComparisonStack cmpStack;
      mutable SelectorStack selStack;
      mutable CombinerStack cmbStack;
      mutable FunctionStack funStack;
      mutable MethodStack methStack;
      mutable TypeStack typeStack;
      mutable IntStack intStack;
      template<typename T>
      Grammar(SelectorPtr & sel, const T *) : 
	sel_(& sel), expr_(& dummyExpr_) { 
	typeStack.push_back(ROOT::Reflex::Type::ByTypeInfo(typeid(T)));
      }
      template<typename T>
      Grammar(ExpressionPtr & expr, const T*) : 
	sel_(& dummySel_), expr_(& expr) { 
	typeStack.push_back(ROOT::Reflex::Type::ByTypeInfo(typeid(T)));
      }
      template <typename ScannerT>
      struct definition : 
	public boost::spirit::grammar_def<boost::spirit::rule<ScannerT>, 
					  boost::spirit::same, 
					  boost::spirit::same>{  
	typedef boost::spirit::rule<ScannerT> rule;
	rule number, var, method, term, power, factor, function1, function2, expression, 
	  comparison_op, binary_comp, trinary_comp,
	  logical_combiner, logical_expression, logical_factor, logical_term,
	  cut, fun;
	definition(const Grammar & self) {
	  using namespace boost::spirit;
	  using namespace std;

	  ExpressionNumberSetter number_s(self.exprStack);
	  IntSetter int_s(self.intStack);
	  ExpressionVarSetter var_s(self.exprStack, self.methStack, self.typeStack);
	  MethodSetter method_s(self.methStack, self.typeStack, self.intStack);
	  ComparisonSetter<less_equal<double> > less_equal_s(self.cmpStack);
	  ComparisonSetter<less<double> > less_s(self.cmpStack);
	  ComparisonSetter<equal_to<double> > equal_to_s(self.cmpStack);
	  ComparisonSetter<greater_equal<double> > greater_equal_s(self.cmpStack);
	  ComparisonSetter<greater<double> > greater_s(self.cmpStack);
	  ComparisonSetter<not_equal_to<double> > not_equal_to_s(self.cmpStack);
	  CombinerSetter and_s(kAnd, self.cmbStack), or_s(kOr, self.cmbStack), not_s(kNot, self.cmbStack);
	  FunctionSetter 
	    abs_s(kAbs, self.funStack), acos_s(kAcos, self.funStack), asin_s(kAsin, self.funStack),
	    atan_s(kAtan, self.funStack), atan2_s(kAtan, self.funStack), 
	    chi2prob_s(kChi2Prob, self.funStack), 
            cos_s(kCos, self.funStack), 
	    cosh_s(kCosh, self.funStack), exp_s(kExp, self.funStack), log_s(kLog, self.funStack), 
	    log10_s(kLog10, self.funStack), max_s(kMax, self.funStack), min_s(kMin, self.funStack),
	    pow_s(kPow, self.funStack), sin_s(kSin, self.funStack), 
	    sinh_s(kSinh, self.funStack), sqrt_s(kSqrt, self.funStack), tan_s(kTan, self.funStack), 
	    tanh_s(kTanh, self.funStack);
	  TrinarySelectorSetter trinary_s(self.selStack, self.cmpStack, self.exprStack);
	  BinarySelectorSetter binary_s(self.selStack, self.cmpStack, self.exprStack);
	  CutSetter cut_s(* self.sel_, self.selStack, self.cmbStack);
	  ExpressionSetter expr_s(* self.expr_, self.exprStack);
	  ExpressionBinaryOperatorSetter<plus<double> > plus_s(self.exprStack);
	  ExpressionBinaryOperatorSetter<minus<double> > minus_s(self.exprStack);
	  ExpressionBinaryOperatorSetter<multiplies<double> > multiplies_s(self.exprStack);
	  ExpressionBinaryOperatorSetter<divides<double> > divides_s(self.exprStack);
	  ExpressionBinaryOperatorSetter<power_of<double> > power_of_s(self.exprStack);
	  ExpressionUnaryOperatorSetter<negate<double> > negate_s(self.exprStack);
	  ExpressionFunctionSetter fun_s(self.exprStack, self.funStack);
	  //	  Abort abort_s;
  
	  number = 
	    real_p [ number_s ];
	  var = 
	    (alpha_p >> * alnum_p >> 
	      ch_p('(') >> int_p [ int_s ] >> * (ch_p(',') >> int_p [ int_s ]) >> ch_p(')')) [ method_s ] |
	    (alpha_p >> * alnum_p) [ method_s ];
	  method = 
	    (var >> * ((ch_p('.') >> var))) [ var_s ];
	  function1 = 
	    chseq_p("abs") [ abs_s ] | chseq_p("acos") [ acos_s ] | chseq_p("asin") [ asin_s ] |
	    chseq_p("atan") [ atan_s ] | chseq_p("cos") [ cos_s ] | chseq_p("cosh") [ cosh_s ] |
	    chseq_p("exp") [ exp_s ] | chseq_p("log") [ log_s ] | chseq_p("log10") [ log10_s ] |
	    chseq_p("sin") [ sin_s ] | chseq_p("sinh") [ sinh_s ] | chseq_p("sqrt") [ sqrt_s ] |
	    chseq_p("tan") [ tan_s ] | chseq_p("tanh") [ tanh_s ];
	  function2 = 
	    chseq_p("atan2") [ atan2_s ] | chseq_p("chi2prob") [ chi2prob_s ] | chseq_p("pow") [ pow_s ] |
            chseq_p("min") [ min_s ] | chseq_p("max") [ max_s ];
	  expression = 
	    term >> * (('+' >> term) [ plus_s ] |
			('-' >> term) [ minus_s ]);
	  term = 
	    power >> * (('*' >> power) [ multiplies_s ] |
			 ('/' >> power) [ divides_s ]);
	  power = 
	    factor >> * (('^' >> factor) [ power_of_s ]);
	  factor = 
	    number | 
	    (function1 >> ch_p('(') >> expression >> ch_p(')')) [ fun_s ] |
	    (function2 >> ch_p('(') >> expression >> ch_p(',') >> expression >> ch_p(')')) [ fun_s ] |
	    method | 
	    '(' >> expression >> ')' |   
	    ('-' >> factor) [ negate_s ] |
	    ('+' >> factor);
	  comparison_op = 
	    (ch_p('<') >> ch_p('=') [ less_equal_s    ]) | 
	    (ch_p('<')              [ less_s          ]) | 
	    (ch_p('=')              [ equal_to_s      ]) | 
	    (ch_p('>') >> ch_p('=') [ greater_equal_s ]) | 
	    (ch_p('>')              [ greater_s       ]) | 
	    (ch_p('!') >> ch_p('=') [ not_equal_to_s  ]);
	  binary_comp = 
	    expression >> comparison_op >> expression;
	  trinary_comp = 
	    expression >> comparison_op >> expression >> comparison_op >> expression;
	  logical_expression = 
	    logical_term >> * ((ch_p('|') [ or_s ] >> logical_term));
	  logical_term = 
	    logical_factor >> * ((ch_p('&') [ and_s ] >> logical_factor));
	  logical_factor =
	    (trinary_comp [ trinary_s ] | 
	      binary_comp [ binary_s ] |
	      (ch_p('!') [ not_s ] >> logical_factor)) [ cut_s ] |
	    '(' >> logical_expression >> ')' |
	    logical_expression;
	  cut = logical_expression;
	  fun = expression [ expr_s ];
	  start_parsers(cut, fun);
	}
      };
    };
  }
}


#endif
