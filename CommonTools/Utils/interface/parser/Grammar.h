#ifndef CommonTools_Utils_Grammar_h
#define CommonTools_Utils_Grammar_h
/* \class reco::parser::Grammar
 *
 * parser grammar
 *
 * \author original version: Chris Jones, Cornell, 
 *         extended by Luca Lista, INFN
 *
 * \version $Revision: 1.13 $
 *
 */
#include "boost/spirit/include/classic_core.hpp"
#include "boost/spirit/include/classic_grammar_def.hpp"
#include "boost/spirit/include/classic_chset.hpp"
#include <functional>
#include "CommonTools/Utils/interface/parser/ExpressionNumberSetter.h"
#include "CommonTools/Utils/interface/parser/ExpressionVarSetter.h"
#include "CommonTools/Utils/interface/parser/ExpressionFunctionSetter.h"
#include "CommonTools/Utils/interface/parser/ExpressionConditionSetter.h"
#include "CommonTools/Utils/interface/parser/ComparisonSetter.h"
#include "CommonTools/Utils/interface/parser/BinarySelectorSetter.h"
#include "CommonTools/Utils/interface/parser/TrinarySelectorSetter.h"
#include "CommonTools/Utils/interface/parser/IntSetter.h"
#include "CommonTools/Utils/interface/parser/MethodStack.h"
#include "CommonTools/Utils/interface/parser/MethodArgumentStack.h"
#include "CommonTools/Utils/interface/parser/TypeStack.h"
#include "CommonTools/Utils/interface/parser/IntStack.h"
#include "CommonTools/Utils/interface/parser/FunctionSetter.h"
#include "CommonTools/Utils/interface/parser/CutSetter.h"
#include "CommonTools/Utils/interface/parser/BinaryCutSetter.h"
#include "CommonTools/Utils/interface/parser/UnaryCutSetter.h"
#include "CommonTools/Utils/interface/parser/ExpressionSetter.h"
#include "CommonTools/Utils/interface/parser/ExpressionBinaryOperatorSetter.h"
#include "CommonTools/Utils/interface/parser/ExpressionUnaryOperatorSetter.h"
#include "CommonTools/Utils/interface/parser/ExpressionSelectorSetter.h"
#include "CommonTools/Utils/interface/parser/MethodSetter.h"
#include "CommonTools/Utils/interface/parser/MethodArgumentSetter.h"
#include "CommonTools/Utils/interface/parser/Exception.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"

namespace reco {
  namespace parser {
    struct Grammar : public boost::spirit::classic::grammar<Grammar> {
      SelectorPtr dummySel_;
      ExpressionPtr dummyExpr_;
      SelectorPtr* sel_;
      ExpressionPtr* expr_;
      bool lazy_;
      mutable ExpressionStack exprStack;
      mutable ComparisonStack cmpStack;
      mutable SelectorStack selStack;
      mutable FunctionStack funStack, finalFunStack;
      mutable MethodStack methStack;
      mutable LazyMethodStack lazyMethStack;
      mutable MethodArgumentStack methArgStack;
      mutable TypeStack typeStack;
      mutable IntStack intStack;

      Grammar(SelectorPtr& sel, const edm::TypeWithDict& iType, bool lazy = false)
          : sel_(&sel), expr_(&dummyExpr_), lazy_(lazy) {
        typeStack.push_back(iType);
      }

      Grammar(ExpressionPtr& expr, const edm::TypeWithDict& iType, bool lazy = false)
          : sel_(&dummySel_), expr_(&expr), lazy_(lazy) {
        typeStack.push_back(iType);
      }

      template <typename ScannerT>
      struct definition : public boost::spirit::classic::grammar_def<boost::spirit::classic::rule<ScannerT>,
                                                                     boost::spirit::classic::same,
                                                                     boost::spirit::classic::same> {
        typedef boost::spirit::classic::rule<ScannerT> rule;
        rule number, var, arrayAccess, metharg, method, term, power, factor, function1, function2, function4,
            expression, comparison_op, binary_comp, trinary_comp, logical_combiner, logical_expression,
            nocond_expression, cond_expression, logical_factor, logical_term, or_op, and_op, cut, fun;
        definition(const Grammar& self) {
          using namespace boost::spirit::classic;
          using namespace std;

          ExpressionNumberSetter number_s(self.exprStack);
          IntSetter int_s(self.intStack);
          ExpressionVarSetter var_s(self.exprStack, self.methStack, self.lazyMethStack, self.typeStack);
          ExpressionConditionSetter cond_s(self.exprStack, self.selStack);
          MethodArgumentSetter methodArg_s(self.methArgStack);
          MethodSetter method_s(self.methStack, self.lazyMethStack, self.typeStack, self.methArgStack, self.lazy_);
          ComparisonSetter<less_equal<double> > less_equal_s(self.cmpStack);
          ComparisonSetter<less<double> > less_s(self.cmpStack);
          ComparisonSetter<equal_to<double> > equal_to_s(self.cmpStack);
          ComparisonSetter<greater_equal<double> > greater_equal_s(self.cmpStack);
          ComparisonSetter<greater<double> > greater_s(self.cmpStack);
          ComparisonSetter<not_equal_to<double> > not_equal_to_s(self.cmpStack);
          FunctionSetter abs_s(kAbs, self.funStack), acos_s(kAcos, self.funStack), asin_s(kAsin, self.funStack),
              atan2_s(kAtan, self.funStack), atan_s(kAtan, self.funStack), chi2prob_s(kChi2Prob, self.funStack),
              cosh_s(kCosh, self.funStack), cos_s(kCos, self.funStack), exp_s(kExp, self.funStack),
              hypot_s(kHypot, self.funStack), log_s(kLog, self.funStack), log10_s(kLog10, self.funStack),
              max_s(kMax, self.funStack), min_s(kMin, self.funStack), pow_s(kPow, self.funStack),
              sinh_s(kSinh, self.funStack), sin_s(kSin, self.funStack), sqrt_s(kSqrt, self.funStack),
              tanh_s(kTanh, self.funStack), tan_s(kTan, self.funStack), deltaPhi_s(kDeltaPhi, self.funStack),
              deltaR_s(kDeltaR, self.funStack), test_bit_s(kTestBit, self.funStack);
          FunctionSetterCommit funOk_s(self.funStack, self.finalFunStack);
          TrinarySelectorSetter trinary_s(self.selStack, self.cmpStack, self.exprStack);
          BinarySelectorSetter binary_s(self.selStack, self.cmpStack, self.exprStack);
          ExpressionSelectorSetter expr_sel_s(self.selStack, self.exprStack);
          BinaryCutSetter<logical_and<bool> > and_s(self.selStack);
          BinaryCutSetter<logical_or<bool> > or_s(self.selStack);
          UnaryCutSetter<logical_not<bool> > not_s(self.selStack);
          CutSetter cut_s(*self.sel_, self.selStack);
          ExpressionSetter expr_s(*self.expr_, self.exprStack);
          ExpressionBinaryOperatorSetter<plus<double> > plus_s(self.exprStack);
          ExpressionBinaryOperatorSetter<minus<double> > minus_s(self.exprStack);
          ExpressionBinaryOperatorSetter<multiplies<double> > multiplies_s(self.exprStack);
          ExpressionBinaryOperatorSetter<divides<double> > divides_s(self.exprStack);
          ExpressionBinaryOperatorSetter<power_of<double> > power_of_s(self.exprStack);
          ExpressionUnaryOperatorSetter<negate<double> > negate_s(self.exprStack);
          ExpressionFunctionSetter fun_s(self.exprStack, self.finalFunStack);
          //	  Abort abort_s;
          BOOST_SPIRIT_DEBUG_RULE(var);
          BOOST_SPIRIT_DEBUG_RULE(arrayAccess);
          BOOST_SPIRIT_DEBUG_RULE(method);
          BOOST_SPIRIT_DEBUG_RULE(logical_expression);
          BOOST_SPIRIT_DEBUG_RULE(cond_expression);
          BOOST_SPIRIT_DEBUG_RULE(logical_term);
          BOOST_SPIRIT_DEBUG_RULE(logical_factor);
          BOOST_SPIRIT_DEBUG_RULE(number);
          BOOST_SPIRIT_DEBUG_RULE(metharg);
          BOOST_SPIRIT_DEBUG_RULE(function1);
          BOOST_SPIRIT_DEBUG_RULE(function2);
          BOOST_SPIRIT_DEBUG_RULE(function4);
          BOOST_SPIRIT_DEBUG_RULE(expression);
          BOOST_SPIRIT_DEBUG_RULE(term);
          BOOST_SPIRIT_DEBUG_RULE(power);
          BOOST_SPIRIT_DEBUG_RULE(factor);
          BOOST_SPIRIT_DEBUG_RULE(or_op);
          BOOST_SPIRIT_DEBUG_RULE(and_op);
          BOOST_SPIRIT_DEBUG_RULE(comparison_op);
          BOOST_SPIRIT_DEBUG_RULE(binary_comp);
          BOOST_SPIRIT_DEBUG_RULE(trinary_comp);
          BOOST_SPIRIT_DEBUG_RULE(cut);
          BOOST_SPIRIT_DEBUG_RULE(fun);

          boost::spirit::classic::assertion<SyntaxErrors> expectParenthesis(kMissingClosingParenthesis);
          boost::spirit::classic::assertion<SyntaxErrors> expect(kSyntaxError);

          number = real_p[number_s];
          metharg = (strict_real_p[methodArg_s]) | (int_p[methodArg_s]) |
                    (ch_p('"') >> *(~ch_p('"')) >> ch_p('"'))[methodArg_s] |
                    (ch_p('\'') >> *(~ch_p('\'')) >> ch_p('\''))[methodArg_s];
          var =  // alnum_p doesn't accept underscores, so we use chset<>; lexeme_d needed to avoid whitespace skipping within method names
              (lexeme_d[alpha_p >> *chset<>("a-zA-Z0-9_")] >> ch_p('(') >> metharg >> *(ch_p(',') >> metharg) >>
               expectParenthesis(ch_p(')')))[method_s] |
              ((lexeme_d[alpha_p >> *chset<>("a-zA-Z0-9_")])[method_s] >> !(ch_p('(') >> ch_p(')')));
          arrayAccess = (ch_p('[') >> metharg >> *(ch_p(',') >> metharg) >> expectParenthesis(ch_p(']')))[method_s];
          method = (var >> *(arrayAccess | (ch_p('.') >> expect(var))))[var_s];
          function1 = chseq_p("abs")[abs_s] | chseq_p("acos")[acos_s] | chseq_p("asin")[asin_s] |
                      chseq_p("atan")[atan_s] | chseq_p("cosh")[cosh_s] | chseq_p("cos")[cos_s] |
                      chseq_p("exp")[exp_s] | chseq_p("log")[log_s] | chseq_p("log10")[log10_s] |
                      chseq_p("sinh")[sinh_s] | chseq_p("sin")[sin_s] | chseq_p("sqrt")[sqrt_s] |
                      chseq_p("tanh")[tanh_s] | chseq_p("tan")[tan_s];
          function2 = chseq_p("atan2")[atan2_s] | chseq_p("chi2prob")[chi2prob_s] | chseq_p("pow")[pow_s] |
                      chseq_p("min")[min_s] | chseq_p("max")[max_s] | chseq_p("deltaPhi")[deltaPhi_s] |
                      chseq_p("hypot")[hypot_s] | chseq_p("test_bit")[test_bit_s];
          function4 = chseq_p("deltaR")[deltaR_s];
          expression = cond_expression | nocond_expression;
          nocond_expression = term >> (*(('+' >> expect(term))[plus_s] | ('-' >> expect(term))[minus_s]));
          cond_expression = (ch_p('?') >> logical_expression >> ch_p('?') >> expect(expression) >> ch_p(":") >>
                             expect(expression))[cond_s];
          term = power >> *(('*' >> expect(power))[multiplies_s] | ('/' >> expect(power))[divides_s]);
          power = factor >> *(('^' >> expect(factor))[power_of_s]);
          factor =
              number | (function1 >> ch_p('(')[funOk_s] >> expect(expression) >> expectParenthesis(ch_p(')')))[fun_s] |
              (function2 >> ch_p('(')[funOk_s] >> expect(expression) >> expect(ch_p(',')) >> expect(expression) >>
               expectParenthesis(ch_p(')')))[fun_s] |
              (function4 >> ch_p('(')[funOk_s] >> expect(expression) >> expect(ch_p(',')) >> expect(expression) >>
               expect(ch_p(',')) >> expect(expression) >> expect(ch_p(',')) >> expect(expression) >>
               expectParenthesis(ch_p(')')))[fun_s] |
              //NOTE: no expect around the first ch_p('(') otherwise it can't parse a method that starts like a function name (i.e. maxSomething)
              method |
              //NOTE: no 'expectedParenthesis around ending ')' because at this point the partial phrase
              //       "(a"
              //could refer to an expression, e.g., "(a+b)*c" or a logical expression "(a<1) &&"
              //so we need to allow the parser to 'backup' and try a different approach.
              //NOTE: if the parser were changed so a logical expression could be used as an expression,e.g.
              //  (a<b)+1 <2
              // then we could remove such an ambiguity.
              ch_p('(') >> expression >> ch_p(')') | (ch_p('-') >> factor)[negate_s] | (ch_p('+') >> factor);
          comparison_op = (ch_p('<') >> ch_p('=')[less_equal_s]) | (ch_p('<')[less_s]) |
                          (ch_p('=') >> ch_p('=')[equal_to_s]) | (ch_p('=')[equal_to_s]) |
                          (ch_p('>') >> ch_p('=')[greater_equal_s]) | (ch_p('>')[greater_s]) |
                          (ch_p('!') >> ch_p('=')[not_equal_to_s]);
          binary_comp = (expression >> comparison_op >> expect(expression))[binary_s];
          trinary_comp =
              (expression >> comparison_op >> expect(expression) >> comparison_op >> expect(expression))[trinary_s];
          or_op = ch_p('|') >> ch_p('|') | ch_p('|');
          and_op = ch_p('&') >> ch_p('&') | ch_p('&');
          logical_expression = logical_term >> *(or_op >> expect(logical_term))[or_s];
          logical_term = logical_factor >> *(and_op >> expect(logical_factor))[and_s];
          logical_factor = trinary_comp | binary_comp |
                           ch_p('(') >> logical_expression >> expectParenthesis(ch_p(')')) |
                           (ch_p('!') >> expect(logical_factor))[not_s] | expression[expr_sel_s];
          ;
          cut = logical_expression[cut_s];
          fun = expression[expr_s];
          this->start_parsers(cut, fun);
        }
      };
    };
  }  // namespace parser
}  // namespace reco

#endif
