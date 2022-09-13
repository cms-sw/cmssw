#ifndef HLTrigger_HLTcore_TriggerExpressionParser_h
#define HLTrigger_HLTcore_TriggerExpressionParser_h

// Note: this requires Boost 1.41 or higher, for Spirit 2.1 or higher
#include <boost/phoenix.hpp>
#include <boost/spirit/include/qi.hpp>

#include "HLTrigger/HLTcore/interface/TriggerExpressionPathReader.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionL1uGTReader.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionOperators.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionPrescaler.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionConstant.h"

namespace triggerExpression {

  namespace qi = boost::spirit::qi;
  namespace ascii = boost::spirit::ascii;

  using boost::phoenix::new_;
  using boost::spirit::unused_type;

  template <typename Iterator>
  class Parser : public qi::grammar<Iterator, Evaluator *(), ascii::space_type> {
  public:
    Parser() : Parser::base_type(expression) {
      auto delimiter = qi::copy(qi::eoi | !qi::char_("a-zA-Z0-9_*?"));

      operand_not = qi::lexeme[qi::lit("NOT") >> delimiter];
      operand_and = qi::lexeme[qi::lit("AND") >> delimiter];
      operand_or = qi::lexeme[qi::lit("OR") >> delimiter];
      operand_xor = qi::lexeme[qi::lit("XOR") >> delimiter];
      operand_masking = qi::lexeme[qi::lit("MASKING") >> delimiter];

      // "TRUE": keyword to accept all events
      token_true = qi::lexeme[qi::lit("TRUE") >> delimiter];

      // "FALSE": keyword to reject all events
      token_false = qi::lexeme[qi::lit("FALSE") >> delimiter];

      // Level-1 Global Trigger decisions: must begin with characters "L1_"
      token_l1algo %= qi::raw[qi::lexeme["L1_" >> +(qi::char_("a-zA-Z0-9_*?"))]];

      // Decisions of Paths in the CMSSW configuration (e.g. high-level triggers):
      // any alphanumeric pattern except for "TRUE", "FALSE", "NOT", "AND", "OR", "XOR" and "MASKING"
      token_path %= qi::raw[qi::lexeme[+(qi::char_("a-zA-Z0-9_*?"))] - token_true - token_false - operand_not -
                            operand_and - operand_or - operand_xor - operand_masking];

      token = (token_true[qi::_val = new_<Constant>(true)] | token_false[qi::_val = new_<Constant>(false)] |
               token_l1algo[qi::_val = new_<L1uGTReader>(qi::_1)] | token_path[qi::_val = new_<PathReader>(qi::_1)]);

      parenthesis %= ('(' >> expression >> ')');

      element %= (token | parenthesis);

      prescale = (element >> '/' >> qi::uint_)[qi::_val = new_<Prescaler>(qi::_1, qi::_2)];

      operand %= (prescale | element);

      unary = ((operand_not >> unary)[qi::_val = new_<OperatorNot>(qi::_1)] | operand[qi::_val = qi::_1]);

      // token_masking is used to restrict the argument (rhs) of the "MASKING"
      // operation to Constant[FALSE], L1uGTReader and PathReader evaluators
      token_masking =
          (token_false[qi::_val = new_<Constant>(false)] | token_l1algo[qi::_val = new_<L1uGTReader>(qi::_1)] |
           token_path[qi::_val = new_<PathReader>(qi::_1)]);

      argument_masking %= (token_masking | ('(' >> argument_masking >> ')'));

      expression = unary[qi::_val = qi::_1] >>
                   *((operand_and >> unary)[qi::_val = new_<OperatorAnd>(qi::_val, qi::_1)] |
                     (operand_or >> unary)[qi::_val = new_<OperatorOr>(qi::_val, qi::_1)] |
                     (operand_xor >> unary)[qi::_val = new_<OperatorXor>(qi::_val, qi::_1)] |
                     (operand_masking >> argument_masking)[qi::_val = new_<OperatorMasking>(qi::_val, qi::_1)]);
    }

  private:
    typedef qi::rule<Iterator, std::string(), ascii::space_type> name_rule;
    typedef qi::rule<Iterator, Evaluator *(), ascii::space_type> rule;
    typedef qi::rule<Iterator> terminal_rule;

    terminal_rule token_true;
    terminal_rule token_false;

    terminal_rule operand_not;
    terminal_rule operand_and;
    terminal_rule operand_or;
    terminal_rule operand_xor;
    terminal_rule operand_masking;

    name_rule token_l1algo;
    name_rule token_path;

    rule token;
    rule parenthesis;
    rule element;
    rule prescale;
    rule operand;
    rule unary;
    rule token_masking;
    rule argument_masking;
    rule expression;
  };

  // generic interface for string-like objects
  template <class T>
  Evaluator *parse(const T &text) {
    typedef typename T::const_iterator Iterator;
    Parser<Iterator> parser;
    Evaluator *evaluator = nullptr;

    Iterator begin = text.begin();
    Iterator end = text.end();

    // the interface of qi::phrase_parse has changed between Boost 1.40 (Spirit 2.0) and Boost 1.41 (Spirit 2.1)
    bool result = qi::phrase_parse(begin, end, parser, ascii::space, evaluator);

    if (not result or begin != end) {
      delete evaluator;
      return nullptr;
    }

    return evaluator;
  }

  // overloaded interface for null-terminated strings
  inline Evaluator *parse(const char *text) {
    Parser<const char *> parser;
    Evaluator *evaluator = nullptr;

    const char *begin = text;
    const char *end = text + strlen(text);

    // the interface of qi::phrase_parse has changed between Boost 1.40 (Spirit 2.0) and Boost 1.41 (Spirit 2.1)
    bool result = qi::phrase_parse(begin, end, parser, ascii::space, evaluator);

    if (not result or begin != end) {
      delete evaluator;
      return nullptr;
    }

    return evaluator;
  }

}  // namespace triggerExpression

#endif  // HLTrigger_HLTcore_TriggerExpressionParser_h
