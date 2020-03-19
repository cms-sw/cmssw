/** \class edm::PathStatusFilter

Allows one to filter based on the results of Paths.
One configures the filter with a text string that
is a logical expression of path names. When the
referenced Paths complete, the PathStatus product
for each Path is retrieved from the Event. If the
PathStatus holds the value "Pass", then the corresponding
operand in the logical expression evaluates to "true",
otherwise it evaluates to false. The overall logical
expression is then evaluated and its result is the
filter return value.

The logical expression syntax is very intuitive. The
operands are path names. The allowed operators in order
of precedence are "not", "and", and "or". Parentheses
can be used. The syntax requires operators and pathnames
be separated by at least one space or parenthesis.
Extra space between operators, path names, or parentheses
is ignored. A path name cannot be the same as an
operator name. For example, a valid expression would be:

  path1 and not (path2 or not path3)

Note that this works only for Paths in the current process.
It does not work for EndPaths or Paths from prior processes.

\author W. David Dagenhart, created 31 July, 2017

*/

#include "DataFormats/Common/interface/PathStatus.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <boost/spirit/include/phoenix_bind.hpp>
#include <boost/spirit/include/qi.hpp>

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <sstream>
#include <vector>
#include <utility>

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace phoenix = boost::phoenix;

namespace edm {

  class EventSetup;
  class StreamID;

  namespace pathStatusExpression {
    class Evaluator;
  }

  class PathStatusFilter : public global::EDFilter<> {
  public:
    explicit PathStatusFilter(ParameterSet const&);
    static void fillDescriptions(ConfigurationDescriptions&);
    bool filter(StreamID, Event&, EventSetup const&) const final;

  private:
    edm::propagate_const<std::unique_ptr<pathStatusExpression::Evaluator>> evaluator_;
    bool verbose_;
  };

  namespace pathStatusExpression {

    class Evaluator {
    public:
      virtual ~Evaluator() {}

      enum EvaluatorType { Name, Not, And, Or, BeginParen };
      virtual EvaluatorType type() const = 0;

      virtual void setLeft(std::unique_ptr<Evaluator>&&) {}
      virtual void setRight(std::unique_ptr<Evaluator>&&) {}

      virtual void print(std::ostream& out, unsigned int indentation) const {}
      virtual void init(ConsumesCollector&) {}
      virtual bool evaluate(Event const& event) const { return true; };
    };

    class Operand : public Evaluator {
    public:
      Operand(std::vector<char> const& pathName) : pathName_(pathName.begin(), pathName.end()) {}

      EvaluatorType type() const override { return Name; }

      void print(std::ostream& out, unsigned int indentation) const override {
        out << std::string(indentation, ' ') << pathName_ << "\n";
      }

      void init(ConsumesCollector& iC) override { token_ = iC.consumes<PathStatus>(InputTag(pathName_)); }

      bool evaluate(Event const& event) const override { return event.get(token_).accept(); }

    private:
      std::string pathName_;
      EDGetTokenT<PathStatus> token_;
    };

    class NotOperator : public Evaluator {
    public:
      EvaluatorType type() const override { return Not; }

      void setLeft(std::unique_ptr<Evaluator>&& v) override { operand_ = std::move(v); }

      void print(std::ostream& out, unsigned int indentation) const override {
        out << std::string(indentation, ' ') << "not\n";
        operand_->print(out, indentation + 4);
      }

      void init(ConsumesCollector& iC) override { operand_->init(iC); }

      bool evaluate(Event const& event) const override { return !operand_->evaluate(event); }

    private:
      edm::propagate_const<std::unique_ptr<Evaluator>> operand_;
    };

    template <typename T>
    class BinaryOperator : public Evaluator {
    public:
      EvaluatorType type() const override;

      void setLeft(std::unique_ptr<Evaluator>&& v) override { left_ = std::move(v); }
      void setRight(std::unique_ptr<Evaluator>&& v) override { right_ = std::move(v); }

      void print(std::ostream& out, unsigned int indentation) const override;

      void init(ConsumesCollector& iC) override {
        left_->init(iC);
        right_->init(iC);
      }

      bool evaluate(Event const& event) const override {
        T op;
        return op(left_->evaluate(event), right_->evaluate(event));
      }

    private:
      edm::propagate_const<std::unique_ptr<Evaluator>> left_;
      edm::propagate_const<std::unique_ptr<Evaluator>> right_;
    };

    template <>
    inline Evaluator::EvaluatorType BinaryOperator<std::logical_and<bool>>::type() const {
      return And;
    }

    template <>
    inline Evaluator::EvaluatorType BinaryOperator<std::logical_or<bool>>::type() const {
      return Or;
    }

    template <>
    void BinaryOperator<std::logical_and<bool>>::print(std::ostream& out, unsigned int indentation) const {
      out << std::string(indentation, ' ') << "and\n";
      left_->print(out, indentation + 4);
      right_->print(out, indentation + 4);
    }
    template <>
    void BinaryOperator<std::logical_or<bool>>::print(std::ostream& out, unsigned int indentation) const {
      out << std::string(indentation, ' ') << "or\n";
      left_->print(out, indentation + 4);
      right_->print(out, indentation + 4);
    }

    using AndOperator = BinaryOperator<std::logical_and<bool>>;
    using OrOperator = BinaryOperator<std::logical_or<bool>>;

    class BeginParenthesis : public Evaluator {
    public:
      EvaluatorType type() const override { return BeginParen; }
    };

    // This class exists to properly handle the precedence of the
    // operators and also handle the order of operations specified
    // by parentheses. (search for shunting yard algorithm on the
    // internet for a description of this algorithm)
    class ShuntingYardAlgorithm {
    public:
      void addPathName(std::vector<char> const& s) { operandStack.push_back(std::make_unique<Operand>(s)); }

      void addOperatorNot() {
        if (operatorStack.empty() || operatorStack.back()->type() != Evaluator::Not) {
          operatorStack.push_back(std::make_unique<NotOperator>());
        } else {
          // Two Not operations in a row cancel and are the same as no operation at all.
          operatorStack.pop_back();
        }
      }

      void moveBinaryOperator() {
        std::unique_ptr<Evaluator>& backEvaluator = operatorStack.back();
        backEvaluator->setRight(std::move(operandStack.back()));
        operandStack.pop_back();
        backEvaluator->setLeft(std::move(operandStack.back()));
        operandStack.pop_back();
        operandStack.push_back(std::move(backEvaluator));
        operatorStack.pop_back();
      }

      void moveNotOperator() {
        std::unique_ptr<Evaluator>& backEvaluator = operatorStack.back();
        backEvaluator->setLeft(std::move(operandStack.back()));
        operandStack.pop_back();
        operandStack.push_back(std::move(backEvaluator));
        operatorStack.pop_back();
      }

      void addOperatorAnd() {
        while (!operatorStack.empty()) {
          std::unique_ptr<Evaluator>& backEvaluator = operatorStack.back();
          if (backEvaluator->type() == Evaluator::And) {
            moveBinaryOperator();
          } else if (backEvaluator->type() == Evaluator::Not) {
            moveNotOperator();
          } else {
            break;
          }
        }
        operatorStack.push_back(std::make_unique<AndOperator>());
      }

      void addOperatorOr() {
        while (!operatorStack.empty()) {
          std::unique_ptr<Evaluator>& backEvaluator = operatorStack.back();
          if (backEvaluator->type() == Evaluator::And || backEvaluator->type() == Evaluator::Or) {
            moveBinaryOperator();
          } else if (backEvaluator->type() == Evaluator::Not) {
            moveNotOperator();
          } else {
            break;
          }
        }
        operatorStack.push_back(std::make_unique<OrOperator>());
      }

      void addBeginParenthesis() { operatorStack.push_back(std::make_unique<BeginParenthesis>()); }

      void addEndParenthesis() {
        while (!operatorStack.empty()) {
          std::unique_ptr<Evaluator>& backEvaluator = operatorStack.back();
          if (backEvaluator->type() == Evaluator::BeginParen) {
            operatorStack.pop_back();
            break;
          }
          if (backEvaluator->type() == Evaluator::And || backEvaluator->type() == Evaluator::Or) {
            moveBinaryOperator();
          } else if (backEvaluator->type() == Evaluator::Not) {
            moveNotOperator();
          }
        }
      }

      std::unique_ptr<Evaluator> finish() {
        while (!operatorStack.empty()) {
          std::unique_ptr<Evaluator>& backEvaluator = operatorStack.back();
          // Just a sanity check. The grammar defined for the boost Spirit parser
          // should catch any errors of this type before we get here.
          if (backEvaluator->type() == Evaluator::BeginParen) {
            throw cms::Exception("LogicError")
                << "Should be impossible to get this error. Contact a Framework developer";
          }
          if (backEvaluator->type() == Evaluator::And || backEvaluator->type() == Evaluator::Or) {
            moveBinaryOperator();
          } else if (backEvaluator->type() == Evaluator::Not) {
            moveNotOperator();
          }
        }
        // Just a sanity check. The grammar defined for the boost Spirit parser
        // should catch any errors of this type before we get here.
        if (!operatorStack.empty() || operandStack.size() != 1U) {
          throw cms::Exception("LogicError") << "Should be impossible to get this error. Contact a Framework developer";
        }
        std::unique_ptr<Evaluator> temp = std::move(operandStack.back());
        operandStack.pop_back();
        return temp;
      }

    private:
      std::vector<std::unique_ptr<Evaluator>> operandStack;
      std::vector<std::unique_ptr<Evaluator>> operatorStack;
    };

    // Use boost Spirit to parse the logical expression character string
    template <typename Iterator>
    class Grammar : public qi::grammar<Iterator, ascii::space_type> {
    public:
      Grammar(ShuntingYardAlgorithm* algorithm) : Grammar::base_type(expression), algorithm_(algorithm) {
        // setup functors that call into shunting algorithm while parsing the logical expression
        auto addPathName = phoenix::bind(&ShuntingYardAlgorithm::addPathName, algorithm_, qi::_1);
        auto addOperatorNot = phoenix::bind(&ShuntingYardAlgorithm::addOperatorNot, algorithm_);
        auto addOperatorAnd = phoenix::bind(&ShuntingYardAlgorithm::addOperatorAnd, algorithm_);
        auto addOperatorOr = phoenix::bind(&ShuntingYardAlgorithm::addOperatorOr, algorithm_);
        auto addBeginParenthesis = phoenix::bind(&ShuntingYardAlgorithm::addBeginParenthesis, algorithm_);
        auto addEndParenthesis = phoenix::bind(&ShuntingYardAlgorithm::addEndParenthesis, algorithm_);

        // Define the syntax allowed in the logical expressions
        pathName = !unaryOperator >> !binaryOperatorTest >> (+qi::char_("a-zA-Z0-9_"))[addPathName];
        binaryOperand = (qi::lit('(')[addBeginParenthesis] >> expression >> qi::lit(')')[addEndParenthesis]) |
                        (unaryOperator[addOperatorNot] >> binaryOperand) | pathName;
        afterOperator = ascii::space | &qi::lit('(') | &qi::eoi;
        unaryOperator = qi::lit("not") >> afterOperator;
        // The only difference in the next two is that one calls a functor and the other does not
        binaryOperatorTest = (qi::lit("and") >> afterOperator) | (qi::lit("or") >> afterOperator);
        binaryOperator =
            (qi::lit("and") >> afterOperator)[addOperatorAnd] | (qi::lit("or") >> afterOperator)[addOperatorOr];
        expression = binaryOperand % binaryOperator;
      }

    private:
      qi::rule<Iterator> pathName;
      qi::rule<Iterator, ascii::space_type> binaryOperand;
      qi::rule<Iterator> afterOperator;
      qi::rule<Iterator> unaryOperator;
      qi::rule<Iterator> binaryOperatorTest;
      qi::rule<Iterator> binaryOperator;
      qi::rule<Iterator, ascii::space_type> expression;

      ShuntingYardAlgorithm* algorithm_;
    };
  }  // namespace pathStatusExpression

  PathStatusFilter::PathStatusFilter(ParameterSet const& pset)
      : evaluator_(nullptr), verbose_(pset.getUntrackedParameter<bool>("verbose")) {
    std::string const logicalExpression = pset.getParameter<std::string>("logicalExpression");
    if (verbose_) {
      LogAbsolute("PathStatusFilter") << "PathStatusFilter logicalExpression = " << logicalExpression;
    }

    if (logicalExpression.empty()) {
      return;
    }

    pathStatusExpression::ShuntingYardAlgorithm shuntingYardAlgorithm;

    pathStatusExpression::Grammar<std::string::const_iterator> grammar(&shuntingYardAlgorithm);

    auto it = logicalExpression.cbegin();
    if (!qi::phrase_parse(it, logicalExpression.cend(), grammar, ascii::space) || (it != logicalExpression.cend())) {
      throw cms::Exception("Configuration") << "Syntax error in logical expression. Here is an example of how\n"
                                            << "the syntax should look:\n"
                                            << "    \"path1 and not (path2 or not path3)\"\n"
                                            << "The expression must contain alternating appearances of operands\n"
                                            << "which are path names and binary operators which can be \'and\'\n"
                                            << "or \'or\', with a path name at the beginning and end. There\n"
                                            << "must be at least one path name. In addition to the alternating\n"
                                            << "path names and binary operators, the unary operator \'not\' can\n"
                                            << "be inserted before a path name or a begin parenthesis.\n"
                                            << "Parentheses are allowed. Parentheses must come in matching pairs.\n"
                                            << "Matching begin and end parentheses must contain a complete and\n"
                                            << "syntactically correct logical expression. There must be at least\n"
                                            << "one space or parenthesis between operators and path names. Extra\n"
                                            << "space is ignored and OK. Path names can only contain upper and\n"
                                            << "lower case letters, numbers, and underscores. A path name cannot\n"
                                            << "be the same as an operator name.\n";
    }

    evaluator_ = shuntingYardAlgorithm.finish();
    if (verbose_) {
      std::stringstream out;
      evaluator_->print(out, 0);
      LogAbsolute("PathStatusFilter") << out.str();
    }
    ConsumesCollector iC(consumesCollector());
    evaluator_->init(iC);
  }

  void PathStatusFilter::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.add<std::string>("logicalExpression", std::string())
        ->setComment(
            "Operands are path names. Operators in precedence order "
            "\'not\', \'and\', and \'or\'. Parentheses allowed.");
    desc.addUntracked<bool>("verbose", false)->setComment("For debugging only");
    descriptions.add("pathStatusFilter", desc);
  }

  bool PathStatusFilter::filter(StreamID, Event& event, EventSetup const&) const {
    if (evaluator_ == nullptr) {
      return true;
    }
    return evaluator_->evaluate(event);
  }
}  // namespace edm

using edm::PathStatusFilter;
DEFINE_FWK_MODULE(PathStatusFilter);
