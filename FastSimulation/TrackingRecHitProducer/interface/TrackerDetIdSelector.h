#ifndef FastSimulation_TrackingRecHitProducer_TrackerDetIdSelector_H
#define FastSimulation_TrackingRecHitProducer_TrackerDetIdSelector_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <boost/variant/recursive_variant.hpp>

#include <iostream>
#include <cstdlib>
#include <typeinfo>
#include <functional>
#include <unordered_map>

struct BinaryOP;
struct UnaryOP;
struct Nil {};

struct ExpressionAST {
  typedef boost::variant<Nil,
                         int,
                         std::string,
                         boost::recursive_wrapper<ExpressionAST>,
                         boost::recursive_wrapper<BinaryOP>,
                         boost::recursive_wrapper<UnaryOP> >
      Type;

  ExpressionAST() : expr(Nil()) {}

  template <typename Expr>
  ExpressionAST(Expr const& expr) : expr(expr) {}

  int evaluate(const DetId& detId, const TrackerTopology& trackerTopology) const;

  ExpressionAST& operator!();

  Type expr;
};

ExpressionAST operator>(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator>=(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator==(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator<=(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator<(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator!=(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator&&(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator||(ExpressionAST const& lhs, ExpressionAST const& rhs);

struct BinaryOP {
  enum class OP { GREATER, GREATER_EQUAL, EQUAL, LESS_EQUAL, LESS, NOT_EQUAL, AND, OR } op;
  ExpressionAST left;
  ExpressionAST right;

  BinaryOP(OP op, ExpressionAST const& left, ExpressionAST const& right) : op(op), left(left), right(right) {}

  int evaluate(const DetId& detId, const TrackerTopology& trackerTopology) const {
    switch (op) {
      case OP::GREATER:
        return left.evaluate(detId, trackerTopology) > right.evaluate(detId, trackerTopology);
      case OP::GREATER_EQUAL:
        return left.evaluate(detId, trackerTopology) >= right.evaluate(detId, trackerTopology);
      case OP::EQUAL:
        return left.evaluate(detId, trackerTopology) == right.evaluate(detId, trackerTopology);
      case OP::LESS_EQUAL:
        return left.evaluate(detId, trackerTopology) <= right.evaluate(detId, trackerTopology);
      case OP::LESS:
        return left.evaluate(detId, trackerTopology) < right.evaluate(detId, trackerTopology);
      case OP::NOT_EQUAL:
        return left.evaluate(detId, trackerTopology) != right.evaluate(detId, trackerTopology);
      case OP::AND:
        return left.evaluate(detId, trackerTopology) && right.evaluate(detId, trackerTopology);
      case OP::OR:
        return left.evaluate(detId, trackerTopology) || right.evaluate(detId, trackerTopology);
    }
    return 0;
  }
};

struct UnaryOP {
  enum class OP { NEG } op;
  ExpressionAST subject;
  UnaryOP(OP op, ExpressionAST const& subject) : op(op), subject(subject) {}

  int evaluate(const DetId& detId, const TrackerTopology& trackerTopology) const {
    switch (op) {
      case OP::NEG:
        return !subject.evaluate(detId, trackerTopology);
    }
    return 0;
  }
};

class TrackerDetIdSelector {
private:
  const DetId& _detId;
  const TrackerTopology& _trackerTopology;

public:
  typedef std::function<int(const TrackerTopology& trackerTopology, const DetId&)> DetIdFunction;
  typedef std::unordered_map<std::string, DetIdFunction> StringFunctionMap;
  const static StringFunctionMap functionTable;

  TrackerDetIdSelector(const DetId& detId, const TrackerTopology& trackerTopology)
      : _detId(detId), _trackerTopology(trackerTopology) {}

  bool passSelection(const std::string& selectionStr) const;
};

class Accessor : public boost::static_visitor<int> {
private:
  const DetId& _detId;
  const TrackerTopology& _trackerTopology;

public:
  Accessor(const DetId& detId, const TrackerTopology& trackerTopology)
      : _detId(detId), _trackerTopology(trackerTopology) {}

  int operator()(Nil i) const {
    throw cms::Exception("FastSimulation/TrackingRecHitProducer/TrackerDetIdSelector",
                         "while evaluating a DetId selection a symbol was not set");
  }
  int operator()(const int& i) const { return i; }
  int operator()(const std::string& s) const {
    TrackerDetIdSelector::StringFunctionMap::const_iterator it = TrackerDetIdSelector::functionTable.find(s);
    int value = 0;
    if (it != TrackerDetIdSelector::functionTable.cend()) {
      value = (it->second)(_trackerTopology, _detId);
      //std::cout<<"attr="<<s<<", value="<<value<<std::endl;
    } else {
      //std::cout<<"attr="<<s<<" unknown"<<std::endl;
      std::string msg =
          "error while parsing DetId selection: named identifier '" + s + "' not known. Possible values are: ";
      for (const TrackerDetIdSelector::StringFunctionMap::value_type& pair : TrackerDetIdSelector::functionTable) {
        msg += pair.first + ",";
      }
      throw cms::Exception("FastSimulation/TrackingRecHitProducer/TrackerDetIdSelector", msg);
    }
    return value;
  }
  int operator()(const ExpressionAST& ast) const { return ast.evaluate(_detId, _trackerTopology); }
  int operator()(const BinaryOP& binaryOP) const { return binaryOP.evaluate(_detId, _trackerTopology); }
  int operator()(const UnaryOP& unaryOP) const { return unaryOP.evaluate(_detId, _trackerTopology); }
};

struct WalkAST {
  Accessor _acc;

  WalkAST(const DetId& detId, const TrackerTopology& trackerTopology) : _acc(detId, trackerTopology) {}

  typedef void result_type;

  void operator()() const {}
  void operator()(int n) const {
    std::cout << n;
    std::cout << " [" << _acc(n) << "] ";
  }
  void operator()(std::string str) const {
    std::cout << str;
    std::cout << " [" << _acc(str) << "] ";
  }
  void operator()(ExpressionAST const& ast) const {
    boost::apply_visitor(*this, ast.expr);
    std::cout << " [=" << _acc(ast) << "] ";
  }

  void operator()(BinaryOP const& expr) const {
    std::cout << "(";
    boost::apply_visitor(*this, expr.left.expr);
    switch (expr.op) {
      case BinaryOP::OP::GREATER:
        std::cout << " > ";
        break;
      case BinaryOP::OP::GREATER_EQUAL:
        std::cout << " >= ";
        break;
      case BinaryOP::OP::EQUAL:
        std::cout << " == ";
        break;
      case BinaryOP::OP::LESS_EQUAL:
        std::cout << " <= ";
        break;
      case BinaryOP::OP::LESS:
        std::cout << " < ";
        break;
      case BinaryOP::OP::NOT_EQUAL:
        std::cout << " != ";
        break;
      case BinaryOP::OP::AND:
        std::cout << " && ";
        break;
      case BinaryOP::OP::OR:
        std::cout << " || ";
        break;
    }
    boost::apply_visitor(*this, expr.right.expr);
    std::cout << ')';
  }

  void operator()(UnaryOP const& expr) const {
    switch (expr.op) {
      case UnaryOP::OP::NEG:
        std::cout << " !(";
        break;
    }
    boost::apply_visitor(*this, expr.subject.expr);
    std::cout << ')';
  }
};

inline int ExpressionAST::evaluate(const DetId& detId, const TrackerTopology& trackerTopology) const {
  return boost::apply_visitor(Accessor(detId, trackerTopology), this->expr);
}

#endif
