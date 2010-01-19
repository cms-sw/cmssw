#ifndef HLTrigger_HLTfilters_ExpressionOperators_h
#define HLTrigger_HLTfilters_ExpressionOperators_h

#include <boost/scoped_ptr.hpp>
#include "HLTrigger/HLTfilters/interface/TriggerExpressionEvaluator.h"

namespace hlt {

// abstract unary operator
class UnaryOperator : public TriggerExpressionEvaluator {
public:
  UnaryOperator(TriggerExpressionEvaluator * arg) :
    m_arg(arg)
  { }

  // configure the dependent module
  void configure(const edm::EventSetup & setup) {
    m_arg->configure(setup);
  }

protected:
  boost::scoped_ptr<TriggerExpressionEvaluator> m_arg;
};

// abstract binary operator
class BinaryOperator : public TriggerExpressionEvaluator {
public:
  BinaryOperator(TriggerExpressionEvaluator * lhs, TriggerExpressionEvaluator * rhs) :
    m_lhs(lhs),
    m_rhs(rhs)
  { }

  // configure the dependent modules
  void configure(const edm::EventSetup & setup) {
    m_lhs->configure(setup);
    m_rhs->configure(setup);
  }

protected:
  boost::scoped_ptr<TriggerExpressionEvaluator> m_lhs;
  boost::scoped_ptr<TriggerExpressionEvaluator> m_rhs;
};


// concrete operators

class OperatorNot : public UnaryOperator {
public:
  OperatorNot(TriggerExpressionEvaluator * arg) :
    UnaryOperator(arg)
  { }

  bool operator()(const TriggerExpressionCache & data) {
    return not (*m_arg)(data);
  }
};

class OperatorAnd : public BinaryOperator {
public:
  OperatorAnd(TriggerExpressionEvaluator * lhs, TriggerExpressionEvaluator * rhs) :
    BinaryOperator(lhs, rhs)
  { }

  bool operator()(const TriggerExpressionCache & data) {
    return (*m_lhs)(data) and (*m_rhs)(data);
  }
};

class OperatorOr : public BinaryOperator {
public:
  OperatorOr(TriggerExpressionEvaluator * lhs, TriggerExpressionEvaluator * rhs) :
    BinaryOperator(lhs, rhs)
  { }

  bool operator()(const TriggerExpressionCache & data) {
    return (*m_lhs)(data) or (*m_rhs)(data);
  }
};

class OperatorXor : public BinaryOperator {
public:
  OperatorOr(TriggerExpressionEvaluator * lhs, TriggerExpressionEvaluator * rhs) :
    BinaryOperator(lhs, rhs)
  { }

  bool operator()(const TriggerExpressionCache & data) {
    return (*m_lhs)(data) xor (*m_rhs)(data);
  }
};

} // namespace hlt

#endif // HLTrigger_HLTfilters_ExpressionOperators_h
