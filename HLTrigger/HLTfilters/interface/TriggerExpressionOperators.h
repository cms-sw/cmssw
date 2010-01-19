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
  BinaryOperator(TriggerExpressionEvaluator * arg1, TriggerExpressionEvaluator * arg2) :
    m_arg1(arg1),
    m_arg2(arg2)
  { }

  // configure the dependent modules
  void configure(const edm::EventSetup & setup) {
    m_arg1->configure(setup);
    m_arg2->configure(setup);
  }

protected:
  boost::scoped_ptr<TriggerExpressionEvaluator> m_arg1;
  boost::scoped_ptr<TriggerExpressionEvaluator> m_arg2;
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
  
  virtual void dump(std::ostream & out) const {
    out << "NOT ";
    m_arg->dump(out);
  }
};

class OperatorAnd : public BinaryOperator {
public:
  OperatorAnd(TriggerExpressionEvaluator * arg1, TriggerExpressionEvaluator * arg2) :
    BinaryOperator(arg1, arg2)
  { }

  bool operator()(const TriggerExpressionCache & data) {
    return (*m_arg1)(data) and (*m_arg2)(data);
  }
  
  virtual void dump(std::ostream & out) const {
    m_arg1->dump(out);
    out << " AND ";
    m_arg2->dump(out);
  }
};

class OperatorOr : public BinaryOperator {
public:
  OperatorOr(TriggerExpressionEvaluator * arg1, TriggerExpressionEvaluator * arg2) :
    BinaryOperator(arg1, arg2)
  { }

  bool operator()(const TriggerExpressionCache & data) {
    return (*m_arg1)(data) or (*m_arg2)(data);
  }
  
  virtual void dump(std::ostream & out) const {
    m_arg1->dump(out);
    out << " OR ";
    m_arg2->dump(out);
  }
};

class OperatorXor : public BinaryOperator {
public:
  OperatorOr(TriggerExpressionEvaluator * arg1, TriggerExpressionEvaluator * arg2) :
    BinaryOperator(arg1, arg2)
  { }

  bool operator()(const TriggerExpressionCache & data) {
    return (*m_arg1)(data) xor (*m_arg2)(data);
  }
  
  virtual void dump(std::ostream & out) const {
    m_arg1->dump(out);
    out << " XOR ";
    m_arg2->dump(out);
  }
};

} // namespace hlt

#endif // HLTrigger_HLTfilters_ExpressionOperators_h
