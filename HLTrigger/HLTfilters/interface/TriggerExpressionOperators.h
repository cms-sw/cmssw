#ifndef HLTrigger_HLTfilters_TriggerExpressionOperators_h
#define HLTrigger_HLTfilters_TriggerExpressionOperators_h

#include <boost/scoped_ptr.hpp>
#include "HLTrigger/HLTfilters/interface/TriggerExpressionEvaluator.h"

namespace triggerExpression {

// abstract unary operator
class UnaryOperator : public Evaluator {
public:
  UnaryOperator(Evaluator * arg) :
    m_arg(arg)
  { }

  // configure the dependent module
  void configure(const edm::EventSetup & setup) {
    m_arg->configure(setup);
  }

protected:
  boost::scoped_ptr<Evaluator> m_arg;
};

// abstract binary operator
class BinaryOperator : public Evaluator {
public:
  BinaryOperator(Evaluator * arg1, Evaluator * arg2) :
    m_arg1(arg1),
    m_arg2(arg2)
  { }

  // configure the dependent modules
  void configure(const edm::EventSetup & setup) {
    m_arg1->configure(setup);
    m_arg2->configure(setup);
  }

protected:
  boost::scoped_ptr<Evaluator> m_arg1;
  boost::scoped_ptr<Evaluator> m_arg2;
};


// concrete operators

class OperatorNot : public UnaryOperator {
public:
  OperatorNot(Evaluator * arg) :
    UnaryOperator(arg)
  { }

  bool operator()(const Data & data) {
    return not (*m_arg)(data);
  }
  
  void dump(std::ostream & out) const {
    out << "NOT ";
    m_arg->dump(out);
  }
};

class OperatorAnd : public BinaryOperator {
public:
  OperatorAnd(Evaluator * arg1, Evaluator * arg2) :
    BinaryOperator(arg1, arg2)
  { }

  bool operator()(const Data & data) {
    return (*m_arg1)(data) and (*m_arg2)(data);
  }
  
  void dump(std::ostream & out) const {
    m_arg1->dump(out);
    out << " AND ";
    m_arg2->dump(out);
  }
};

class OperatorOr : public BinaryOperator {
public:
  OperatorOr(Evaluator * arg1, Evaluator * arg2) :
    BinaryOperator(arg1, arg2)
  { }

  bool operator()(const Data & data) {
    return (*m_arg1)(data) or (*m_arg2)(data);
  }
  
  void dump(std::ostream & out) const {
    m_arg1->dump(out);
    out << " OR ";
    m_arg2->dump(out);
  }
};

class OperatorXor : public BinaryOperator {
public:
  OperatorXor(Evaluator * arg1, Evaluator * arg2) :
    BinaryOperator(arg1, arg2)
  { }

  bool operator()(const Data & data) {
    return (*m_arg1)(data) xor (*m_arg2)(data);
  }
  
  void dump(std::ostream & out) const {
    m_arg1->dump(out);
    out << " XOR ";
    m_arg2->dump(out);
  }
};

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionOperators_h
