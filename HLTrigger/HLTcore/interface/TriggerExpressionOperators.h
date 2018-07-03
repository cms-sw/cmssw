#ifndef HLTrigger_HLTfilters_TriggerExpressionOperators_h
#define HLTrigger_HLTfilters_TriggerExpressionOperators_h

#include <boost/scoped_ptr.hpp>
#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"

namespace triggerExpression {

// abstract unary operator
class UnaryOperator : public Evaluator {
public:
  UnaryOperator(Evaluator * arg) :
    m_arg(arg)
  { }

  // initialize the depending modules
  void init(const Data & data) override { 
    m_arg->init(data);
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

  // initialize the depending modules
  void init(const Data & data) override { 
    m_arg1->init(data);
    m_arg2->init(data);
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

  bool operator()(const Data & data) const override {
    return not (*m_arg)(data);
  }
  
  void dump(std::ostream & out) const override {
    out << "NOT ";
    m_arg->dump(out);
  }
};

class OperatorAnd : public BinaryOperator {
public:
  OperatorAnd(Evaluator * arg1, Evaluator * arg2) :
    BinaryOperator(arg1, arg2)
  { }

  bool operator()(const Data & data) const override {
    // force the execution af both arguments, otherwise precalers won't work properly
    bool r1 = (*m_arg1)(data);
    bool r2 = (*m_arg2)(data);
    return r1 and r2;
  }
  
  void dump(std::ostream & out) const override {
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

  bool operator()(const Data & data) const override {
    // force the execution af both arguments, otherwise precalers won't work properly
    bool r1 = (*m_arg1)(data);
    bool r2 = (*m_arg2)(data);
    return r1 or r2;
  }
  
  void dump(std::ostream & out) const override {
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

  bool operator()(const Data & data) const override {
    // force the execution af both arguments, otherwise precalers won't work properly
    bool r1 = (*m_arg1)(data);
    bool r2 = (*m_arg2)(data);
    return r1 xor r2;
  }
  
  void dump(std::ostream & out) const override {
    m_arg1->dump(out);
    out << " XOR ";
    m_arg2->dump(out);
  }
};

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionOperators_h
