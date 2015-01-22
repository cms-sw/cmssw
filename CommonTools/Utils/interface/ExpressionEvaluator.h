#ifndef CommonToolsUtilsExpressionEvaluator_H
#define CommonToolsUtilsExpressionEvaluator_H


#include<string>

namespace reco {

class ExpressionEvaluator {
public:
  ExpressionEvaluator(const char * pkg,  const char * iname, const char * iexpr);
  ~ExpressionEvaluator();
  
  template< typename EXPR>
  EXPR * expr() const { 
    typedef EXPR * factoryP();
    return reinterpret_cast<factoryP*>(m_expr)();
  }

private:

  std::string m_name;
  void * m_expr;
};

}


#endif // CommonToolsUtilsExpressionEvaluator_H

