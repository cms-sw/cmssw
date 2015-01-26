#ifndef CommonToolsUtilsExpressionEvaluator_H
#define CommonToolsUtilsExpressionEvaluator_H


#include<string>

namespace reco {

class ExpressionEvaluator {
public:
  ExpressionEvaluator(const char * pkg,  const char * iname, const std::string & iexpr);
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


  template< typename EXPR>
  EXPR * expressionEvaluator(const char * pkg,  const char * iname, const std::string & iexpr) {
    ExpressionEvaluator ee(pkg, iname,iexpr);
    return ee.expr<EXPR>();
 }

}

#define RECO_XSTR(s) RECO_STR(s)
#define RECO_STR(s) #s
#define reco_expressionEvaluator(pkg, EXPR, iexpr) reco::expressionEvaluator<EXPR>(pkg,RECO_XSTR(EXPR),iexpr)

#endif // CommonToolsUtilsExpressionEvaluator_H

