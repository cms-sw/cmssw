#ifndef CommonToolsUtilsExpressionEvaluator_H
#define CommonToolsUtilsExpressionEvaluator_H


#include<string>

namespace reco {

class ExpressionEvaluator {
public:
  ExpressionEvaluator(const char * pkg,  const char * iname, const std::string & iexpr);
  ~ExpressionEvaluator();
  
  template<typename EXPR, typename... CArgs>
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

#define SINGLE_ARG(...) __VA_ARGS__
#define RECO_XSTR(...) RECO_STR(__VA_ARGS__)
#define RECO_STR(...) #__VA_ARGS__
#define reco_expressionEvaluator(pkg, EXPR, iexpr) reco::expressionEvaluator<EXPR>(pkg,RECO_XSTR(EXPR),iexpr)

#endif // CommonToolsUtilsExpressionEvaluator_H

