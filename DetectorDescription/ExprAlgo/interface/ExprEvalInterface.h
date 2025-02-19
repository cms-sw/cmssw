#ifndef ExprEvalInterface_h
#define ExprEvalInterface_h

#include <string>
//#include <vector>
//#include <utility>

//! Interface of an Expression Evaluator.
class ExprEvalInterface
{
public:
  //ExprEvalInterface();
  virtual ~ExprEvalInterface();

  //! put a new variable named 'namespace:name' into the dictionary of the evaluator
  virtual 
  void set(const std::string & ns, //< current namespace
           const std::string & name, //< name of variable inside current namespace
           const std::string & valueExpr
	   ) = 0;
  
  //! evaluate an expression expr inside the local namespace
  virtual 
  double eval(const std::string & ns, //< current namespace
              const std::string & expr //< expression to be evaluated inside current namespace
	     ) = 0;
  
  //! check whether a variable is already defined or not
  virtual
  bool isDefined(const std::string & ns, //< current namespace
                 const std::string & name //< name of the variable inside current namespace
		 ) = 0;

  //! access to the dictionary (namespace,name)->value
  /** if not implemented in a sub-class it returns 0 and does nothing to result*/
  //virtual size_t dictionary(std::vector<pair<std::string,std::string>,double> & result) const; 
  
  //! clear the dictionary of the evaluator
  virtual
  void clear() = 0;  
};

#endif
