#ifndef CommonTools_Utils_ExpressionVar_h
#define CommonTools_Utils_ExpressionVar_h
/* \class reco::parser::ExpressionVar
 *
 * Variable expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.7 $
 *
 */
#include "CommonTools/Utils/src/ExpressionBase.h"
#include "CommonTools/Utils/src/TypeCode.h"
#include "CommonTools/Utils/src/MethodInvoker.h"
#include <vector>

namespace reco {
  namespace parser {
    /// Evaluate an object's method or datamember (or chain of them) to get a number
    struct ExpressionVar : public ExpressionBase {
      ExpressionVar(const std::vector<MethodInvoker> & methods, method::TypeCode retType);

      ~ExpressionVar() ;
      ExpressionVar(const ExpressionVar &var) ;

      virtual double value(const edm::ObjectWithDict & o) const;

      static bool isValidReturnType(method::TypeCode);
      /// performs the needed conversion from void * to double
      /// this method is used also from the ExpressionLazyVar code
      static double objToDouble(const edm::ObjectWithDict &obj, method::TypeCode type) ;

      /// allocate an object to hold the result of a given member (if needed)
      /// this method is used also from the LazyInvoker code
      /// returns true if objects returned from this will require a destructor 
      static bool makeStorage(edm::ObjectWithDict &obj, const edm::TypeWithDict &retType) ;
      /// delete an objecty, if needed
      /// this method is used also from the LazyInvoker code
      static void delStorage(edm::ObjectWithDict &obj);

    private:
      std::vector<MethodInvoker>  methods_;
      mutable std::vector<edm::ObjectWithDict> objects_;
      mutable std::vector<bool>           needsDestructor_;
      method::TypeCode retType_;
      void initObjects_();
    }; 

  /// Same as ExpressionVar but with lazy resolution of object methods
  /// using the final type of the object, and not the one fixed at compile time
  struct ExpressionLazyVar : public ExpressionBase {
      ExpressionLazyVar(const std::vector<LazyInvoker> & methods);
      ~ExpressionLazyVar() ;

      virtual double value(const edm::ObjectWithDict & o) const;

    private:
      std::vector<LazyInvoker> methods_;
      mutable std::vector<edm::ObjectWithDict> objects_;
  }; 

 }
}

#endif
