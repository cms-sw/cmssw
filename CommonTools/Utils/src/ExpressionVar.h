#ifndef CommonTools_Utils_ExpressionVar_h
#define CommonTools_Utils_ExpressionVar_h

/* \class reco::parser::ExpressionVar
 *
 * Variable expression
 *
 * \author original version: Chris Jones, Cornell,
 *         adapted by Luca Lista, INFN
 *
 */

#include "CommonTools/Utils/interface/parser/ExpressionBase.h"
#include "CommonTools/Utils/interface/parser/MethodInvoker.h"
#include "CommonTools/Utils/interface/TypeCode.h"

#include <vector>

namespace reco {
  namespace parser {

    /// Evaluate an object's method or datamember (or chain of them) to get a number
    class ExpressionVar : public ExpressionBase {
    private:  // Private Data Members
      std::vector<MethodInvoker> methods_;
      mutable std::vector<edm::ObjectWithDict> objects_;
      mutable std::vector<bool> needsDestructor_;
      method::TypeCode retType_;

    private:  // Private Methods
      void initObjects_();

    public:  // Public Static Methods
      static bool isValidReturnType(method::TypeCode);

      /// performs the needed conversion from void* to double
      /// this method is used also from the ExpressionLazyVar code
      static double objToDouble(const edm::ObjectWithDict& obj, method::TypeCode type);

      /// allocate an object to hold the result of a given member (if needed)
      /// this method is used also from the LazyInvoker code
      /// returns true if objects returned from this will require a destructor
      static bool makeStorage(edm::ObjectWithDict& obj, const edm::TypeWithDict& retType);

      /// delete an objecty, if needed
      /// this method is used also from the LazyInvoker code
      static void delStorage(edm::ObjectWithDict&);

    public:  // Public Methods
      ExpressionVar(const std::vector<MethodInvoker>& methods, method::TypeCode retType);
      ExpressionVar(const ExpressionVar&);
      ~ExpressionVar() override;
      double value(const edm::ObjectWithDict&) const override;
    };

    /// Same as ExpressionVar but with lazy resolution of object methods
    /// using the dynamic type of the object, and not the one fixed at compile time
    class ExpressionLazyVar : public ExpressionBase {
    private:  // Private Data Members
      std::vector<LazyInvoker> methods_;
      mutable std::vector<edm::ObjectWithDict> objects_;

    public:
      ExpressionLazyVar(const std::vector<LazyInvoker>& methods);
      ~ExpressionLazyVar() override;
      double value(const edm::ObjectWithDict&) const override;
    };

  }  // namespace parser
}  // namespace reco

#endif  // CommonTools_Utils_ExpressionVar_h
