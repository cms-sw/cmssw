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
#include <oneapi/tbb/concurrent_queue.h>

namespace reco {
  namespace parser {

    /// Evaluate an object's method or datamember (or chain of them) to get a number
    class ExpressionVar : public ExpressionBase {
    private:  // Private Data Members
      std::vector<MethodInvoker> methods_;
      using Objects = std::vector<std::pair<edm::ObjectWithDict, bool>>;
      mutable oneapi::tbb::concurrent_queue<Objects> objectsCache_;
      method::TypeCode retType_;

    private:  // Private Methods
      [[nodiscard]] Objects initObjects_() const;

      Objects borrowObjects() const;
      void returnObjects(Objects&&) const;

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

    public:
      ExpressionLazyVar(const std::vector<LazyInvoker>& methods);
      ~ExpressionLazyVar() override;
      double value(const edm::ObjectWithDict&) const override;
    };

  }  // namespace parser
}  // namespace reco

#endif  // CommonTools_Utils_ExpressionVar_h
