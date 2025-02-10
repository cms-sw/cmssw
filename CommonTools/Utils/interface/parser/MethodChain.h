#ifndef CommonTools_Utils_MethodChain_h
#define CommonTools_Utils_MethodChain_h

/* \class reco::parser::MethodChain
 *
 * Chain of methods
 * Based on ExpressionBase and ExpressionVar, but remove final conversion to double
 *
 */

#include "CommonTools/Utils/interface/parser/MethodInvoker.h"
#include "CommonTools/Utils/interface/TypeCode.h"

#include <vector>
#include <oneapi/tbb/concurrent_queue.h>

namespace reco {
  namespace parser {

    /// Based on Expression, but its value method returns an edm::ObjectWithDict instead of a double
    class MethodChainBase {
    public:  // Public Methods
      virtual ~MethodChainBase() {}
      virtual edm::ObjectWithDict value(const edm::ObjectWithDict&) const = 0;
    };

    /// Shared ptr to MethodChainBase
    typedef std::shared_ptr<MethodChainBase> MethodChainPtr;

    /// Evaluate an object's method or datamember (or chain of them)
    class MethodChain : public MethodChainBase {
    private:  // Private Data Members
      std::vector<MethodInvoker> methods_;
      using Objects = std::vector<std::pair<edm::ObjectWithDict, bool>>;
      mutable oneapi::tbb::concurrent_queue<Objects> objectsCache_;

    private:  // Private Methods
      [[nodiscard]] Objects initObjects_() const;

      Objects borrowObjects() const;
      void returnObjects(Objects&&) const;

    public:  // Public Static Methods
      /// allocate an object to hold the result of a given member (if needed)
      /// this method is used also from the LazyInvoker code
      /// returns true if objects returned from this will require a destructor
      static bool makeStorage(edm::ObjectWithDict& obj, const edm::TypeWithDict& retType);

      /// delete an objecty, if needed
      /// this method is used also from the LazyInvoker code
      static void delStorage(edm::ObjectWithDict&);

    public:  // Public Methods
      MethodChain(const std::vector<MethodInvoker>& methods);
      MethodChain(const MethodChain&);
      ~MethodChain();
      edm::ObjectWithDict value(const edm::ObjectWithDict&) const override;
    };

    /// Same as MethodChain but with lazy resolution of object methods
    /// using the dynamic type of the object, and not the one fixed at compile time
    class LazyMethodChain : public MethodChainBase {
    private:  // Private Data Members
      std::vector<LazyInvoker> methods_;

    public:
      LazyMethodChain(const std::vector<LazyInvoker>& methods);
      ~LazyMethodChain() override;
      edm::ObjectWithDict value(const edm::ObjectWithDict&) const override;
    };

  }  // namespace parser
}  // namespace reco

#endif  // CommonTools_Utils_MethodChain_h
