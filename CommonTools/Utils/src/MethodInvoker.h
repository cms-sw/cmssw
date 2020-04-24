#ifndef CommonTools_Utils_MethodInvoker_h
#define CommonTools_Utils_MethodInvoker_h

#include "CommonTools/Utils/src/AnyMethodArgument.h"
#include "CommonTools/Utils/src/TypeCode.h"
#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

#include <map>
#include <vector>

#include "tbb/concurrent_unordered_map.h"

namespace edm {
  struct TypeIDHasher {
    size_t operator()(TypeID const& tid) const {
      tbb::tbb_hash<std::string> hasher;
      return hasher(std::string(tid.name()));
    }
  };
}

namespace reco {
namespace parser {

class MethodInvoker {
private: // Private Data Members
  edm::FunctionWithDict method_;
  edm::MemberWithDict member_;
  std::vector<AnyMethodArgument> ints_; // already fixed to the correct type
  std::vector<void*> args_;

  bool isFunction_;
  edm::TypeWithDict retTypeFinal_;
private: // Private Function Members
  void setArgs();
public: // Public Function Members
  explicit MethodInvoker(const edm::FunctionWithDict& method,
                         const std::vector<AnyMethodArgument>& ints =
                           std::vector<AnyMethodArgument>());
  explicit MethodInvoker(const edm::MemberWithDict&);
  MethodInvoker(const MethodInvoker&);
  MethodInvoker& operator=(const MethodInvoker&);

  edm::FunctionWithDict const method() const { return method_; }
  edm::MemberWithDict const member() const { return member_; }
  bool isFunction() const { return isFunction_; }
  std::string methodName() const;
  std::string returnTypeName() const;

  /// Invokes the method, putting the result in retval.
  /// Returns the Object that points to the result value,
  /// after removing any "*" and "&"
  /// Caller code is responsible for allocating retstore
  /// before calling 'invoke', and of deallocating it afterwards
  edm::ObjectWithDict invoke(const edm::ObjectWithDict& obj,
                             edm::ObjectWithDict& retstore) const;
};

/// A bigger brother of the MethodInvoker:
/// - it owns also the object in which to store the result
/// - it handles by itself the popping out of Refs and Ptrs
/// in this way, it can map 1-1 to a name and set of args
struct SingleInvoker : boost::noncopyable {
private: // Private Data Members
  method::TypeCode retType_;
  std::vector<MethodInvoker> invokers_;
  mutable edm::ObjectWithDict storage_;
  bool storageNeedsDestructor_;
  /// true if this invoker just pops out a ref and returns (ref.get(), false)
  bool isRefGet_;
public:
  SingleInvoker(const edm::TypeWithDict&, const std::string& name,
                const std::vector<AnyMethodArgument>& args);
  ~SingleInvoker();

  /// If the member is found in object o, evaluate and
  /// return (value,true)
  /// If the member is not found but o is a Ref/RefToBase/Ptr,
  /// (return o.get(), false)
  /// the actual edm::ObjectWithDict where the result is stored
  /// will be pushed in vector so that, if needed, its destructor
  /// can be called
  std::pair<edm::ObjectWithDict, bool>
  invoke(const edm::ObjectWithDict& o,
         std::vector<edm::ObjectWithDict>& v) const;

  /// convert the output of invoke to a double, if possible
  double retToDouble(const edm::ObjectWithDict&) const;

  void throwFailedConversion(const edm::ObjectWithDict&) const;
};


/// Keeps different SingleInvokers for each dynamic type of the objects passed to invoke()
struct LazyInvoker {
  typedef std::shared_ptr<SingleInvoker> SingleInvokerPtr;
  typedef tbb::concurrent_unordered_map<edm::TypeID, SingleInvokerPtr,edm::TypeIDHasher> InvokerMap;
private: // Private Data Members
  std::string name_;
  std::vector<AnyMethodArgument> argsBeforeFixups_;
  // the shared ptr is only to make the code exception safe
  // otherwise I think it could leak if the constructor of
  // SingleInvoker throws an exception (which can happen) 
  mutable InvokerMap invokers_;
private: // Private Function Members
  const SingleInvoker& invoker(const edm::TypeWithDict&) const;
public: // Public Function Members
  explicit LazyInvoker(const std::string& name,
                       const std::vector<AnyMethodArgument>& args);
  ~LazyInvoker();

  /// invoke method, returns object that points to result
  /// (after stripping '*' and '&')
  /// the object is still owned by the LazyInvoker
  /// the actual edm::ObjectWithDict where the result is
  /// stored will be pushed in vector
  /// so that, if needed, its destructor can be called
  edm::ObjectWithDict invoke(const edm::ObjectWithDict& o,
                             std::vector<edm::ObjectWithDict>& v) const;

  /// invoke and coerce result to double
  double invokeLast(const edm::ObjectWithDict& o,
                    std::vector<edm::ObjectWithDict>& v) const;
};

} // namesapce parser
} // namespace reco

#endif // CommonTools_Utils_MethodInvoker_h
