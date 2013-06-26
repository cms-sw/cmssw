#ifndef CommonTools_Utils_MethodInvoker_h
#define CommonTools_Utils_MethodInvoker_h
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "CommonTools/Utils/src/AnyMethodArgument.h"
#include "CommonTools/Utils/src/TypeCode.h"
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <map>

namespace reco {
  namespace parser {

    struct MethodInvoker {
      explicit MethodInvoker(const edm::FunctionWithDict & method,
			     const std::vector<AnyMethodArgument>    & ints   = std::vector<AnyMethodArgument>() );
      explicit MethodInvoker(const edm::MemberWithDict & member);
      MethodInvoker(const MethodInvoker &); 

      /// Invokes the method, putting the result in retval.
      /// Returns the Object that points to the result value, after removing any "*" and "&" 
      /// Caller code is responsible for allocating retstore before calling 'invoke', and of deallocating it afterwards
      edm::ObjectWithDict
      invoke(const edm::ObjectWithDict & o, edm::ObjectWithDict &retstore) const;
      edm::FunctionWithDict const method() const {return method_;}
      edm::MemberWithDict const member() const {return member_;}
      MethodInvoker & operator=(const MethodInvoker &);
      bool isFunction() const {return isFunction_;}
      std::string methodName() const;
      std::string returnTypeName() const;
    private:
      edm::FunctionWithDict method_;
      edm::MemberWithDict member_;
      std::vector<AnyMethodArgument> ints_; // already fixed to the correct type
      std::vector<void*> args_;
      bool isFunction_;
      void setArgs();
    };

    /// A bigger brother of the MethodInvoker:
    /// - it owns also the object in which to store the result
    /// - it handles by itself the popping out of Refs and Ptrs
    /// in this way, it can map 1-1 to a name and set of args
    struct SingleInvoker : boost::noncopyable {
        SingleInvoker(const edm::TypeWithDict &t,
                      const std::string &name, 
                      const std::vector<AnyMethodArgument> &args)  ;
        ~SingleInvoker();

        /// If the member is found in object o, evaluate and return (value,true)
        /// If the member is not found but o is a Ref/RefToBase/Ptr, (return o.get(), false)
        /// the actual edm::ObjectWithDict where the result is stored will be pushed in vector
        /// so that, if needed, its destructor can be called
        std::pair<edm::ObjectWithDict,bool> invoke(const edm::ObjectWithDict & o, std::vector<edm::ObjectWithDict> &v) const;
        
        // convert the output of invoke to a double, if possible
        double retToDouble(const edm::ObjectWithDict & o) const;
        void   throwFailedConversion(const edm::ObjectWithDict & o) const;
      private:
        method::TypeCode            retType_;
        std::vector<MethodInvoker>  invokers_;
        mutable edm::ObjectWithDict      storage_;
        bool                        storageNeedsDestructor_;
        /// true if this invoker just pops out a ref and returns (ref.get(), false)
        bool isRefGet_;
    };


    /// Keeps different SingleInvokers for each dynamic type of the objects passed to invoke()
    struct LazyInvoker {
      explicit LazyInvoker(const std::string &name,
			   const std::vector<AnyMethodArgument> &args);
      ~LazyInvoker();
      /// invoke method, returns object that points to result (after stripping '*' and '&')
      /// the object is still owned by the LazyInvoker
      /// the actual edm::ObjectWithDict where the result is stored will be pushed in vector
      /// so that, if needed, its destructor can be called
      edm::ObjectWithDict invoke(const edm::ObjectWithDict & o, std::vector<edm::ObjectWithDict> &v) const;
      /// invoke and coerce result to double
      double invokeLast(const edm::ObjectWithDict & o, std::vector<edm::ObjectWithDict> &v) const;
    private:
      std::string name_;
      std::vector<AnyMethodArgument> argsBeforeFixups_;
      typedef boost::shared_ptr<SingleInvoker> SingleInvokerPtr; // the shared ptr is only to make the code exception safe
      mutable std::map<edm::TypeID, SingleInvokerPtr> invokers_;        // otherwise I think it could leak if the constructor of
      const SingleInvoker & invoker(const edm::TypeWithDict &t) const ; // SingleInvoker throws an exception (which can happen)
    };

  }
}

#endif
