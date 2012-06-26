#ifndef CommonTools_Utils_MethodInvoker_h
#define CommonTools_Utils_MethodInvoker_h
#include "Reflex/Object.h"
#include "Reflex/Member.h"
#include "CommonTools/Utils/src/AnyMethodArgument.h"
#include "CommonTools/Utils/src/TypeCode.h"
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <map>

namespace reco {
  namespace parser {

    struct MethodInvoker {
      explicit MethodInvoker(const Reflex::Member & method,
			     const std::vector<AnyMethodArgument>    & ints   = std::vector<AnyMethodArgument>() );
      MethodInvoker(const MethodInvoker &); 

      /// Invokes the method, putting the result in retval.
      /// Returns the Object that points to the result value, after removing any "*" and "&" 
      /// Caller code is responsible for allocating retstore before calling 'invoke', and of deallocating it afterwards
      Reflex::Object
      invoke(const Reflex::Object & o, Reflex::Object &retstore) const;
      const Reflex::Member & method() const { return method_; }
      MethodInvoker & operator=(const MethodInvoker &);
    private:
      Reflex::Member method_;
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
        SingleInvoker(const Reflex::Type &t,
                      const std::string &name, 
                      const std::vector<AnyMethodArgument> &args)  ;
        ~SingleInvoker();

        /// If the member is found in object o, evaluate and return (value,true)
        /// If the member is not found but o is a Ref/RefToBase/Ptr, (return o.get(), false)
        /// the actual Reflex::Object where the result is stored will be pushed in vector
        /// so that, if needed, its destructor can be called
        std::pair<Reflex::Object,bool> invoke(const Reflex::Object & o, std::vector<Reflex::Object> &v) const;
        
        // convert the output of invoke to a double, if possible
        double retToDouble(const Reflex::Object & o) const;
        void   throwFailedConversion(const Reflex::Object & o) const;
      private:
        method::TypeCode            retType_;
        std::vector<MethodInvoker>  invokers_;
        mutable Reflex::Object      storage_;
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
      /// the actual Reflex::Object where the result is stored will be pushed in vector
      /// so that, if needed, its destructor can be called
      Reflex::Object invoke(const Reflex::Object & o, std::vector<Reflex::Object> &v) const;
      /// invoke and coerce result to double
      double invokeLast(const Reflex::Object & o, std::vector<Reflex::Object> &v) const;
    private:
      std::string name_;
      std::vector<AnyMethodArgument> argsBeforeFixups_;
      typedef boost::shared_ptr<SingleInvoker> SingleInvokerPtr; // the shared ptr is only to make the code exception safe
      mutable std::map<void *, SingleInvokerPtr> invokers_;        // otherwise I think it could leak if the constructor of
      const SingleInvoker & invoker(const Reflex::Type &t) const ; // SingleInvoker throws an exception (which can happen)
    };

  }
}

#endif
