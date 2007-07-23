#ifndef Common_RefHolderBase_h
#define Common_RefHolderBase_h
/* \class edm::reftobase::Base
 *
 * $Id: RefHolderBase.h,v 1.2 2007/07/12 12:08:57 llista Exp $
 *
 */
#include "Reflex/Type.h"

namespace edm {
  namespace reftobase {
    using ROOT::Reflex::Type;

    class RefVectorHolderBase;

    class RefHolderBase {
    public:
      template <class T> T const* getPtr() const;
      virtual ~RefHolderBase();
      virtual RefHolderBase* clone() const = 0;

      virtual ProductID id() const = 0;
      virtual size_t key() const = 0;

      // Check to see if the Ref hidden in 'rhs' is equal to the Ref
      // hidden in 'this'. They can not be equal if they are of
      // different types.
      virtual bool isEqualTo(RefHolderBase const& rhs) const = 0;

      // If the type of Ref I contain matches the type contained in
      // 'fillme', set the Ref in 'fillme' equal to mine and return
      // true. If not, write the name of the type I really contain to
      // msg, and return false.

      virtual bool fillRefIfMyTypeMatches(RefHolderBase& ref,
					  std::string& msg) const = 0;

      virtual std::auto_ptr<RefVectorHolderBase> makeVectorHolder() const = 0;

    private:
      // "cast" the real type of the element (the T of contained Ref),
      // and cast it to the type specified by toType, using Reflex.
      // Return 0 if the real type is not toType nor a subclass of
      // toType.
      virtual void const* pointerToType(Type const& toType) const = 0;
    };

    //------------------------------------------------------------------
    // Implementation of RefHolderBase
    //------------------------------------------------------------------

    inline
    RefHolderBase::~RefHolderBase()
    { }

    template <class T>
    T const*
    RefHolderBase::getPtr() const
    {
      static Type s_type(Type::ByTypeInfo(typeid(T)));
      return static_cast<T const*>(pointerToType(s_type));
    }

  }
}

#endif
