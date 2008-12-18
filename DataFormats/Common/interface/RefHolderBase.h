#ifndef DataFormats_Common_RefHolderBase_h
#define DataFormats_Common_RefHolderBase_h
/* \class edm::reftobase::Base
 *
 *
 */
#include "Reflex/Type.h"
#include "FWCore/Utilities/interface/UseReflex.h"

namespace edm {
  class ProductID;
  class EDProductGetter;
  namespace reftobase {

    class RefVectorHolderBase;

    class RefHolderBase {
    public:
      RefHolderBase() { }
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
      virtual EDProductGetter const* productGetter() const = 0;
      virtual bool hasProductCache() const = 0;
      virtual void const * product() const = 0;

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      virtual bool isAvailable() const = 0;

    private:
      // "cast" the real type of the element (the T of contained Ref),
      // and cast it to the type specified by toType, using Reflex.
      // Return 0 if the real type is not toType nor a subclass of
      // toType.
      virtual void const* pointerToType(Reflex::Type const& toType) const = 0;
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
      static Reflex::Type s_type(Reflex::Type::ByTypeInfo(typeid(T)));
      return static_cast<T const*>(pointerToType(s_type));
    }

  }
}

#endif
