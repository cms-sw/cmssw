#ifndef GeneratorInterface_Herwig7Interface_Proxy_h
#define GeneratorInterface_Herwig7Interface_Proxy_h
#include <memory>

namespace ThePEG {

  // forward declarations
  class LHEEvent;
  class LHERunInfo;

  template <class T>
  class Proxy;

  class ProxyBase {
  public:
    typedef unsigned long ProxyID;

    virtual ~ProxyBase();

    ProxyID getID() const { return id; }

  private:
    typedef ProxyBase *(*ctor_t)(ProxyID id);

    template <class T>
    friend class Proxy;

    ProxyBase(ProxyID id);

    static std::shared_ptr<ProxyBase> create(ctor_t ctor);
    static std::shared_ptr<ProxyBase> find(ProxyID id);

    // not allowed and not implemented
    ProxyBase(const ProxyBase &orig) = delete;
    ProxyBase &operator=(const ProxyBase &orig) = delete;

    const ProxyID id;
  };

  template <class T>
  class Proxy : public ProxyBase {
  public:
    typedef Proxy Base;

    static inline std::shared_ptr<T> create() { return std::static_pointer_cast<T>(ProxyBase::create(&Proxy::ctor)); }
    static inline std::shared_ptr<T> find(ProxyID id) { return std::dynamic_pointer_cast<T>(ProxyBase::find(id)); }

  protected:
    inline Proxy(ProxyID id) : ProxyBase(id) {}

  private:
    static ProxyBase *ctor(ProxyID id) { return new T(id); }
  };

}  // namespace ThePEG

#endif  // GeneratorProxy_Herwig7Interface_Proxy_h
