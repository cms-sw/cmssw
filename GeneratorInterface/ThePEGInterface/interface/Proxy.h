#ifndef GeneratorInterface_ThePEGInterface_Proxy_h
#define GeneratorInterface_ThePEGInterface_Proxy_h

#include <boost/shared_ptr.hpp>

namespace ThePEG {

// forward declarations
class LHEEvent;
class LHERunInfo;

template<class T> class Proxy;

class ProxyBase {
    public:
	typedef unsigned long ProxyID;

	virtual ~ProxyBase();

	ProxyID getID() const { return id; }

    private:
	typedef ProxyBase *(*ctor_t)(ProxyID id);

	template<class T> friend class Proxy;

	ProxyBase(ProxyID id);

	static boost::shared_ptr<ProxyBase> create(ctor_t ctor);
	static boost::shared_ptr<ProxyBase> find(ProxyID id);

	// not allowed and not implemented
	ProxyBase(const ProxyBase &orig);
	ProxyBase &operator = (const ProxyBase &orig);

	const ProxyID	id;
};

template<class T>
class Proxy : public ProxyBase {
    public:
	typedef Proxy Base;

	static inline boost::shared_ptr<T> create()
	{ return boost::static_pointer_cast<T>(ProxyBase::create(&Proxy::ctor)); }
	static inline boost::shared_ptr<T> find(ProxyID id)
	{ return boost::dynamic_pointer_cast<T>(ProxyBase::find(id)); }

    protected:
	inline Proxy(ProxyID id) : ProxyBase(id) {}

    private:
	static ProxyBase *ctor(ProxyID id) { return new T(id); }
};

} // namespace ThePEG

#endif // GeneratorProxy_ThePEGInterface_Proxy_h
