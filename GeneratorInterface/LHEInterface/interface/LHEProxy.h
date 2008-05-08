#ifndef GeneratorInterface_LHEInterface_LHEProxy_h
#define GeneratorInterface_LHEInterface_LHEProxy_h

#include <boost/shared_ptr.hpp>

namespace lhef {

// forward declarations
class LHEEvent;
class LHECommon;

class LHEProxy {
    public:
	typedef unsigned long ProxyID;

	~LHEProxy();

	const boost::shared_ptr<LHECommon> &getCommon() const { return common; }
	const boost::shared_ptr<LHEEvent> &getEvent() const { return event; }

	void loadCommon(const boost::shared_ptr<LHECommon> &common)
	{ this->common = common; }
	void loadEvent(const boost::shared_ptr<LHEEvent> &event)
	{ this->event = event; }

	ProxyID getID() const { return id; }

	static boost::shared_ptr<LHEProxy> create();
	static boost::shared_ptr<LHEProxy> find(ProxyID id);

    private:
	LHEProxy(ProxyID id);

	// not allowed and not implemented
	LHEProxy(const LHEProxy &orig);
	LHEProxy &operator = (const LHEProxy &orig);

	const ProxyID			id;

	boost::shared_ptr<LHECommon>	common;
	boost::shared_ptr<LHEEvent>	event;
};

} // namespace lhef

#endif // GeneratorProxy_LHEInterface_LHEProxy_h
