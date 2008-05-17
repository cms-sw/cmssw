#ifndef GeneratorInterface_LHEInterface_LHEProxy_h
#define GeneratorInterface_LHEInterface_LHEProxy_h

#include <boost/shared_ptr.hpp>

namespace lhef {

// forward declarations
class LHEEvent;
class LHERunInfo;

class LHEProxy {
    public:
	typedef unsigned long ProxyID;

	~LHEProxy();

	const boost::shared_ptr<LHERunInfo> &getRunInfo() const
	{ return runInfo; }
	const boost::shared_ptr<LHEEvent> &getEvent() const
	{ return event; }

	boost::shared_ptr<LHERunInfo> releaseRunInfo()
	{
		boost::shared_ptr<LHERunInfo> result(runInfo);
		runInfo.reset();
		return result;
	}
	boost::shared_ptr<LHEEvent> releaseEvent()
	{
		boost::shared_ptr<LHEEvent> result(event);
		event.reset();
		return result;
	}

	void clearRunInfo() { runInfo.reset(); }
	void clearEvent() { event.reset(); }

	void loadRunInfo(const boost::shared_ptr<LHERunInfo> &runInfo)
	{ this->runInfo = runInfo; }
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

	boost::shared_ptr<LHERunInfo>	runInfo;
	boost::shared_ptr<LHEEvent>	event;
};

} // namespace lhef

#endif // GeneratorProxy_LHEInterface_LHEProxy_h
