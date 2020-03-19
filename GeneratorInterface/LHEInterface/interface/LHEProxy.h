#ifndef GeneratorInterface_LHEInterface_LHEProxy_h
#define GeneratorInterface_LHEInterface_LHEProxy_h

#include <memory>

namespace lhef {

  // forward declarations
  class LHEEvent;
  class LHERunInfo;

  class LHEProxy {
  public:
    typedef unsigned long ProxyID;

    ~LHEProxy();

    const std::shared_ptr<LHERunInfo> &getRunInfo() const { return runInfo; }
    const std::shared_ptr<LHEEvent> &getEvent() const { return event; }

    std::shared_ptr<LHERunInfo> releaseRunInfo() {
      std::shared_ptr<LHERunInfo> result(runInfo);
      runInfo.reset();
      return result;
    }
    std::shared_ptr<LHEEvent> releaseEvent() {
      std::shared_ptr<LHEEvent> result(event);
      event.reset();
      return result;
    }

    void clearRunInfo() { runInfo.reset(); }
    void clearEvent() { event.reset(); }

    void loadRunInfo(const std::shared_ptr<LHERunInfo> &runInfo) { this->runInfo = runInfo; }
    void loadEvent(const std::shared_ptr<LHEEvent> &event) { this->event = event; }

    ProxyID getID() const { return id; }

    static std::shared_ptr<LHEProxy> create();
    static std::shared_ptr<LHEProxy> find(ProxyID id);

  private:
    LHEProxy(ProxyID id);

    // not allowed and not implemented
    LHEProxy(const LHEProxy &orig) = delete;
    LHEProxy &operator=(const LHEProxy &orig) = delete;

    const ProxyID id;

    std::shared_ptr<LHERunInfo> runInfo;
    std::shared_ptr<LHEEvent> event;
  };

}  // namespace lhef

#endif  // GeneratorProxy_LHEInterface_LHEProxy_h
