#ifndef EventFilter_RPCRawToDigi_RPCAMCUnpacker_h
#define EventFilter_RPCRawToDigi_RPCAMCUnpacker_h

#include <map>
#include <vector>

#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"
#include "EventFilter/RPCRawToDigi/interface/RPCAMC13Record.h"
#include "FWCore/Framework/interface/ProducesCollector.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
  class ParameterSetDescription;
  class Run;
}  // namespace edm

class RPCAMCUnpacker {
public:
  RPCAMCUnpacker(edm::ParameterSet const&, edm::ProducesCollector);
  virtual ~RPCAMCUnpacker();

  static void fillDescription(edm::ParameterSetDescription& desc);

  virtual void beginRun(edm::Run const& run, edm::EventSetup const& setup);
  virtual void produce(edm::Event& event,
                       edm::EventSetup const& setup,
                       std::map<RPCAMCLink, rpcamc13::AMCPayload> const& amc_payload);

  std::vector<int> const& getFeds() const;

protected:
  std::vector<int> feds_;
};

inline std::vector<int> const& RPCAMCUnpacker::getFeds() const { return feds_; }

#endif  // EventFilter_RPCRawToDigi_RPCAMCUnpacker_h
