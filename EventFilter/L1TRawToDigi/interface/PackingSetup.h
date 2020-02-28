#ifndef EventFilter_L1TRawToDigi_PackingSetup_h
#define EventFilter_L1TRawToDigi_PackingSetup_h

#include <map>

#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace edm {
  class ConsumesCollector;
  class Event;
  class ParameterSet;
}  // namespace edm

namespace l1t {
  // Mapping of board id to list of unpackers.  Different for each set of (FED, Firmware) ids.
  typedef std::map<std::pair<int, int>, Packers> PackerMap;
  // Mapping of block id to unpacker.  Different for each set of (FED, Board, AMC, Firmware) ids.
  typedef std::map<int, std::shared_ptr<Unpacker>> UnpackerMap;

  class PackingSetup {
  public:
    PackingSetup(){};
    virtual ~PackingSetup(){};
    virtual std::unique_ptr<PackerTokens> registerConsumes(const edm::ParameterSet&, edm::ConsumesCollector&) = 0;
    virtual void registerProducts(edm::ProducesCollector) = 0;

    // Get a map of (amc #, board id) ↔ list of packing functions for a specific FED, FW combination
    virtual PackerMap getPackers(int fed, unsigned int fw) = 0;

    // Get a map of Block IDs ↔ unpacker for a specific FED, board, AMC, FW combination
    virtual UnpackerMap getUnpackers(int fed, int board, int amc, unsigned int fw) = 0;
    virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event&) = 0;

    // Fill description with needed parameters for the setup, i.e.,
    // special input tags
    virtual void fillDescription(edm::ParameterSetDescription&) = 0;
  };

}  // namespace l1t

#endif
