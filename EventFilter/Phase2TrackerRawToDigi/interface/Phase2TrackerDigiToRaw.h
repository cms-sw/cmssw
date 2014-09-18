#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDHeader.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

namespace Phase2Tracker
{
  class Phase2TrackerDigiToRaw
  {
    public:
      Phase2TrackerDigiToRaw() {}
      Phase2TrackerDigiToRaw(FEDDAQHeader, FEDDAQTrailer, Phase2TrackerFEDHeader, int);
      ~Phase2TrackerDigiToRaw() {}
      void buildPayload(edm::Event&, const TrackerTopology *&, const Phase2TrackerCabling*&, edm::Handle< edmNew::DetSetVector<SiPixelCluster> >&);
      uint8_t* buildBuffer();
    private:
      FEDDAQHeader  daq_header_;
      FEDDAQTrailer daq_trailer_;
      Phase2TrackerFEDHeader     fed_header_;
      int mode_;
      uint8_t* payload_;
      uint8_t* buffer_;
      int payload_size_;
  };
}
