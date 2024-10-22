#ifndef L1Trigger_L1TMuonEndCapPhase2_EMTFfwd_h
#define L1Trigger_L1TMuonEndCapPhase2_EMTFfwd_h

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitiveFwd.h"

namespace emtf::phase2 {
  class EMTFContext;
  class EMTFConfiguration;
  class EMTFModel;

  struct TPInfo;
  class TPEntry;
  class TrackFinder;
  class SectorProcessor;
  class TPCollector;
  class CSCTPCollector;
  class RPCTPCollector;
  class GEMTPCollector;
  class ME0TPCollector;
  class TPSelector;
  class CSCTPSelector;
  class RPCTPSelector;
  class GEMTPSelector;
  class ME0TPSelector;
  class TPConverter;
}  // namespace emtf::phase2

#endif  // namespace L1Trigger_L1TMuonEndCapPhase2_EMTFfwd_h
