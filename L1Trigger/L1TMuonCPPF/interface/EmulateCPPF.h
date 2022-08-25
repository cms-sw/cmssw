#ifndef L1Trigger_L1TMuonCPPF_EmulateCPPF_h
#define L1Trigger_L1TMuonCPPF_EmulateCPPF_h

#include "L1Trigger/L1TMuonCPPF/interface/RecHitProcessor.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

class EmulateCPPF {
public:
  explicit EmulateCPPF(const edm::ParameterSet &iConfig, edm::ConsumesCollector &&iConsumes);
  ~EmulateCPPF();

  void process(
      // Input
      const edm::Event &iEvent,
      const edm::EventSetup &iSetup,
      // Output
      l1t::CPPFDigiCollection &cppf_recHit);

private:
  // For now, treat CPPF as single board
  // In the future, may want to treat the 4 CPPF boards in each endcap as
  // separate entities
  std::array<RecHitProcessor, 1> recHit_processors_;

  const edm::EDGetToken rpcDigiToken_;
  const edm::EDGetToken recHitToken_;
  const edm::EDGetToken rpcDigiSimLinkToken_;
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeomToken_;

  enum class CppfSource { File, EventSetup } cppfSource_;
  std::vector<RecHitProcessor::CppfItem> CppfVec_1;
  int MaxClusterSize_;
};  // End class EmulateCPPF

#endif  // #define L1Trigger_L1TMuonCPPF_EmulateCPPF_h
