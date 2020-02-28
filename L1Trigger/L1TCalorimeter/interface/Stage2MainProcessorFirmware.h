///
/// Description: Firmware headers
///
/// Implementation:
///    Concrete firmware implementations
///
/// \author: Jim Brooke - University of Bristol
///

//
//

#ifndef Stage2MainProcessorFirmware_H
#define Stage2MainProcessorFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2MainProcessor.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2TowerDecompressAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2ClusterAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2EGammaAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2TauAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2JetAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2EtSumAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2JetSumAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxEGAlgoFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxTauAlgoFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxJetAlgoFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxSumsAlgoFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include <memory>

namespace l1t {

  // first iteration
  class Stage2MainProcessorFirmwareImp1 : public Stage2MainProcessor {
  public:
    Stage2MainProcessorFirmwareImp1(unsigned fwv, CaloParamsHelper const* params);

    ~Stage2MainProcessorFirmwareImp1() override = default;

    void processEvent(const std::vector<l1t::CaloTower>& inTowers,
                      std::vector<l1t::CaloTower>& outTowers,
                      std::vector<l1t::CaloCluster>& clusters,
                      std::vector<l1t::EGamma>& mpEGammas,
                      std::vector<l1t::Tau>& mpTaus,
                      std::vector<l1t::Jet>& mpJets,
                      std::vector<l1t::EtSum>& mpSums,
                      std::vector<l1t::EGamma>& egammas,
                      std::vector<l1t::Tau>& taus,
                      std::vector<l1t::Jet>& jets,
                      std::vector<l1t::EtSum>& etSums) override;

    void print(std::ostream&) const;

    friend std::ostream& operator<<(std::ostream& o, const Stage2MainProcessorFirmwareImp1& p) {
      p.print(o);
      return o;
    }

  private:
    std::unique_ptr<Stage2TowerDecompressAlgorithm> m_towerAlgo;
    std::unique_ptr<Stage2Layer2ClusterAlgorithm> m_egClusterAlgo;
    std::unique_ptr<Stage2Layer2EGammaAlgorithm> m_egAlgo;
    std::unique_ptr<Stage2Layer2ClusterAlgorithm> m_tauClusterAlgo;
    std::unique_ptr<Stage2Layer2TauAlgorithm> m_tauAlgo;
    std::unique_ptr<Stage2Layer2JetAlgorithm> m_jetAlgo;
    std::unique_ptr<Stage2Layer2EtSumAlgorithm> m_sumAlgo;
    std::unique_ptr<Stage2Layer2JetSumAlgorithm> m_jetSumAlgo;

    std::unique_ptr<Stage2Layer2DemuxEGAlgo> m_demuxEGAlgo;
    std::unique_ptr<Stage2Layer2DemuxTauAlgo> m_demuxTauAlgo;
    std::unique_ptr<Stage2Layer2DemuxJetAlgo> m_demuxJetAlgo;
    std::unique_ptr<Stage2Layer2DemuxSumsAlgo> m_demuxSumsAlgo;
  };

}  // namespace l1t

#endif
