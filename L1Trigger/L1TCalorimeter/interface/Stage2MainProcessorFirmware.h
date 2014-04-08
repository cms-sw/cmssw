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
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2ClusterAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2EGammaAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2TauAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2JetAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2EtSumAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2JetSumAlgorithm.h"

#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"


namespace l1t {

  // first iteration
  class Stage2MainProcessorFirmwareImp1 : public Stage2MainProcessor {
  public:
    Stage2MainProcessorFirmwareImp1(const FirmwareVersion & fwv, CaloParams* params);

    virtual ~Stage2MainProcessorFirmwareImp1();

    virtual void processEvent(const std::vector<l1t::CaloTower> & inTowers,
			      std::vector<l1t::CaloTower> & outTowers,
			      std::vector<l1t::CaloCluster> & clusters,
			      std::vector<l1t::EGamma> & egammas,
			      std::vector<l1t::Tau> & taus,
			      std::vector<l1t::Jet> & jets,
			      std::vector<l1t::EtSum> & etSums);

    void print(std::ostream&) const;

    friend std::ostream& operator<<(std::ostream& o, const Stage2MainProcessorFirmwareImp1 & p) { p.print(o); return o; }

  private:
    
    FirmwareVersion const & m_fwv;
    CaloParams* m_params;

    Stage2TowerDecompressAlgorithm* m_towerAlgo;
    CaloStage2ClusterAlgorithm* m_egClusterAlgo;
    CaloStage2EGammaAlgorithm* m_egAlgo;
    CaloStage2ClusterAlgorithm* m_tauClusterAlgo;
    CaloStage2TauAlgorithm* m_tauAlgo;
    CaloStage2JetAlgorithm* m_jetAlgo;
    CaloStage2EtSumAlgorithm* m_sumAlgo;
    CaloStage2JetSumAlgorithm* m_jetSumAlgo;
    
  };
  
}

#endif
