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

#ifndef CaloStage1MainProcessorFirmware_H
#define CaloStage1MainProcessorFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1MainProcessor.h"
#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"
#include "CaloStage1JetAlgorithm.h"

namespace l1t {

  class CaloStage1MainProcessorFirmwareImp1 : public CaloStage1MainProcessor {
  public:
    CaloStage1MainProcessorFirmwareImp1(const FirmwareVersion & fwv /*const CaloParams & dbPars*/);
    virtual ~CaloStage1MainProcessorFirmwareImp1();
    virtual void processEvent(const std::vector<CaloEmCand> &,
                              const std::vector<CaloRegion> &,
			      std::vector<EGamma> & egammas,
			      std::vector<Tau> & taus,
			      std::vector<Jet> & jets,
			      std::vector<EtSum> & etsums);
  private:

    //CaloParams const & m_db;
    FirmwareVersion const & m_fwv;

    //CaloStage1EGammaAlgorithm* m_egAlgo;
    //CaloStage1TauAlgoritmh* m_tauAlgo;
    CaloStage1JetAlgorithm* m_jetAlgo;
    //CaloStage1EtSumAlgorithm* m_sumAlgo;

    //CaloStage1JetSumAlgorithm* m_jetSumAlgo;
    //CaloStage1TowerAlgorithm* m_towerAlgo;
    //CaloStage1ClusterAlgorithm* m_clusterAlgo;

  };

}

#endif
