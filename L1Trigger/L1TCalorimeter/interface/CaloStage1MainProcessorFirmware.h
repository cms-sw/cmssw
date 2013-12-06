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
#include "CondFormats/L1TCalorimeter/interface/CaloStage1MainProcessorParams.h"

//#include "L1Trigger/L1TYellow/interface/YellowFirmwareFactory.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage1MainProcessorFirmwareImp1 : public CaloStage1MainProcessor {
  public:
    CaloStage1MainProcessorFirmware1(const CaloStage1MainProcessorParams & dbPars);
    virtual ~CaloStage1MainProcessorFirmware1();
    virtual void processEvent(const EcalTriggerPrimitiveDigiCollection &,
			      const HcalTriggerPrimitiveCollection &,
			      BXVector<EGamma> & egammas,
			      BXVector<Tau> & taus,
			      BXVector<Jet> & jets,
			      BXVector<EtSum> & etsums);
  private:

    CaloStage1MainProcessorParams const & m_db;

    CaloStage1TowerAlgorithm* m_towerAlgo;
    CaloStage1ClusterAlgorithm* m_clusterAlgo;
    CaloStage1EGAlgorithm* m_egAlgo;
    CaloStage1TauAlgoritmh* m_tauAlgo;
    CaloStage1JetAlgorithm* m_jetAlgo;
    CaloStage1SumAlgorithm* m_sumAlgo;
    CaloStage1JetSumAlgorithm* m_jetSumAlgo;

  };

}

#endif
