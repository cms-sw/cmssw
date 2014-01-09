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

#ifndef CaloStage2MainProcessorFirmware_H
#define CaloStage2MainProcessorFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloStage2MainProcessor.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2ClusterAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2EGammaAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2TauAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2JetAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2EtSumAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2JetSumAlgorithm.h"

//#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"


namespace l1t {

  // first iteration
  class CaloStage2MainProcessorFirmwareImp1 : public CaloStage2MainProcessor {
  public:
    CaloStage2MainProcessorFirmwareImp1(const FirmwareVersion & fwv ); //const CaloParams & dbPars);
    virtual ~CaloStage2MainProcessorFirmwareImp1();
    virtual void processEvent(const std::vector<l1t::CaloTower> &,
			      std::vector<l1t::EGamma> & egammas,
			      std::vector<l1t::Tau> & taus,
			      std::vector<l1t::Jet> & jets,
			      std::vector<l1t::EtSum> & etsums);
  private:
    
    //    CaloParams const & m_params;
    FirmwareVersion const & m_fwv;
 
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
