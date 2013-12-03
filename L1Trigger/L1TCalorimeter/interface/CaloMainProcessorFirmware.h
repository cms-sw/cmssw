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

#ifndef CaloMainProcessorFirmware_H
#define CaloMainProcessorFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloMainProcessor.h"
#include "CondFormats/L1TCalorimeter/interface/CaloMainProcessorParams.h"

//#include "L1Trigger/L1TYellow/interface/YellowFirmwareFactory.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloMainProcessorFirmwareImp1 : public CaloMainProcessor {
  public:
    CaloMainProcessorFirmware1(const CaloMainProcessorParams & dbPars);
    virtual ~CaloMainProcessorFirmware1();
    virtual void processEvent(const EcalTriggerPrimitiveDigiCollection &,
							  const HcalTriggerPrimitiveCollection &,
 							  BXVector<EGamma> & egammas,
							  BXVector<Tau> & taus,
							  BXVector<Jet> & jets,
							  BXVector<EtSum> & etsums);
  private:
	
    CaloMainProcessorParams const & m_params;

	CaloTowerAlgorithm* m_towerAlgo;
	CaloClusterAlgorithm* m_clusterAlgo;
	CaloEGAlgorithm* m_egAlgo;
	CaloTauAlgoritmh* m_tauAlgo;
	CaloJetAlgorithm* m_jetAlgo;
	CaloSumAlgorithm* m_sumAlgo;
	CaloJetSumAlgorithm* m_jetSumAlgo;

  };
  
}

#endif