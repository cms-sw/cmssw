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

#include "CondFormats/L1TCalorimeter/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage2MainProcessorFirmwareImp1 : public CaloStage2MainProcessor {
  public:
    CaloStage2MainProcessorFirmware1(const CaloParams & dbPars);
    virtual ~CaloStage2MainProcessorFirmware1();
    virtual void processEvent(const l1t::BXVector<l1t::CaloTower> &,
 							  l1t::BXVector<l1t::EGamma> & egammas,
							  l1t::BXVector<l1t::Tau> & taus,
							  l1t::BXVector<l1t::Jet> & jets,
							  l1t::BXVector<l1t::EtSum> & etsums);
  private:
	
    CaloParams const & m_params;

	CaloClusterAlgorithm* m_clusterAlgo;
	CaloEGAlgorithm* m_egAlgo;
	CaloTauAlgoritmh* m_tauAlgo;
	CaloJetAlgorithm* m_jetAlgo;
	CaloSumAlgorithm* m_sumAlgo;
	CaloJetSumAlgorithm* m_jetSumAlgo;
	
	l1t::BXVector<l1t::CaloCluster> m_clusters;
	l1t::BXVector<l1t::EGamma> m_egammas;
	l1t::BXVector<l1t::Tau> m_taus;
	l1t::BXVector<l1t::Jet> m_jets;
	l1t::BXVector<l1t::EtSums> m_etsums;
	l1t::BXVector<l1t::EtSums> m_jetsums;

  };
  
}

#endif