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

#ifndef Stage1Layer2MainProcessorFirmware_H
#define Stage1Layer2MainProcessorFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2MainProcessor.h"
//#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsStage1.h"


#include "Stage1Layer2EGammaAlgorithm.h"
#include "Stage1Layer2EtSumAlgorithm.h"
#include "Stage1Layer2JetAlgorithm.h"
#include "Stage1Layer2TauAlgorithm.h"
#include "Stage1Layer2HFRingSumAlgorithm.h"
#include "Stage1Layer2HFBitCountAlgorithm.h"

namespace l1t {

  class Stage1Layer2MainProcessorFirmwareImp1 : public Stage1Layer2MainProcessor {
  public:
    //Stage1Layer2MainProcessorFirmwareImp1(const FirmwareVersion & fwv /*const CaloParamsStage1 & dbPars*/);
    Stage1Layer2MainProcessorFirmwareImp1(const int fwv , CaloParamsStage1* dbPars);
    virtual ~Stage1Layer2MainProcessorFirmwareImp1();
    virtual void processEvent(const std::vector<CaloEmCand> &,
                              const std::vector<CaloRegion> &,
			      std::vector<EGamma> * egammas,
			      std::vector<Tau> * taus,
			      std::vector<Tau> * isoTaus,
			      std::vector<Jet> * jets,
			      std::vector<Jet> * preGtJets,
			      std::vector<EtSum> * etsums,
			      CaloSpare * hfSums,
			      CaloSpare * hfCounts);
  private:

    int m_fwv;
    CaloParamsStage1* m_db;

    Stage1Layer2EGammaAlgorithm* m_egAlgo;
    Stage1Layer2TauAlgorithm* m_tauAlgo;
    Stage1Layer2JetAlgorithm* m_jetAlgo;
    Stage1Layer2EtSumAlgorithm* m_sumAlgo;
    Stage1Layer2HFRingSumAlgorithm* m_hfRingAlgo;
    Stage1Layer2HFBitCountAlgorithm* m_hfBitAlgo;
  };

}

#endif
