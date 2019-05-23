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
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include "Stage1Layer2EGammaAlgorithm.h"
#include "Stage1Layer2EtSumAlgorithm.h"
#include "Stage1Layer2JetAlgorithm.h"
#include "Stage1Layer2TauAlgorithm.h"
#include "Stage1Layer2HFRingSumAlgorithm.h"
#include "Stage1Layer2HFBitCountAlgorithm.h"

#include <memory>

namespace l1t {

  class Stage1Layer2MainProcessorFirmwareImp1 : public Stage1Layer2MainProcessor {
  public:
    //Stage1Layer2MainProcessorFirmwareImp1(const FirmwareVersion & fwv /*const CaloParamsHelper & dbPars*/);
    Stage1Layer2MainProcessorFirmwareImp1(const int fwv, CaloParamsHelper const *dbPars);
    ~Stage1Layer2MainProcessorFirmwareImp1() override = default;
    void processEvent(const std::vector<CaloEmCand> &,
                      const std::vector<CaloRegion> &,
                      std::vector<EGamma> *egammas,
                      std::vector<Tau> *taus,
                      std::vector<Tau> *isoTaus,
                      std::vector<Jet> *jets,
                      std::vector<Jet> *preGtJets,
                      std::vector<EtSum> *etsums,
                      CaloSpare *hfSums,
                      CaloSpare *hfCounts) override;

  private:
    int m_fwv;

    std::unique_ptr<Stage1Layer2EGammaAlgorithm> m_egAlgo;
    std::unique_ptr<Stage1Layer2TauAlgorithm> m_tauAlgo;
    std::unique_ptr<Stage1Layer2JetAlgorithm> m_jetAlgo;
    std::unique_ptr<Stage1Layer2EtSumAlgorithm> m_sumAlgo;
    std::unique_ptr<Stage1Layer2HFRingSumAlgorithm> m_hfRingAlgo;
    std::unique_ptr<Stage1Layer2HFBitCountAlgorithm> m_hfBitAlgo;
  };

}  // namespace l1t

#endif
