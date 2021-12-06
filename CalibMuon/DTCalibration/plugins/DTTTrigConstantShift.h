#ifndef CalibMuon_DTTTrigConstantShift_H
#define CalibMuon_DTTTrigConstantShift_H

/** \class DTTTrigConstantShift
 *  Concrete implementation of a DTTTrigBaseCorrection.
 *  Applies constant shift to tTrig values
 *
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTTTrigBaseCorrection.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <string>

namespace edm {
  class ParameterSet;
}

class DTTtrig;

namespace dtCalibration {

  class DTTTrigConstantShift : public DTTTrigBaseCorrection {
  public:
    // Constructor
    DTTTrigConstantShift(const edm::ParameterSet&, edm::ConsumesCollector);

    // Destructor
    ~DTTTrigConstantShift() override;

    void setES(const edm::EventSetup& setup) override;
    DTTTrigData correction(const DTSuperLayerId&) override;

  private:
    std::string calibChamber_;
    double value_;

    const DTTtrig* tTrigMap_;
    DTChamberId chosenChamberId_;
    edm::ESGetToken<DTTtrig, DTTtrigRcd> ttrigToken_;
  };

}  // namespace dtCalibration
#endif
