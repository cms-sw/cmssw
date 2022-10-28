#ifndef CalibMuon_DTTTrigMatchRPhi_H
#define CalibMuon_DTTTrigMatchRPhi_H

/** \class DTTTrigMatchRPhi
 *  Concrete implementation of a DTTTrigBaseCorrection.
 *  Matches tTrig values for RPhi SL's
 *
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTTTrigBaseCorrection.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <string>

namespace edm {
  class ParameterSet;
}

class DTTtrig;

namespace dtCalibration {

  class DTTTrigMatchRPhi : public DTTTrigBaseCorrection {
  public:
    // Constructor
    DTTTrigMatchRPhi(const edm::ParameterSet&, edm::ConsumesCollector);

    // Destructor
    ~DTTTrigMatchRPhi() override;

    void setES(const edm::EventSetup& setup) override;
    DTTTrigData correction(const DTSuperLayerId&) override;

  private:
    const DTTtrig* tTrigMap_;
    edm::ESGetToken<DTTtrig, DTTtrigRcd> ttrigToken_;
  };

}  // namespace dtCalibration
#endif
