#ifndef CalibMuon_DTT0ChamberReferenceCorrection_H
#define CalibMuon_DTT0ChamberReferenceCorrection_H

/** \class DTT0ChamberReferenceCorrection
 *  Concrete implementation of a DTT0BaseCorrection.
 *  Computes correction for t0
 *
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTT0BaseCorrection.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include "CondFormats/DTObjects/interface/DTT0.h"

#include <string>

namespace edm {
  class ParameterSet;
}

class DTT0;

namespace dtCalibration {

  class DTT0ChamberReferenceCorrection : public DTT0BaseCorrection {
  public:
    // Constructor
    DTT0ChamberReferenceCorrection(const edm::ParameterSet&, edm::ConsumesCollector);

    // Destructor
    ~DTT0ChamberReferenceCorrection() override;

    void setES(const edm::EventSetup& setup) override;
    DTT0Data correction(const DTWireId&) override;

  private:
    DTT0Data defaultT0(const DTWireId&);

    std::string calibChamber_;

    DTChamberId chosenChamberId_;
    const DTT0* t0Map_;
    edm::ESGetToken<DTT0, DTT0Rcd> t0Token_;
  };

}  // namespace dtCalibration
#endif
