#ifndef CalibMuon_DTT0WireInChamberReferenceCorrection_H
#define CalibMuon_DTT0WireInChamberReferenceCorrection_H

/** \class DTT0WireInChamberReferenceCorrection
 *  Concrete implementation of a DTT0BaseCorrection.
 *  Computes correction for t0
 *
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTT0BaseCorrection.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <string>

namespace edm {
  class ParameterSet;
}

class DTT0;
class DTGeometry;

namespace dtCalibration {

  class DTT0WireInChamberReferenceCorrection : public DTT0BaseCorrection {
  public:
    // Constructor
    DTT0WireInChamberReferenceCorrection(const edm::ParameterSet&);

    // Destructor
    ~DTT0WireInChamberReferenceCorrection() override;

    void setES(const edm::EventSetup& setup) override;
    DTT0Data correction(const DTWireId&) override;

  private:
    DTT0Data defaultT0(const DTWireId&);

    std::string calibChamber_;

    DTChamberId chosenChamberId_;
    const DTT0* t0Map_;
    edm::ESHandle<DTGeometry> dtGeom_;
  };

}  // namespace dtCalibration
#endif
