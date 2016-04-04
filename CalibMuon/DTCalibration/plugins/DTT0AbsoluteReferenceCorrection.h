#ifndef CalibMuon_DTT0AbsoluteReferenceCorrection_H
#define CalibMuon_DTT0AbsoluteReferenceCorrection_H

/** \class DTT0AbsoluteReferenceCorrection
 *  Concrete implementation of a DTT0BaseCorrection.
 *  Computes correction for t0
 *
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTT0BaseCorrection.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include <string>

namespace edm {
  class ParameterSet;
}

class DTT0;

namespace dtCalibration {

class DTT0AbsoluteReferenceCorrection: public DTT0BaseCorrection {
public:
  // Constructor
  DTT0AbsoluteReferenceCorrection(const edm::ParameterSet&);

  // Destructor
  virtual ~DTT0AbsoluteReferenceCorrection();

  virtual void setES(const edm::EventSetup& setup);
  virtual DTT0Data correction(const DTWireId&);

private:
  DTT0Data defaultT0(const DTWireId&);

  std::string calibChamber_;
  double reference_;

  DTChamberId chosenChamberId_;
  const DTT0 *t0Map_;
};

} // namespace
#endif
