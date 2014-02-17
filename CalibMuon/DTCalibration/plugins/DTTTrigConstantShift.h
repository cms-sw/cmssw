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

#include <string>

namespace edm {
  class ParameterSet;
}

class DTTtrig;

namespace dtCalibration {

class DTTTrigConstantShift: public DTTTrigBaseCorrection {
public:
  // Constructor
  DTTTrigConstantShift(const edm::ParameterSet&);

  // Destructor
  virtual ~DTTTrigConstantShift();

  virtual void setES(const edm::EventSetup& setup);
  virtual DTTTrigData correction(const DTSuperLayerId&);

private:
  std::string dbLabel_;
  std::string calibChamber_;
  double value_;

  const DTTtrig *tTrigMap_;
  DTChamberId chosenChamberId_;
};

} // namespace
#endif
