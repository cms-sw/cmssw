#ifndef CalibMuon_DTTTrigMatchRPhi_H
#define CalibMuon_DTTTrigMatchRPhi_H

/** \class DTTTrigMatchRPhi
 *  Concrete implementation of a DTTTrigBaseCorrection.
 *  Matches tTrig values for RPhi SL's
 *
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTTTrigBaseCorrection.h"

#include <string>

namespace edm {
  class ParameterSet;
}

class DTTtrig;

namespace dtCalibration {

class DTTTrigMatchRPhi: public DTTTrigBaseCorrection {
public:
  // Constructor
  DTTTrigMatchRPhi(const edm::ParameterSet&);

  // Destructor
  ~DTTTrigMatchRPhi() override;

  void setES(const edm::EventSetup& setup) override;
  DTTTrigData correction(const DTSuperLayerId&) override;

private:
  const DTTtrig *tTrigMap_;

  std::string dbLabel;
};

} // namespace
#endif
