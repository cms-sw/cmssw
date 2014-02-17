#ifndef CalibMuon_DTTTrigMatchRPhi_H
#define CalibMuon_DTTTrigMatchRPhi_H

/** \class DTTTrigMatchRPhi
 *  Concrete implementation of a DTTTrigBaseCorrection.
 *  Matches tTrig values for RPhi SL's
 *
 *  $Revision: 1.3 $
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
  virtual ~DTTTrigMatchRPhi();

  virtual void setES(const edm::EventSetup& setup);
  virtual DTTTrigData correction(const DTSuperLayerId&);

private:
  const DTTtrig *tTrigMap_;

  std::string dbLabel;
};

} // namespace
#endif
