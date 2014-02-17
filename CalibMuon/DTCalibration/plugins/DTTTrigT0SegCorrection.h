#ifndef CalibMuon_DTTTrigT0SegCorrection_H
#define CalibMuon_DTTTrigT0SegCorrection_H

/** \class DTTTrigT0SegCorrection
 *  Concrete implementation of a DTTTrigBaseCorrection.
 *  Computes t0-seg correction for tTrig
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

class TH1F;
class TFile;

namespace dtCalibration {

class DTTTrigT0SegCorrection: public DTTTrigBaseCorrection {
public:
  // Constructor
  DTTTrigT0SegCorrection(const edm::ParameterSet&);

  // Destructor
  virtual ~DTTTrigT0SegCorrection();

  virtual void setES(const edm::EventSetup& setup);
  virtual DTTTrigData correction(const DTSuperLayerId&);

private:
  const TH1F* getHisto(const DTSuperLayerId&);
  std::string getHistoName(const DTSuperLayerId& slID);

  TFile* rootFile_;

  std::string dbLabel;

  const DTTtrig *tTrigMap_;
};

} // namespace
#endif
