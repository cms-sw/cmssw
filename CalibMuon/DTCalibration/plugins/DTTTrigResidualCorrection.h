#ifndef CalibMuon_DTTTrigResidualCorrection_H
#define CalibMuon_DTTTrigResidualCorrection_H

/** \class DTTTrigResidualCorrection
 *  Concrete implementation of a DTTTrigBaseCorrection.
 *  Computes residual correction for tTrig
 *
 *  $Revision: 1.2 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTTTrigBaseCorrection.h"

#include <string>

namespace edm {
  class ParameterSet;
}

class DTTtrig;
class DTMtime;

class TH1F;
class TFile;

class DTTTrigResidualCorrection: public DTTTrigBaseCorrection {
public:
  // Constructor
  DTTTrigResidualCorrection(const edm::ParameterSet&);

  // Destructor
  virtual ~DTTTrigResidualCorrection();

  virtual void setES(const edm::EventSetup& setup);
  virtual DTTTrigData correction(const DTSuperLayerId&);

private:
  const TH1F* getHisto(const DTSuperLayerId&);
  std::string getHistoName(const DTSuperLayerId& slID);

  TFile* rootFile_;  

  bool useFit_;

  std::string dbLabel;

  double v_eff[5][14][4][3];

  const DTTtrig *tTrigMap_;
  const DTMtime *mTimeMap_;
};
#endif
