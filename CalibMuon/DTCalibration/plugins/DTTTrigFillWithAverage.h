#ifndef CalibMuon_DTTTrigFillWithAverage_H
#define CalibMuon_DTTTrigFillWithAverage_H

/** \class DTTTrigFillWithAverage
 *  Concrete implementation of a DTTTrigBaseCorrection.
 *  Fills missing tTrig values in DB 
 *
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTTTrigBaseCorrection.h"
#include "FWCore/Framework/interface/ESHandle.h"

namespace edm {
  class ParameterSet;
}

class DTTtrig;
class DTGeometry;

namespace dtCalibration {

class DTTTrigFillWithAverage: public DTTTrigBaseCorrection {
public:
  // Constructor
  DTTTrigFillWithAverage(const edm::ParameterSet&);

  // Destructor
  ~DTTTrigFillWithAverage() override;

  void setES(const edm::EventSetup& setup) override;
  DTTTrigData correction(const DTSuperLayerId&) override;

private:
  void getAverage();

  const DTTtrig *tTrigMap_;
  edm::ESHandle<DTGeometry> muonGeom_;

  std::string dbLabel;

  struct {
    float aveMean;
    float rmsMean;
    float aveSigma;
    float rmsSigma;
    float aveKFactor;
  } initialTTrig_;

  bool foundAverage_; 
};

} // namespace
#endif
