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
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

namespace edm {
  class ParameterSet;
}

class DTTtrig;
class DTGeometry;

namespace dtCalibration {

  class DTTTrigFillWithAverage : public DTTTrigBaseCorrection {
  public:
    // Constructor
    DTTTrigFillWithAverage(const edm::ParameterSet&, edm::ConsumesCollector);

    // Destructor
    ~DTTTrigFillWithAverage() override;

    void setES(const edm::EventSetup& setup) override;
    DTTTrigData correction(const DTSuperLayerId&) override;

  private:
    void getAverage();

    const DTTtrig* tTrigMap_;
    edm::ESHandle<DTGeometry> muonGeom_;

    edm::ESGetToken<DTTtrig, DTTtrigRcd> ttrigToken_;
    edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;

    struct {
      float aveMean;
      float rmsMean;
      float aveSigma;
      float rmsSigma;
      float aveKFactor;
    } initialTTrig_;

    bool foundAverage_;
  };

}  // namespace dtCalibration
#endif
