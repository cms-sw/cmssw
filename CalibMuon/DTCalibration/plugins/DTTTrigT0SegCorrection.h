#ifndef CalibMuon_DTTTrigT0SegCorrection_H
#define CalibMuon_DTTTrigT0SegCorrection_H

/** \class DTTTrigT0SegCorrection
 *  Concrete implementation of a DTTTrigBaseCorrection.
 *  Computes t0-seg correction for tTrig
 *
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTTTrigBaseCorrection.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <string>

namespace edm {
  class ParameterSet;
  class ConsumesCollector;
}  // namespace edm

class DTTtrig;

class TH1F;
class TFile;

namespace dtCalibration {

  class DTTTrigT0SegCorrection : public DTTTrigBaseCorrection {
  public:
    // Constructor
    DTTTrigT0SegCorrection(const edm::ParameterSet&, edm::ConsumesCollector);

    // Destructor
    ~DTTTrigT0SegCorrection() override;

    void setES(const edm::EventSetup& setup) override;
    DTTTrigData correction(const DTSuperLayerId&) override;

  private:
    const TH1F* getHisto(const DTSuperLayerId&);
    std::string getHistoName(const DTSuperLayerId& slID);

    TFile* rootFile_;

    const DTTtrig* tTrigMap_;
    edm::ESGetToken<DTTtrig, DTTtrigRcd> ttrigToken_;
  };

}  // namespace dtCalibration
#endif
