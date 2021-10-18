#ifndef CalibMuon_DTTTrigResidualCorrection_H
#define CalibMuon_DTTTrigResidualCorrection_H

/** \class DTTTrigResidualCorrection
 *  Concrete implementation of a DTTTrigBaseCorrection.
 *  Computes residual correction for tTrig
 *
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTTTrigBaseCorrection.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsVdriftRcd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <string>

namespace edm {
  class ParameterSet;
}

class DTTtrig;
class DTMtime;
class DTRecoConditions;
class DTResidualFitter;

class TH1F;
class TFile;

namespace dtCalibration {

  class DTTTrigResidualCorrection : public DTTTrigBaseCorrection {
  public:
    // Constructor
    DTTTrigResidualCorrection(const edm::ParameterSet&, edm::ConsumesCollector cc);

    // Destructor
    ~DTTTrigResidualCorrection() override;

    void setES(const edm::EventSetup& setup) override;
    DTTTrigData correction(const DTSuperLayerId&) override;

  private:
    const TH1F* getHisto(const DTSuperLayerId&);
    std::string getHistoName(const DTSuperLayerId& slID);

    TFile* rootFile_;

    std::string rootBaseDir_;
    bool useFit_;
    bool useSlopesCalib_;

    double vDriftEff_[5][14][4][3];

    const DTTtrig* tTrigMap_;
    const DTMtime* mTimeMap_;            // legacy vdrift DB object
    const DTRecoConditions* vDriftMap_;  // vdrift DB object in new format
    bool readLegacyVDriftDB;             // which one to use

    DTResidualFitter* fitter_;

    edm::ESGetToken<DTTtrig, DTTtrigRcd> ttrigToken_;
    edm::ESGetToken<DTMtime, DTMtimeRcd> mTimeMapToken_;
    edm::ESGetToken<DTRecoConditions, DTRecoConditionsVdriftRcd> vDriftMapToken_;
  };

}  // namespace dtCalibration
#endif
