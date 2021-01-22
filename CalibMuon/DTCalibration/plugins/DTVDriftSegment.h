#ifndef CalibMuon_DTCalibration_DTVDriftSegment_h
#define CalibMuon_DTCalibration_DTVDriftSegment_h

/** \class DTVDriftSegment
 *  Concrete implementation of a DTVDriftBaseAlgo.
 *  Computes vDrift using fit result segment by segment.
 *
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTVDriftBaseAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class DTMtime;
class DTRecoConditions;
class DTResidualFitter;
class TH1F;
class TFile;

namespace dtCalibration {

  class DTVDriftSegment : public DTVDriftBaseAlgo {
  public:
    DTVDriftSegment(edm::ParameterSet const&);
    ~DTVDriftSegment() override;

    void setES(const edm::EventSetup& setup) override;
    DTVDriftData compute(const DTSuperLayerId&) override;

  private:
    TH1F* getHisto(const DTSuperLayerId&);
    std::string getHistoName(const DTSuperLayerId&);

    unsigned int nSigmas_;

    const DTMtime* mTimeMap_;            // legacy DB object
    const DTRecoConditions* vDriftMap_;  // DB object in new format
    bool readLegacyVDriftDB;             // which one to use
    TFile* rootFile_;
    DTResidualFitter* fitter_;
  };

}  // namespace dtCalibration
#endif
