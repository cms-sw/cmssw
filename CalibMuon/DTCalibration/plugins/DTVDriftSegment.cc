
/*
 *  See header file for a description of this class.
 *
 *  \author A. Vilela Pereira
 */

#include "DTVDriftSegment.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"
#include "CondFormats/DTObjects/interface/DTRecoConditions.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsVdriftRcd.h"

#include "CalibMuon/DTCalibration/interface/DTResidualFitter.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include <string>
#include <vector>

#include "TH1F.h"
#include "TFile.h"

using namespace std;
using namespace edm;

namespace dtCalibration {

  DTVDriftSegment::DTVDriftSegment(const ParameterSet& pset)
      : nSigmas_(pset.getUntrackedParameter<unsigned int>("nSigmasFitRange", 1)),
        mTimeMap_(nullptr),
        vDriftMap_(nullptr) {
    string rootFileName = pset.getParameter<string>("rootFileName");
    rootFile_ = new TFile(rootFileName.c_str(), "READ");

    bool debug = pset.getUntrackedParameter<bool>("debug", false);
    fitter_ = new DTResidualFitter(debug);
    //if(debug) fitter_->setVerbosity(1);

    readLegacyVDriftDB = pset.getParameter<bool>("readLegacyVDriftDB");
  }

  DTVDriftSegment::~DTVDriftSegment() {
    rootFile_->Close();
    delete fitter_;
  }

  void DTVDriftSegment::setES(const edm::EventSetup& setup) {
    // Get the map of vdrift from the setup
    if (readLegacyVDriftDB) {
      ESHandle<DTMtime> mTime;
      setup.get<DTMtimeRcd>().get(mTime);
      mTimeMap_ = &*mTime;
    } else {
      ESHandle<DTRecoConditions> hVdrift;
      setup.get<DTRecoConditionsVdriftRcd>().get(hVdrift);
      vDriftMap_ = &*hVdrift;
      // Consistency check: no parametrization is implemented for the time being
      int version = vDriftMap_->version();
      if (version != 1) {
        throw cms::Exception("Configuration") << "only version 1 is presently supported for VDriftDB";
      }
    }
  }

  DTVDriftData DTVDriftSegment::compute(DTSuperLayerId const& slId) {
    // Get original value from DB; vdrift is cm/ns , resolution is cm
    // Note that resolution is irrelevant as it is no longer used anywhere in reconstruction.

    float vDrift = 0., resolution = 0.;
    if (readLegacyVDriftDB) {  // Legacy format
      int status = mTimeMap_->get(slId, vDrift, resolution, DTVelocityUnits::cm_per_ns);
      if (status != 0)
        throw cms::Exception("DTCalibration") << "Could not find vDrift entry in DB for" << slId << endl;
    } else {  // New DB format
      vDrift = vDriftMap_->get(DTWireId(slId.rawId()));
    }

    // For RZ superlayers use original value
    if (slId.superLayer() == 2) {
      LogTrace("Calibration") << "[DTVDriftSegment]: RZ superlayer\n"
                              << "                   Will use original vDrift and resolution.";
      return DTVDriftData(vDrift, resolution);
    } else {
      TH1F* vDriftCorrHisto = getHisto(slId);
      // If empty histogram
      if (vDriftCorrHisto->GetEntries() == 0) {
        LogError("Calibration") << "[DTVDriftSegment]: Histogram " << vDriftCorrHisto->GetName() << " is empty.\n"
                                << "                   Will use original vDrift and resolution.";
        return DTVDriftData(vDrift, resolution);
      }

      LogTrace("Calibration") << "[DTVDriftSegment]: Fitting histogram " << vDriftCorrHisto->GetName();
      DTResidualFitResult fitResult = fitter_->fitResiduals(*vDriftCorrHisto, nSigmas_);
      LogTrace("Calibration") << "[DTVDriftSegment]: \n"
                              << "   Fit Mean  = " << fitResult.fitMean << " +/- " << fitResult.fitMeanError << "\n"
                              << "   Fit Sigma = " << fitResult.fitSigma << " +/- " << fitResult.fitSigmaError;

      float vDriftCorr = fitResult.fitMean;
      float vDriftNew = vDrift * (1. - vDriftCorr);
      float resolutionNew = resolution;
      return DTVDriftData(vDriftNew, resolutionNew);
    }
  }

  TH1F* DTVDriftSegment::getHisto(const DTSuperLayerId& slId) {
    string histoName = getHistoName(slId);
    TH1F* histo = static_cast<TH1F*>(rootFile_->Get(histoName.c_str()));
    if (!histo)
      throw cms::Exception("DTCalibration") << "v-drift correction histogram not found:" << histoName << endl;
    return histo;
  }

  string DTVDriftSegment::getHistoName(const DTSuperLayerId& slId) {
    DTChamberId chId = slId.chamberId();

    // Compose the chamber name
    std::string wheel = std::to_string(chId.wheel());
    std::string station = std::to_string(chId.station());
    std::string sector = std::to_string(chId.sector());

    string chHistoName = "_W" + wheel + "_St" + station + "_Sec" + sector;

    return (slId.superLayer() != 2) ? ("hRPhiVDriftCorr" + chHistoName) : ("hRZVDriftCorr" + chHistoName);
  }

}  // namespace dtCalibration
