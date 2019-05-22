#include "DQM/SiStripCommissioningAnalysis/interface/CalibrationAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/CalibrationAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommissioningAnalysis/interface/SiStripPulseShape.h"
#include "TProfile.h"
#include "TF1.h"
#include "TH1.h"
#include "TVirtualFitter.h"
#include "TFitResultPtr.h"
#include "TFitResult.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include "Math/MinimizerOptions.h"

using namespace sistrip;

// ----------------------------------------------------------------------------
//
CalibrationAlgorithm::CalibrationAlgorithm(const edm::ParameterSet& pset, CalibrationAnalysis* const anal)
    : CommissioningAlgorithm(anal), cal_(nullptr) {}

// ----------------------------------------------------------------------------
//
void CalibrationAlgorithm::extract(const std::vector<TH1*>& histos) {
  // extract analysis object which should be already created
  if (!anal()) {
    edm::LogWarning(mlCommissioning_) << "[CalibrationAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to base Analysis object!";
    return;
  }

  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>(anal());
  cal_ = dynamic_cast<CalibrationAnalysis*>(tmp);

  if (!cal_) {
    edm::LogWarning(mlCommissioning_) << "[CalibrationAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to derived Analysis object!";
    return;
  }

  // Extract FED key from histo title
  if (!histos.empty()) {
    cal_->fedKey(extractFedKey(histos.front()));
  }

  // Extract histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  unsigned int cnt = 0;
  for (; ihis != histos.end(); ihis++, cnt++) {
    // Check for NULL pointer
    if (!(*ihis)) {
      continue;
    }

    // Check name
    SiStripHistoTitle title((*ihis)->GetName());
    if (title.runType() != sistrip::CALIBRATION && title.runType() != sistrip::CALIBRATION_DECO) {
      cal_->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }

    /// extract isha, vfs and calchan values, as well as filling the histogram objects
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(title.extraInfo());
    while (std::getline(tokenStream, token, '_')) {
      tokens.push_back(token);
    }

    ////////
    Histo histo_temp;
    histo_temp.first = *ihis;
    histo_temp.second = (*ihis)->GetTitle();
    histo_temp.first->Sumw2();
    histo_.push_back(histo_temp);
    apvId_.push_back(title.channel() % 2);
    stripId_.push_back(std::stoi(tokens.at(1)) * 16 + std::stoi(tokens.at(3)));
    calChan_.push_back(std::stoi(tokens.at(1)));
  }
}

// ----------------------------------------------------------------------------
//
void CalibrationAlgorithm::analyse() {
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Migrad");
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);

  if (!cal_) {
    edm::LogWarning(mlCommissioning_) << "[CalibrationAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to derived Analysis object!";
    return;
  }

  float Amean[2] = {-1., -1.};
  float Amin[2] = {-1., -1.};
  float Amax[2] = {-1., -1.};
  float Aspread[2] = {-1., -1.};
  float Tmean[2] = {-1., -1.};
  float Tmin[2] = {-1., -1.};
  float Tmax[2] = {-1., -1.};
  float Tspread[2] = {-1., -1.};
  float Rmean[2] = {-1., -1.};
  float Rmin[2] = {-1., -1.};
  float Rmax[2] = {-1., -1.};
  float Rspread[2] = {-1., -1.};
  float Cmean[2] = {-1., -1.};
  float Cmin[2] = {-1., -1.};
  float Cmax[2] = {-1., -1.};
  float Cspread[2] = {-1., -1.};
  float Smean[2] = {-1., -1.};
  float Smin[2] = {-1., -1.};
  float Smax[2] = {-1., -1.};
  float Sspread[2] = {-1., -1.};
  float Kmean[2] = {-1., -1.};
  float Kmin[2] = {-1., -1.};
  float Kmax[2] = {-1., -1.};
  float Kspread[2] = {-1., -1.};
  // turnOn
  float Omean[2] = {-1., -1.};
  float Omin[2] = {-1., -1.};
  float Omax[2] = {-1., -1.};
  float Ospread[2] = {-1., -1.};
  // maximum
  float Mmean[2] = {-1., -1.};
  float Mmin[2] = {-1., -1.};
  float Mmax[2] = {-1., -1.};
  float Mspread[2] = {-1., -1.};
  // undershoot
  float Umean[2] = {-1., -1.};
  float Umin[2] = {-1., -1.};
  float Umax[2] = {-1., -1.};
  float Uspread[2] = {-1., -1.};
  // baseline
  float Bmean[2] = {-1., -1.};
  float Bmin[2] = {-1., -1.};
  float Bmax[2] = {-1., -1.};
  float Bspread[2] = {-1., -1.};

  ////////
  TFitResultPtr fit_result;
  TF1* fit_function = nullptr;
  if (cal_->deconv_) {
    fit_function = new TF1("fit_function_deco", fdeconv, 0, 400, 7);
    fit_function->SetParameters(4, 25, 25, 50, 250, 25, 0.75);
  } else {
    fit_function = new TF1("fit_function_peak", fpeak, 0, 400, 6);
    fit_function->SetParameters(4, 50, 50, 70, 250, 20);
  }

  //////////
  std::vector<unsigned int> nStrips(2, 0.);

  for (size_t ihist = 0; ihist < histo_.size(); ihist++) {
    if (!histo_[ihist].first) {
      edm::LogWarning(mlCommissioning_) << " NULL pointer to histogram for: " << histo_[ihist].second << " !";
      return;
    }

    cal_->amplitude_[apvId_[ihist]][stripId_[ihist]] = 0;
    cal_->baseline_[apvId_[ihist]][stripId_[ihist]] = 0;
    cal_->riseTime_[apvId_[ihist]][stripId_[ihist]] = 0;
    cal_->turnOn_[apvId_[ihist]][stripId_[ihist]] = 0;
    cal_->peakTime_[apvId_[ihist]][stripId_[ihist]] = 0;
    cal_->undershoot_[apvId_[ihist]][stripId_[ihist]] = 0;
    cal_->tail_[apvId_[ihist]][stripId_[ihist]] = 0;
    cal_->decayTime_[apvId_[ihist]][stripId_[ihist]] = 0;
    cal_->smearing_[apvId_[ihist]][stripId_[ihist]] = 0;
    cal_->chi2_[apvId_[ihist]][stripId_[ihist]] = 0;
    cal_->isvalid_[apvId_[ihist]][stripId_[ihist]] = true;

    if (histo_[ihist].first->Integral() == 0) {
      cal_->isvalid_[apvId_[ihist]][stripId_[ihist]] = false;
      continue;
    }

    // rescale the plot and set reasonable errors
    correctDistribution(histo_[ihist].first);

    // from NOTE2009_021 : The charge injection provided by the calibration circuit is known with a precision of 5%
    float error = histo_[ihist].first->GetMaximum() * 0.05;
    for (int i = 1; i <= histo_[ihist].first->GetNbinsX(); ++i)
      histo_[ihist].first->SetBinError(i, error);

    // set intial par
    if (cal_->deconv_)
      fit_function->SetParameters(10, 15, 30, 10, 350, 50, 0.75);
    else
      fit_function->SetParameters(6, 40, 40, 70, 350, 20);

    fit_result = histo_[ihist].first->Fit(fit_function, "QS");

    // fit-result should exist and have a resonably good status
    if (not fit_result.Get())
      continue;

    float maximum_ampl = fit_function->GetMaximum();
    float peak_time = fit_function->GetMaximumX();
    float baseline = baseLine(fit_function);
    float turn_on_time = turnOn(fit_function, baseline);
    float rise_time = peak_time - turn_on_time;

    // start filling info
    cal_->amplitude_[apvId_[ihist]][stripId_[ihist]] = maximum_ampl - baseline;
    cal_->baseline_[apvId_[ihist]][stripId_[ihist]] = baseline;
    cal_->riseTime_[apvId_[ihist]][stripId_[ihist]] = rise_time;
    cal_->turnOn_[apvId_[ihist]][stripId_[ihist]] = turn_on_time;
    cal_->peakTime_[apvId_[ihist]][stripId_[ihist]] = peak_time;

    if (cal_->deconv_ and fit_function->GetMinimumX() > peak_time)  // make sure the minimum is after the peak-time
      cal_->undershoot_[apvId_[ihist]][stripId_[ihist]] =
          100 * (fit_function->GetMinimum() - baseline) / (maximum_ampl - baseline);
    else
      cal_->undershoot_[apvId_[ihist]][stripId_[ihist]] = 0;

    // Bin related to peak + 125 ns
    int lastBin = histo_[ihist].first->FindBin(peak_time + 125);
    if (lastBin > histo_[ihist].first->GetNbinsX() - 4)
      lastBin = histo_[ihist].first->GetNbinsX() - 4;

    // tail is the amplitude at 5 bx from the maximum
    cal_->tail_[apvId_[ihist]][stripId_[ihist]] =
        100 * (histo_[ihist].first->GetBinContent(lastBin) - baseline) / (maximum_ampl - baseline);

    // reaches 1/e of the peak amplitude
    cal_->decayTime_[apvId_[ihist]][stripId_[ihist]] = decayTime(fit_function) - peak_time;
    cal_->smearing_[apvId_[ihist]][stripId_[ihist]] = 0;
    cal_->chi2_[apvId_[ihist]][stripId_[ihist]] =
        fit_function->GetChisquare() / (histo_[ihist].first->GetNbinsX() - fit_function->GetNpar());

    // calibration channel
    cal_->calChan_ = calChan_[ihist];

    // apply quality requirements
    bool isvalid = true;
    if (not cal_->deconv_) {  // peak-mode

      if (cal_->amplitude_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minAmplitudeThreshold_)
        isvalid = false;
      if (cal_->baseline_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minBaselineThreshold_)
        isvalid = false;
      else if (cal_->baseline_[apvId_[ihist]][stripId_[ihist]] > CalibrationAnalysis::maxBaselineThreshold_)
        isvalid = false;
      if (cal_->decayTime_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minDecayTimeThreshold_)
        isvalid = false;
      else if (cal_->decayTime_[apvId_[ihist]][stripId_[ihist]] > CalibrationAnalysis::maxDecayTimeThreshold_)
        isvalid = false;
      if (cal_->peakTime_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minPeakTimeThreshold_)
        isvalid = false;
      else if (cal_->peakTime_[apvId_[ihist]][stripId_[ihist]] > CalibrationAnalysis::maxPeakTimeThreshold_)
        isvalid = false;
      if (cal_->riseTime_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minRiseTimeThreshold_)
        isvalid = false;
      else if (cal_->riseTime_[apvId_[ihist]][stripId_[ihist]] > CalibrationAnalysis::maxRiseTimeThreshold_)
        isvalid = false;
      if (cal_->turnOn_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minTurnOnThreshold_)
        isvalid = false;
      else if (cal_->turnOn_[apvId_[ihist]][stripId_[ihist]] > CalibrationAnalysis::maxTurnOnThreshold_)
        isvalid = false;
      if (cal_->chi2_[apvId_[ihist]][stripId_[ihist]] > CalibrationAnalysis::maxChi2Threshold_)
        isvalid = false;

    } else {
      if (fit_function->GetMinimumX() < peak_time)
        isvalid = false;
      if (cal_->amplitude_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minAmplitudeThreshold_)
        isvalid = false;
      if (cal_->baseline_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minBaselineThreshold_)
        isvalid = false;
      if (cal_->baseline_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minBaselineThreshold_)
        isvalid = false;
      else if (cal_->baseline_[apvId_[ihist]][stripId_[ihist]] > CalibrationAnalysis::maxBaselineThreshold_)
        isvalid = false;
      if (cal_->chi2_[apvId_[ihist]][stripId_[ihist]] > CalibrationAnalysis::maxChi2Threshold_)
        isvalid = false;
      if (cal_->turnOn_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minTurnOnThresholdDeco_)
        isvalid = false;
      else if (cal_->turnOn_[apvId_[ihist]][stripId_[ihist]] > CalibrationAnalysis::maxTurnOnThresholdDeco_)
        isvalid = false;
      if (cal_->decayTime_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minDecayTimeThresholdDeco_)
        isvalid = false;
      else if (cal_->decayTime_[apvId_[ihist]][stripId_[ihist]] > CalibrationAnalysis::maxDecayTimeThresholdDeco_)
        isvalid = false;
      if (cal_->peakTime_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minPeakTimeThresholdDeco_)
        isvalid = false;
      else if (cal_->peakTime_[apvId_[ihist]][stripId_[ihist]] > CalibrationAnalysis::maxPeakTimeThresholdDeco_)
        isvalid = false;
      if (cal_->riseTime_[apvId_[ihist]][stripId_[ihist]] < CalibrationAnalysis::minRiseTimeThresholdDeco_)
        isvalid = false;
      else if (cal_->riseTime_[apvId_[ihist]][stripId_[ihist]] > CalibrationAnalysis::maxRiseTimeThresholdDeco_)
        isvalid = false;
    }

    if (not isvalid) {  // not valid set default to zero for all quantities

      cal_->amplitude_[apvId_[ihist]][stripId_[ihist]] = 0;
      cal_->baseline_[apvId_[ihist]][stripId_[ihist]] = 0;
      cal_->riseTime_[apvId_[ihist]][stripId_[ihist]] = 0;
      cal_->turnOn_[apvId_[ihist]][stripId_[ihist]] = 0;
      cal_->peakTime_[apvId_[ihist]][stripId_[ihist]] = 0;
      cal_->undershoot_[apvId_[ihist]][stripId_[ihist]] = 0;
      cal_->tail_[apvId_[ihist]][stripId_[ihist]] = 0;
      cal_->decayTime_[apvId_[ihist]][stripId_[ihist]] = 0;
      cal_->smearing_[apvId_[ihist]][stripId_[ihist]] = 0;
      cal_->chi2_[apvId_[ihist]][stripId_[ihist]] = 0;
      cal_->isvalid_[apvId_[ihist]][stripId_[ihist]] = false;
      continue;
    }

    // in case is valid
    nStrips[apvId_[ihist]]++;

    //compute mean, max, min, spread only for valid strips
    Amean[apvId_[ihist]] += cal_->amplitude_[apvId_[ihist]][stripId_[ihist]];
    Amin[apvId_[ihist]] = Amin[apvId_[ihist]] < cal_->amplitude_[apvId_[ihist]][stripId_[ihist]]
                              ? Amin[apvId_[ihist]]
                              : cal_->amplitude_[apvId_[ihist]][stripId_[ihist]];
    Amax[apvId_[ihist]] = Amax[apvId_[ihist]] > cal_->amplitude_[apvId_[ihist]][stripId_[ihist]]
                              ? Amax[apvId_[ihist]]
                              : cal_->amplitude_[apvId_[ihist]][stripId_[ihist]];
    Aspread[apvId_[ihist]] +=
        cal_->amplitude_[apvId_[ihist]][stripId_[ihist]] * cal_->amplitude_[apvId_[ihist]][stripId_[ihist]];

    Tmean[apvId_[ihist]] += cal_->tail_[apvId_[ihist]][stripId_[ihist]];
    Tmin[apvId_[ihist]] = Tmin[apvId_[ihist]] < cal_->tail_[apvId_[ihist]][stripId_[ihist]]
                              ? Tmin[apvId_[ihist]]
                              : cal_->tail_[apvId_[ihist]][stripId_[ihist]];
    Tmax[apvId_[ihist]] = Tmax[apvId_[ihist]] > cal_->tail_[apvId_[ihist]][stripId_[ihist]]
                              ? Tmax[apvId_[ihist]]
                              : cal_->tail_[apvId_[ihist]][stripId_[ihist]];
    Tspread[apvId_[ihist]] += cal_->tail_[apvId_[ihist]][stripId_[ihist]] * cal_->tail_[apvId_[ihist]][stripId_[ihist]];

    Rmean[apvId_[ihist]] += cal_->riseTime_[apvId_[ihist]][stripId_[ihist]];
    Rmin[apvId_[ihist]] = Rmin[apvId_[ihist]] < cal_->riseTime_[apvId_[ihist]][stripId_[ihist]]
                              ? Rmin[apvId_[ihist]]
                              : cal_->riseTime_[apvId_[ihist]][stripId_[ihist]];
    Rmax[apvId_[ihist]] = Rmax[apvId_[ihist]] > cal_->riseTime_[apvId_[ihist]][stripId_[ihist]]
                              ? Rmax[apvId_[ihist]]
                              : cal_->riseTime_[apvId_[ihist]][stripId_[ihist]];
    Rspread[apvId_[ihist]] +=
        cal_->riseTime_[apvId_[ihist]][stripId_[ihist]] * cal_->riseTime_[apvId_[ihist]][stripId_[ihist]];

    Cmean[apvId_[ihist]] += cal_->decayTime_[apvId_[ihist]][stripId_[ihist]];
    Cmin[apvId_[ihist]] = Cmin[apvId_[ihist]] < cal_->decayTime_[apvId_[ihist]][stripId_[ihist]]
                              ? Cmin[apvId_[ihist]]
                              : cal_->decayTime_[apvId_[ihist]][stripId_[ihist]];
    Cmax[apvId_[ihist]] = Cmax[apvId_[ihist]] > cal_->decayTime_[apvId_[ihist]][stripId_[ihist]]
                              ? Cmax[apvId_[ihist]]
                              : cal_->decayTime_[apvId_[ihist]][stripId_[ihist]];
    Cspread[apvId_[ihist]] +=
        cal_->decayTime_[apvId_[ihist]][stripId_[ihist]] * cal_->decayTime_[apvId_[ihist]][stripId_[ihist]];

    Smean[apvId_[ihist]] += cal_->smearing_[apvId_[ihist]][stripId_[ihist]];
    Smin[apvId_[ihist]] = Smin[apvId_[ihist]] < cal_->smearing_[apvId_[ihist]][stripId_[ihist]]
                              ? Smin[apvId_[ihist]]
                              : cal_->smearing_[apvId_[ihist]][stripId_[ihist]];
    Smax[apvId_[ihist]] = Smax[apvId_[ihist]] > cal_->smearing_[apvId_[ihist]][stripId_[ihist]]
                              ? Smax[apvId_[ihist]]
                              : cal_->smearing_[apvId_[ihist]][stripId_[ihist]];
    Sspread[apvId_[ihist]] +=
        cal_->smearing_[apvId_[ihist]][stripId_[ihist]] * cal_->smearing_[apvId_[ihist]][stripId_[ihist]];

    Kmean[apvId_[ihist]] += cal_->chi2_[apvId_[ihist]][stripId_[ihist]];
    Kmin[apvId_[ihist]] = Kmin[apvId_[ihist]] < cal_->chi2_[apvId_[ihist]][stripId_[ihist]]
                              ? Kmin[apvId_[ihist]]
                              : cal_->chi2_[apvId_[ihist]][stripId_[ihist]];
    Kmax[apvId_[ihist]] = Kmax[apvId_[ihist]] > cal_->chi2_[apvId_[ihist]][stripId_[ihist]]
                              ? Kmax[apvId_[ihist]]
                              : cal_->chi2_[apvId_[ihist]][stripId_[ihist]];
    Kspread[apvId_[ihist]] += cal_->chi2_[apvId_[ihist]][stripId_[ihist]] * cal_->chi2_[apvId_[ihist]][stripId_[ihist]];

    Omean[apvId_[ihist]] += cal_->turnOn_[apvId_[ihist]][stripId_[ihist]];
    Omin[apvId_[ihist]] = Omin[apvId_[ihist]] < cal_->turnOn_[apvId_[ihist]][stripId_[ihist]]
                              ? Omin[apvId_[ihist]]
                              : cal_->turnOn_[apvId_[ihist]][stripId_[ihist]];
    Omax[apvId_[ihist]] = Omax[apvId_[ihist]] > cal_->turnOn_[apvId_[ihist]][stripId_[ihist]]
                              ? Omax[apvId_[ihist]]
                              : cal_->turnOn_[apvId_[ihist]][stripId_[ihist]];
    Ospread[apvId_[ihist]] +=
        cal_->turnOn_[apvId_[ihist]][stripId_[ihist]] * cal_->turnOn_[apvId_[ihist]][stripId_[ihist]];

    Mmean[apvId_[ihist]] += cal_->peakTime_[apvId_[ihist]][stripId_[ihist]];
    Mmin[apvId_[ihist]] = Mmin[apvId_[ihist]] < cal_->peakTime_[apvId_[ihist]][stripId_[ihist]]
                              ? Mmin[apvId_[ihist]]
                              : cal_->peakTime_[apvId_[ihist]][stripId_[ihist]];
    Mmax[apvId_[ihist]] = Mmax[apvId_[ihist]] > cal_->peakTime_[apvId_[ihist]][stripId_[ihist]]
                              ? Mmax[apvId_[ihist]]
                              : cal_->peakTime_[apvId_[ihist]][stripId_[ihist]];
    Mspread[apvId_[ihist]] +=
        cal_->peakTime_[apvId_[ihist]][stripId_[ihist]] * cal_->peakTime_[apvId_[ihist]][stripId_[ihist]];

    Umean[apvId_[ihist]] += cal_->undershoot_[apvId_[ihist]][stripId_[ihist]];
    Umin[apvId_[ihist]] = Umin[apvId_[ihist]] < cal_->undershoot_[apvId_[ihist]][stripId_[ihist]]
                              ? Umin[apvId_[ihist]]
                              : cal_->undershoot_[apvId_[ihist]][stripId_[ihist]];
    Umax[apvId_[ihist]] = Umax[apvId_[ihist]] > cal_->undershoot_[apvId_[ihist]][stripId_[ihist]]
                              ? Umax[apvId_[ihist]]
                              : cal_->undershoot_[apvId_[ihist]][stripId_[ihist]];
    Uspread[apvId_[ihist]] +=
        cal_->undershoot_[apvId_[ihist]][stripId_[ihist]] * cal_->undershoot_[apvId_[ihist]][stripId_[ihist]];

    Bmean[apvId_[ihist]] += cal_->baseline_[apvId_[ihist]][stripId_[ihist]];
    Bmin[apvId_[ihist]] = Bmin[apvId_[ihist]] < cal_->baseline_[apvId_[ihist]][stripId_[ihist]]
                              ? Bmin[apvId_[ihist]]
                              : cal_->baseline_[apvId_[ihist]][stripId_[ihist]];
    Bmax[apvId_[ihist]] = Bmax[apvId_[ihist]] > cal_->baseline_[apvId_[ihist]][stripId_[ihist]]
                              ? Bmax[apvId_[ihist]]
                              : cal_->baseline_[apvId_[ihist]][stripId_[ihist]];
    Bspread[apvId_[ihist]] +=
        cal_->baseline_[apvId_[ihist]][stripId_[ihist]] * cal_->baseline_[apvId_[ihist]][stripId_[ihist]];
  }

  // make mean values
  for (int i = 0; i < 2; i++) {
    if (nStrips[i] != 0) {
      Amean[i] = Amean[i] / nStrips[i];
      Tmean[i] = Tmean[i] / nStrips[i];
      Rmean[i] = Rmean[i] / nStrips[i];
      Cmean[i] = Cmean[i] / nStrips[i];
      Omean[i] = Omean[i] / nStrips[i];
      Mmean[i] = Mmean[i] / nStrips[i];
      Umean[i] = Umean[i] / nStrips[i];
      Bmean[i] = Bmean[i] / nStrips[i];
      Smean[i] = Smean[i] / nStrips[i];
      Kmean[i] = Kmean[i] / nStrips[i];

      Aspread[i] = Aspread[i] / nStrips[i];
      Tspread[i] = Tspread[i] / nStrips[i];
      Rspread[i] = Rspread[i] / nStrips[i];
      Cspread[i] = Cspread[i] / nStrips[i];
      Ospread[i] = Ospread[i] / nStrips[i];
      Mspread[i] = Mspread[i] / nStrips[i];
      Uspread[i] = Uspread[i] / nStrips[i];
      Bspread[i] = Bspread[i] / nStrips[i];
      Sspread[i] = Sspread[i] / nStrips[i];
      Kspread[i] = Kspread[i] / nStrips[i];
    }
  }

  // fill the mean, max, min, spread, ... histograms.
  for (int i = 0; i < 2; ++i) {
    cal_->mean_amplitude_[i] = Amean[i];
    cal_->mean_tail_[i] = Tmean[i];
    cal_->mean_riseTime_[i] = Rmean[i];
    cal_->mean_decayTime_[i] = Cmean[i];
    cal_->mean_turnOn_[i] = Omean[i];
    cal_->mean_peakTime_[i] = Mmean[i];
    cal_->mean_undershoot_[i] = Umean[i];
    cal_->mean_baseline_[i] = Bmean[i];
    cal_->mean_smearing_[i] = Smean[i];
    cal_->mean_chi2_[i] = Kmean[i];

    cal_->min_amplitude_[i] = Amin[i];
    cal_->min_tail_[i] = Tmin[i];
    cal_->min_riseTime_[i] = Rmin[i];
    cal_->min_decayTime_[i] = Cmin[i];
    cal_->min_turnOn_[i] = Omin[i];
    cal_->min_peakTime_[i] = Mmin[i];
    cal_->min_undershoot_[i] = Umin[i];
    cal_->min_baseline_[i] = Bmin[i];
    cal_->min_smearing_[i] = Smin[i];
    cal_->min_chi2_[i] = Kmin[i];

    cal_->max_amplitude_[i] = Amax[i];
    cal_->max_tail_[i] = Tmax[i];
    cal_->max_riseTime_[i] = Rmax[i];
    cal_->max_decayTime_[i] = Cmax[i];
    cal_->max_turnOn_[i] = Omax[i];
    cal_->max_peakTime_[i] = Mmax[i];
    cal_->max_undershoot_[i] = Umax[i];
    cal_->max_baseline_[i] = Bmax[i];
    cal_->max_smearing_[i] = Smax[i];
    cal_->max_chi2_[i] = Kmax[i];

    cal_->spread_amplitude_[i] = sqrt(fabs(Aspread[i] - Amean[i] * Amean[i]));
    cal_->spread_tail_[i] = sqrt(fabs(Tspread[i] - Tmean[i] * Tmean[i]));
    cal_->spread_riseTime_[i] = sqrt(fabs(Rspread[i] - Rmean[i] * Rmean[i]));
    cal_->spread_decayTime_[i] = sqrt(fabs(Cspread[i] - Cmean[i] * Cmean[i]));
    cal_->spread_turnOn_[i] = sqrt(fabs(Ospread[i] - Omean[i] * Omean[i]));
    cal_->spread_peakTime_[i] = sqrt(fabs(Mspread[i] - Mmean[i] * Mmean[i]));
    cal_->spread_undershoot_[i] = sqrt(fabs(Uspread[i] - Umean[i] * Umean[i]));
    cal_->spread_baseline_[i] = sqrt(fabs(Bspread[i] - Bmean[i] * Bmean[i]));
    cal_->spread_smearing_[i] = sqrt(fabs(Sspread[i] - Smean[i] * Smean[i]));
    cal_->spread_chi2_[i] = sqrt(fabs(Kspread[i] - Kmean[i] * Kmean[i]));
  }

  if (fit_function)
    delete fit_function;
}

// ------
void CalibrationAlgorithm::correctDistribution(TH1* histo) const {
  // 20 events per point in the TM loop  --> divide by 20 to have the amplitude of a single event readout
  for (int iBin = 0; iBin < histo->GetNbinsX(); iBin++) {
    histo->SetBinContent(iBin + 1, -histo->GetBinContent(iBin + 1) / 20.);
  }
}

// ----------------------------------------------------------------------------
float CalibrationAlgorithm::baseLine(TF1* f) {
  float xmax = 10;
  float baseline = 0;
  int npoints = 0;
  float x = f->GetXmin();
  for (; x < xmax; x += 0.1) {
    baseline += f->Eval(x);
    npoints++;
  }
  return baseline / npoints;
}

// ----------------------------------------------------------------------------
float CalibrationAlgorithm::turnOn(TF1* f,
                                   const float& baseline) {  // should happen within 100 ns in both deco and peak modes
  float max_amplitude = f->GetMaximum();
  float time = 10.;
  for (; time < 100 && (f->Eval(time) - baseline) < 0.05 * (max_amplitude - baseline); time += 0.1) {
  }  // flucutation higher than 5% of the pulse height
  return time;
}

// ----------------------------------------------------------------------------
float CalibrationAlgorithm::decayTime(
    TF1* f) {  // if we approximate the decay to an exp(-t/tau), in one constant unit, the amplited is reduced by e^{-1}
  float xval = f->GetMaximumX();
  float max_amplitude = f->GetMaximum();
  float x = xval;
  for (; x < 1000; x = x + 0.1) {
    if (f->Eval(x) < max_amplitude * exp(-1))
      break;
  }
  return x;
}
