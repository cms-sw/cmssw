#include "DQM/SiStripCommissioningAnalysis/interface/CalibrationScanAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/CalibrationScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommissioningAnalysis/interface/SiStripPulseShape.h"
#include "TProfile.h"
#include "TF1.h"
#include "TH1.h"
#include "TVirtualFitter.h"
#include "TFitResult.h"
#include "TMath.h"
#include "TGraph2D.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TROOT.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include "Math/MinimizerOptions.h"

using namespace sistrip;

// ----------------------------------------------------------------------------
//
CalibrationScanAlgorithm::CalibrationScanAlgorithm(const edm::ParameterSet& pset, CalibrationScanAnalysis* const anal)
    : CommissioningAlgorithm(anal), cal_(nullptr) {}

// ----------------------------------------------------------------------------
//
void CalibrationScanAlgorithm::extract(const std::vector<TH1*>& histos) {
  if (!anal()) {
    edm::LogWarning(mlCommissioning_) << "[CalibrationScanAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to base Analysis object!";
    return;
  }

  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>(anal());
  cal_ = dynamic_cast<CalibrationScanAnalysis*>(tmp);
  if (!cal_) {
    edm::LogWarning(mlCommissioning_) << "[CalibrationScanAlgorithm::" << __func__ << "]"
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
    if (title.runType() != sistrip::CALIBRATION_SCAN && title.runType() != sistrip::CALIBRATION_SCAN_DECO) {
      cal_->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }

    /// extract isha, vfs and calchan values, as well as filling the histogram objects
    Histo histo_temp;
    histo_temp.first = *ihis;
    histo_temp.first->Sumw2();
    histo_temp.second = (*ihis)->GetTitle();
    histo_[title.extraInfo()].resize(2);
    if (title.channel() % 2 == 0)
      histo_[title.extraInfo()][0] = histo_temp;
    else
      histo_[title.extraInfo()][1] = histo_temp;
  }
}
// ----------------------------------------------------------------------------
//
void CalibrationScanAlgorithm::analyse() {
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Migrad");
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);

  if (!cal_) {
    edm::LogWarning(mlCommissioning_) << "[CalibrationScanAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to derived Analysis object!";
    return;
  }

  ////////
  TFitResultPtr fit_result;
  TF1* fit_function_turnOn = nullptr;
  TF1* fit_function_decay = nullptr;
  TF1* fit_function_deco = nullptr;
  if (cal_->deconv_) {
    fit_function_deco = new TF1("fit_function_deco", fdeconv, 0, 400, 7);
    fit_function_deco->SetParameters(4, 25, 25, 50, 250, 25, 0.75);
  } else {
    fit_function_turnOn = new TF1("fit_function_turnOn", fturnOn, 0, 400, 4);
    fit_function_decay = new TF1("fit_function_decay", fdecay, 0, 400, 3);
    fit_function_turnOn->SetParameters(50, 50, 40, 20);
    fit_function_decay->SetParameters(-150, -0.01, -0.1);
  }

  /// loop over histograms for this fiber
  for (auto map_element : histo_) {
    // add to the analysis result
    cal_->addOneCalibrationPoint(map_element.first);

    // stored as integer the scanned isha and vfs values
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(map_element.first);
    while (std::getline(tokenStream, token, '_')) {
      tokens.push_back(token);
    }

    scanned_isha_.push_back(std::stoi(tokens.at(1)));
    scanned_vfs_.push_back(std::stoi(tokens.at(3)));

    // loop on APVs
    for (size_t iapv = 0; iapv < 2; iapv++) {
      if (!map_element.second[iapv].first) {
        edm::LogWarning(mlCommissioning_)
            << " NULL pointer to histogram for: " << map_element.second[iapv].second << " !";
        return;
      }

      cal_->amplitude_[map_element.first][iapv] = 0;
      cal_->baseline_[map_element.first][iapv] = 0;
      cal_->riseTime_[map_element.first][iapv] = 0;
      cal_->turnOn_[map_element.first][iapv] = 0;
      cal_->peakTime_[map_element.first][iapv] = 0;
      cal_->undershoot_[map_element.first][iapv] = 0;
      cal_->tail_[map_element.first][iapv] = 0;
      cal_->decayTime_[map_element.first][iapv] = 0;
      cal_->smearing_[map_element.first][iapv] = 0;
      cal_->chi2_[map_element.first][iapv] = 0;
      cal_->isvalid_[map_element.first][iapv] = true;

      if (map_element.second[iapv].first->Integral() == 0) {
        cal_->isvalid_[map_element.first][iapv] = false;
        continue;
      }

      // rescale the plot
      correctDistribution(map_element.second[iapv].first, false);

      // from NOTE2009_021 : The charge injection provided by the calibration circuit is known with a precision of 5%;
      float error = (map_element.second[iapv].first->GetMaximum() * 0.05);
      for (int i = 1; i <= map_element.second[iapv].first->GetNbinsX(); ++i)
        map_element.second[iapv].first->SetBinError(i, error);

      /////
      if (cal_->deconv_) {  // deconvolution mode
        fit_function_deco->SetParameters(4, 25, 25, 50, 250, 25, 0.75);
        fit_result = map_element.second[iapv].first->Fit(fit_function_deco, "QRS");

        if (not fit_result.Get()) {
          cal_->isvalid_[map_element.first][iapv] = false;
          continue;
        }

        /// make the fit
        float maximum_ampl = fit_function_deco->GetMaximum();
        float peak_time = fit_function_deco->GetMaximumX();
        float baseline = baseLine(fit_function_deco);
        float turn_on_time = turnOn(fit_function_deco, baseline);
        float rise_time = peak_time - turn_on_time;

        // start filling info
        cal_->amplitude_[map_element.first][iapv] = maximum_ampl - baseline;
        cal_->baseline_[map_element.first][iapv] = baseline;
        cal_->riseTime_[map_element.first][iapv] = rise_time;
        cal_->turnOn_[map_element.first][iapv] = turn_on_time;
        cal_->peakTime_[map_element.first][iapv] = peak_time;
        if (fit_function_deco->GetMinimumX() > rise_time)
          cal_->undershoot_[map_element.first][iapv] =
              100 * (fit_function_deco->GetMinimum() - baseline) / (maximum_ampl - baseline);
        else
          cal_->undershoot_[map_element.first][iapv] = 0;

        // Bin related to peak + 125 ns
        int lastBin = map_element.second[iapv].first->FindBin(peak_time + 125);
        if (lastBin > map_element.second[iapv].first->GetNbinsX() - 4)
          lastBin = map_element.second[iapv].first->GetNbinsX() - 4;

        // tail is the amplitude at 5 bx from the maximum
        cal_->tail_[map_element.first][iapv] =
            100 * (map_element.second[iapv].first->GetBinContent(lastBin) - baseline) / (maximum_ampl - baseline);

        // reaches 1/e of the peak amplitude
        cal_->decayTime_[map_element.first][iapv] = decayTime(fit_function_deco) - peak_time;
        cal_->smearing_[map_element.first][iapv] = 0;
        cal_->chi2_[map_element.first][iapv] =
            fit_function_deco->GetChisquare() /
            (map_element.second[iapv].first->GetNbinsX() - fit_function_deco->GetNpar());

      } else {
        // peak mode
        fit_function_turnOn->SetParameters(50, 50, 40, 20);
        fit_function_turnOn->SetRange(fit_function_turnOn->GetXmin(),
                                      map_element.second[iapv].first->GetBinCenter(
                                          map_element.second[iapv].first->GetMaximumBin()));  // up to the maximum
        fit_result = map_element.second[iapv].first->Fit(fit_function_turnOn, "QSR");
        if (not fit_result.Get()) {
          cal_->isvalid_[map_element.first][iapv] = false;
          continue;
        }

        /// make the fit
        float maximum_ampl = fit_function_turnOn->GetMaximum();
        float peak_time = fit_function_turnOn->GetMaximumX();
        float baseline = baseLine(fit_function_turnOn);
        float turn_on_time = turnOn(fit_function_turnOn, baseline);
        float rise_time = peak_time - turn_on_time;

        // start filling info
        cal_->amplitude_[map_element.first][iapv] = maximum_ampl - baseline;
        cal_->baseline_[map_element.first][iapv] = baseline;
        cal_->riseTime_[map_element.first][iapv] = rise_time;
        cal_->turnOn_[map_element.first][iapv] = turn_on_time;
        cal_->peakTime_[map_element.first][iapv] = peak_time;

        fit_function_decay->SetParameters(-150, -0.01, -0.1);
        fit_function_decay->SetRange(
            map_element.second[iapv].first->GetBinCenter(map_element.second[iapv].first->GetMaximumBin()) + 10.,
            fit_function_decay->GetXmax());  // up to the maximum
        fit_result = map_element.second[iapv].first->Fit(fit_function_decay, "QSR+");

        if (fit_result.Get() and fit_result->Status() >= 4) {
          cal_->isvalid_[map_element.first][iapv] = false;
          continue;
        }

        cal_->undershoot_[map_element.first][iapv] = 0;

        // Bin related to peak + 125 ns
        int lastBin = map_element.second[iapv].first->FindBin(peak_time + 125);
        if (lastBin > map_element.second[iapv].first->GetNbinsX() - 4)
          lastBin = map_element.second[iapv].first->GetNbinsX() - 4;

        // tail is the amplitude at 5 bx from the maximum
        cal_->tail_[map_element.first][iapv] =
            100 * (map_element.second[iapv].first->GetBinContent(lastBin) - baseline) / (maximum_ampl - baseline);

        // reaches 1/e of the peak amplitude
        cal_->decayTime_[map_element.first][iapv] = decayTime(fit_function_decay) - peak_time;
        cal_->smearing_[map_element.first][iapv] = 0;
        cal_->chi2_[map_element.first][iapv] =
            (fit_function_turnOn->GetChisquare() + fit_function_decay->GetChisquare()) /
            (map_element.second[iapv].first->GetNbinsX() - fit_function_turnOn->GetNpar() -
             fit_function_decay->GetNpar());

        // apply quality requirements
        bool isvalid = true;
        if (cal_->amplitude_[map_element.first][iapv] < CalibrationScanAnalysis::minAmplitudeThreshold_)
          isvalid = false;
        if (cal_->baseline_[map_element.first][iapv] < CalibrationScanAnalysis::minBaselineThreshold_)
          isvalid = false;
        else if (cal_->baseline_[map_element.first][iapv] > CalibrationScanAnalysis::maxBaselineThreshold_)
          isvalid = false;
        if (cal_->decayTime_[map_element.first][iapv] < CalibrationScanAnalysis::minDecayTimeThreshold_)
          isvalid = false;
        else if (cal_->decayTime_[map_element.first][iapv] > CalibrationScanAnalysis::maxDecayTimeThreshold_)
          isvalid = false;
        if (cal_->peakTime_[map_element.first][iapv] < CalibrationScanAnalysis::minPeakTimeThreshold_)
          isvalid = false;
        else if (cal_->peakTime_[map_element.first][iapv] > CalibrationScanAnalysis::maxPeakTimeThreshold_)
          isvalid = false;
        if (cal_->riseTime_[map_element.first][iapv] < CalibrationScanAnalysis::minRiseTimeThreshold_)
          isvalid = false;
        else if (cal_->riseTime_[map_element.first][iapv] > CalibrationScanAnalysis::maxRiseTimeThreshold_)
          isvalid = false;
        if (cal_->turnOn_[map_element.first][iapv] < CalibrationScanAnalysis::minTurnOnThreshold_)
          isvalid = false;
        else if (cal_->turnOn_[map_element.first][iapv] > CalibrationScanAnalysis::maxTurnOnThreshold_)
          isvalid = false;
        if (cal_->chi2_[map_element.first][iapv] > CalibrationScanAnalysis::maxChi2Threshold_)
          isvalid = false;

        if (not isvalid) {
          cal_->amplitude_[map_element.first][iapv] = 0;
          cal_->baseline_[map_element.first][iapv] = 0;
          cal_->riseTime_[map_element.first][iapv] = 0;
          cal_->turnOn_[map_element.first][iapv] = 0;
          cal_->peakTime_[map_element.first][iapv] = 0;
          cal_->undershoot_[map_element.first][iapv] = 0;
          cal_->tail_[map_element.first][iapv] = 0;
          cal_->decayTime_[map_element.first][iapv] = 0;
          cal_->smearing_[map_element.first][iapv] = 0;
          cal_->chi2_[map_element.first][iapv] = 0;
          cal_->isvalid_[map_element.first][iapv] = false;
        }
      }
    }
  }

  if (fit_function_deco)
    delete fit_function_deco;
  if (fit_function_decay)
    delete fit_function_decay;
  if (fit_function_turnOn)
    delete fit_function_turnOn;
}

// ------
void CalibrationScanAlgorithm::correctDistribution(TH1* histo, const bool& isShape) const {
  // 5 events per point in the TM loop
  // total signal is obtained by summing 16 strips of the same calChan
  if (not isShape) {
    for (int iBin = 0; iBin < histo->GetNbinsX(); iBin++) {
      histo->SetBinContent(iBin + 1, -histo->GetBinContent(iBin + 1) / 16.);
      histo->SetBinContent(iBin + 1, histo->GetBinContent(iBin + 1) / 5.);
    }
  } else
    histo->Scale(1. / histo->Integral());
}

// ----------------------------------------------------------------------------
float CalibrationScanAlgorithm::baseLine(TF1* f) {
  float x = f->GetXmin();
  float xmax = 10;
  float baseline = 0;
  int npoints = 0;
  for (; x < xmax; x += 0.1) {
    baseline += f->Eval(x);
    npoints++;
  }
  return baseline / npoints;
}

// ----------------------------------------------------------------------------
float CalibrationScanAlgorithm::turnOn(
    TF1* f, const float& baseline) {  // should happen within 100 ns in both deco and peak modes
  float max_amplitude = f->GetMaximum();
  float time = 10.;
  for (; time < 100 && (f->Eval(time) - baseline) < 0.05 * (max_amplitude - baseline); time += 0.1) {
  }  // flucutation higher than 5% of the pulse height
  return time;
}

// ----------------------------------------------------------------------------
float CalibrationScanAlgorithm::decayTime(
    TF1* f) {  // if we approximate the decay to an exp(-t/tau), in one constant unit, the amplited is reduced by e^{-1}
  float xval = std::max(f->GetXmin(), f->GetMaximumX());
  float max_amplitude = f->GetMaximum();
  float x = xval;
  for (; x < 1000;
       x = x +
           0.1) {  // 1000 is a reasoable large bound to compute the decay time .. in case the function is bad it is useful to break the loop
    if (f->Eval(x) < max_amplitude * exp(-1))
      break;
  }
  return x;
}

// --- function to extract the VFS value corresponding to decay time of 125ns, then ISHA close to 50 ns
void CalibrationScanAlgorithm::tuneIndependently(const int& iapv,
                                                 const float& targetRiseTime,
                                                 const float& targetDecayTime) {
  std::map<int, std::vector<float> > decayTime_vs_vfs;
  TString name;
  int imap = 0;

  for (auto map_element : histo_) {
    // only consider isha values in the middle of the scanned range
    if (scanned_isha_.at(imap) <= CalibrationScanAnalysis::minISHAforVFSTune_ or
        scanned_isha_.at(imap) >= CalibrationScanAnalysis::maxISHAforVFSTune_) {
      imap++;
      continue;
    }

    if (cal_->isValid(map_element.first)[iapv])
      decayTime_vs_vfs[scanned_vfs_.at(imap)].push_back(cal_->decayTime(map_element.first)[iapv]);

    if (name == "") {  // store the base name
      name = Form("%s", map_element.second[iapv].first->GetName());
      name.ReplaceAll("_" + map_element.first, "");
    }
    imap++;
  }

  // sort before taking the median
  for (auto iter : decayTime_vs_vfs)
    sort(iter.second.begin(), iter.second.end());

  name.ReplaceAll("ExpertHisto_", "");

  // transform the dependance vs vfs in graph
  cal_->decayTime_vs_vfs_.push_back(new TGraph());
  cal_->decayTime_vs_vfs_.back()->SetName(Form("decayTime_%s", name.Data()));

  // transform the dependance vs isha in graph
  cal_->riseTime_vs_isha_.push_back(new TGraph());
  cal_->riseTime_vs_isha_.back()->SetName(Form("riseTime_%s", name.Data()));

  if (!decayTime_vs_vfs.empty()) {
    int ipoint = 0;
    for (auto map_element : decayTime_vs_vfs) {
      if (!map_element.second.empty()) {
        cal_->decayTime_vs_vfs_.at(iapv)->SetPoint(
            ipoint, map_element.second.at(round(map_element.second.size() / 2)), map_element.first);
        ipoint++;
      }
    }

    double max_apv =
        TMath::MaxElement(cal_->decayTime_vs_vfs_.at(iapv)->GetN(), cal_->decayTime_vs_vfs_.at(iapv)->GetY());
    double min_apv =
        TMath::MinElement(cal_->decayTime_vs_vfs_.at(iapv)->GetN(), cal_->decayTime_vs_vfs_.at(iapv)->GetY());

    cal_->vfs_[iapv] = cal_->decayTime_vs_vfs_.at(iapv)->Eval(targetDecayTime);

    // avoid extrapolations
    if (cal_->vfs_[iapv] < min_apv)
      cal_->vfs_[iapv] = min_apv;
    else if (cal_->vfs_[iapv] > max_apv)
      cal_->vfs_[iapv] = max_apv;

    // value for each isha but different ISHA
    std::map<int, std::vector<float> > riseTime_vs_isha;
    imap = 0;
    // store for each isha value all rise time (changing isha)
    for (auto map_element : histo_) {
      if (fabs(scanned_vfs_.at(imap) - cal_->vfs_[iapv]) < CalibrationScanAnalysis::VFSrange_ and
          cal_->isValid(map_element.first)[iapv])  //around chosen VFS by \pm 20
        riseTime_vs_isha[scanned_isha_.at(imap)].push_back(cal_->riseTime(map_element.first)[iapv]);
      if (name == "") {
        name = Form("%s", map_element.second[iapv].first->GetName());
        name.ReplaceAll("_" + map_element.first, "");
      }
      imap++;
    }

    // sort before taking the median
    for (auto iter : riseTime_vs_isha)
      sort(iter.second.begin(), iter.second.end());
    name.ReplaceAll("ExpertHisto_", "");

    ////
    if (!riseTime_vs_isha.empty()) {
      int ipoint = 0;
      for (auto map_element : riseTime_vs_isha) {
        if (!map_element.second.empty()) {
          cal_->riseTime_vs_isha_.at(iapv)->SetPoint(
              ipoint, map_element.second.at(round(map_element.second.size() / 2)), map_element.first);
          ipoint++;
        }
      }

      double max_apv =
          TMath::MaxElement(cal_->riseTime_vs_isha_.at(iapv)->GetN(), cal_->riseTime_vs_isha_.at(iapv)->GetY());
      double min_apv =
          TMath::MinElement(cal_->riseTime_vs_isha_.at(iapv)->GetN(), cal_->riseTime_vs_isha_.at(iapv)->GetY());

      cal_->isha_[iapv] = cal_->riseTime_vs_isha_.at(iapv)->Eval(targetRiseTime);

      if (cal_->isha_[iapv] < min_apv)
        cal_->isha_[iapv] = min_apv;
      else if (cal_->isha_[iapv] > max_apv)
        cal_->isha_[iapv] = max_apv;
    } else
      cal_->isha_[iapv] = -1;
  }
}

////////////// Simultaneously tune isha and vfs
void CalibrationScanAlgorithm::tuneSimultaneously(const int& iapv,
                                                  const float& targetRiseTime,
                                                  const float& targetDecayTime) {
  // Build 2D graph for each APV with rise and decay time trend vs ISHA and VFS
  cal_->decayTime_vs_isha_vfs_.push_back(new TGraph2D());
  cal_->riseTime_vs_isha_vfs_.push_back(new TGraph2D());

  // store for each vfs value all decay time (changing vfs)
  TString name_apv;
  int ipoint_apv = 0;
  int imap = 0;

  for (auto map_element : histo_) {
    if (cal_->isValid(map_element.first)[iapv]) {
      cal_->decayTime_vs_isha_vfs_.at(iapv)->SetPoint(
          ipoint_apv, scanned_isha_.at(imap), scanned_vfs_.at(imap), cal_->decayTime(map_element.first)[iapv]);
      cal_->riseTime_vs_isha_vfs_.at(iapv)->SetPoint(
          ipoint_apv, scanned_isha_.at(imap), scanned_vfs_.at(imap), cal_->riseTime(map_element.first)[iapv]);
      ipoint_apv++;
    }
    if (name_apv == "") {  // store the base name
      name_apv = Form("%s", map_element.second[iapv].first->GetName());
      name_apv.ReplaceAll("_" + map_element.first, "");
    }
    imap++;
  }

  name_apv.ReplaceAll("ExpertHisto_", "");

  cal_->decayTime_vs_isha_vfs_.at(iapv)->SetName(Form("decayTime_%s", name_apv.Data()));
  cal_->riseTime_vs_isha_vfs_.at(iapv)->SetName(Form("riseTime_%s", name_apv.Data()));

  // Define 2D histogram for the distance between values and target
  TH2F* hist_decay_apv = new TH2F("hist_decay_apv",
                                  "hist_decay_apv",
                                  500,
                                  *min_element(scanned_isha_.begin(), scanned_isha_.end()),
                                  *max_element(scanned_isha_.begin(), scanned_isha_.end()),
                                  500,
                                  *min_element(scanned_vfs_.begin(), scanned_vfs_.end()),
                                  *max_element(scanned_vfs_.begin(), scanned_vfs_.end()));

  TH2F* hist_rise_apv = (TH2F*)hist_decay_apv->Clone();
  hist_rise_apv->SetName("hist_rise_apv");
  hist_rise_apv->Reset();

  TH2F* hist_distance = (TH2F*)hist_decay_apv->Clone();
  hist_distance->SetName("hist_distance");
  hist_distance->Reset();

  for (int iBin = 1; iBin <= hist_decay_apv->GetNbinsX(); iBin++) {
    for (int jBin = 1; jBin <= hist_decay_apv->GetNbinsY(); jBin++) {
      if (ipoint_apv != 0) {
        if (cal_->decayTime_vs_isha_vfs_.at(iapv)->GetN() > 10)  // to make sure the interpolation can work
          hist_decay_apv->SetBinContent(
              iBin,
              jBin,
              cal_->decayTime_vs_isha_vfs_.at(iapv)->Interpolate(hist_decay_apv->GetXaxis()->GetBinCenter(iBin),
                                                                 hist_decay_apv->GetYaxis()->GetBinCenter(jBin)));
        if (cal_->riseTime_vs_isha_vfs_.at(iapv)->GetN() > 10)
          hist_rise_apv->SetBinContent(
              iBin,
              jBin,
              cal_->riseTime_vs_isha_vfs_.at(iapv)->Interpolate(hist_rise_apv->GetXaxis()->GetBinCenter(iBin),
                                                                hist_rise_apv->GetYaxis()->GetBinCenter(jBin)));
      }
    }
  }

  // further smoothing --> a smooth behaviour is indeed expected
  hist_decay_apv->Smooth();
  hist_rise_apv->Smooth();

  for (int iBin = 1; iBin <= hist_decay_apv->GetNbinsX(); iBin++) {
    for (int jBin = 1; jBin <= hist_decay_apv->GetNbinsY(); jBin++) {
      hist_distance->SetBinContent(
          iBin,
          jBin,
          sqrt(pow((hist_decay_apv->GetBinContent(iBin, jBin) - targetDecayTime) / targetDecayTime, 2) +
               pow((hist_rise_apv->GetBinContent(iBin, jBin) - targetRiseTime) / targetRiseTime, 2)));
    }
  }

  int minx, miny, minz;
  hist_distance->GetMinimumBin(minx, miny, minz);

  cal_->isha_[iapv] = round(hist_distance->GetXaxis()->GetBinCenter(minx));
  cal_->vfs_[iapv] = round(hist_distance->GetYaxis()->GetBinCenter(miny));

  delete hist_decay_apv;
  delete hist_rise_apv;
  delete hist_distance;
}

void CalibrationScanAlgorithm::fillTunedObservables(const int& apvid) {
  // find the closest isha and vfs for each APV
  int distance_apv = 10000;

  // find close by ISHA
  for (size_t i = 0; i < scanned_isha_.size(); i++) {
    if (fabs(scanned_isha_.at(i) - cal_->bestISHA().at(apvid)) < distance_apv) {
      distance_apv = fabs(scanned_isha_.at(i) - cal_->bestISHA().at(apvid));
      cal_->tunedISHA_.at(apvid) = scanned_isha_.at(i);
    }
  }

  distance_apv = 10000;

  // find close by VFS
  for (size_t i = 0; i < scanned_vfs_.size(); i++) {
    if (fabs(scanned_vfs_.at(i) - cal_->bestVFS().at(apvid)) < distance_apv) {
      distance_apv = fabs(scanned_vfs_.at(i) - cal_->bestVFS().at(apvid));
      cal_->tunedVFS_.at(apvid) = scanned_vfs_.at(i);
    }
  }

  ///
  std::string key_apv = std::string(Form("isha_%d_vfs_%d", cal_->tunedISHA().at(apvid), cal_->tunedVFS().at(apvid)));
  if (!cal_->amplitude(key_apv).empty()) {
    cal_->tunedAmplitude_[apvid] = cal_->amplitude(key_apv)[apvid];
    cal_->tunedTail_[apvid] = cal_->tail(key_apv)[apvid];
    cal_->tunedRiseTime_[apvid] = cal_->riseTime(key_apv)[apvid];
    cal_->tunedDecayTime_[apvid] = cal_->decayTime(key_apv)[apvid];
    cal_->tunedTurnOn_[apvid] = cal_->turnOn(key_apv)[apvid];
    cal_->tunedPeakTime_[apvid] = cal_->peakTime(key_apv)[apvid];
    cal_->tunedUndershoot_[apvid] = cal_->undershoot(key_apv)[apvid];
    cal_->tunedBaseline_[apvid] = cal_->baseline(key_apv)[apvid];
    cal_->tunedSmearing_[apvid] = cal_->smearing(key_apv)[apvid];
    cal_->tunedChi2_[apvid] = cal_->chi2(key_apv)[apvid];
  } else {
    cal_->tunedAmplitude_[apvid] = 0;
    cal_->tunedTail_[apvid] = 0;
    cal_->tunedRiseTime_[apvid] = 0;
    cal_->tunedDecayTime_[apvid] = 0;
    cal_->tunedTurnOn_[apvid] = 0;
    cal_->tunedPeakTime_[apvid] = 0;
    cal_->tunedUndershoot_[apvid] = 0;
    cal_->tunedBaseline_[apvid] = 0;
    cal_->tunedSmearing_[apvid] = 0;
    cal_->tunedChi2_[apvid] = 0;
  }
}
