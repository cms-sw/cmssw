#include <climits>
#include <cmath>
#include <iostream>
#include <memory>

#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFitOOTPileupCorrection.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"

PulseShapeFitOOTPileupCorrection::PulseShapeFitOOTPileupCorrection()
    : cntsetPulseShape(0),
      psfPtr_(nullptr),
      spfunctor_(nullptr),
      dpfunctor_(nullptr),
      tpfunctor_(nullptr),
      TSMin_(0),
      TSMax_(0),
      vts4Chi2_(0),
      pedestalConstraint_(false),
      timeConstraint_(false),
      addPulseJitter_(false),
      applyTimeSlew_(false),
      ts4Min_(0),
      vts4Max_(0),
      pulseJitter_(0),
      timeMean_(0),
      timeSig_(0),
      pedMean_(0) {
  hybridfitter = new PSFitter::HybridMinimizer(PSFitter::HybridMinimizer::kMigrad);
  iniTimesArr = {{-100, -75, -50, -25, 0, 25, 50, 75, 100, 125}};
}

PulseShapeFitOOTPileupCorrection::~PulseShapeFitOOTPileupCorrection() {
  if (hybridfitter)
    delete hybridfitter;
}

void PulseShapeFitOOTPileupCorrection::setPUParams(bool iPedestalConstraint,
                                                   bool iTimeConstraint,
                                                   bool iAddPulseJitter,
                                                   bool iApplyTimeSlew,
                                                   double iTS4Min,
                                                   const std::vector<double> &iTS4Max,
                                                   double iPulseJitter,
                                                   double iTimeMean,
                                                   double iTimeSigHPD,
                                                   double iTimeSigSiPM,
                                                   double iPedMean,
                                                   double iTMin,
                                                   double iTMax,
                                                   const std::vector<double> &its4Chi2,
                                                   HcalTimeSlew::BiasSetting slewFlavor,
                                                   int iFitTimes) {
  TSMin_ = iTMin;
  TSMax_ = iTMax;
  //  ts4Chi2_   = its4Chi2;
  vts4Chi2_ = its4Chi2;
  pedestalConstraint_ = iPedestalConstraint;
  timeConstraint_ = iTimeConstraint;
  addPulseJitter_ = iAddPulseJitter;
  applyTimeSlew_ = iApplyTimeSlew;
  ts4Min_ = iTS4Min;
  //  ts4Max_            = iTS4Max;
  vts4Max_ = iTS4Max;
  pulseJitter_ = iPulseJitter * iPulseJitter;
  timeMean_ = iTimeMean;
  //  timeSig_            = iTimeSig;
  timeSigHPD_ = iTimeSigHPD;
  timeSigSiPM_ = iTimeSigSiPM;
  pedMean_ = iPedMean;
  slewFlavor_ = slewFlavor;
  fitTimes_ = iFitTimes;
}

void PulseShapeFitOOTPileupCorrection::setPulseShapeTemplate(const HcalPulseShapes::Shape &ps,
                                                             bool isHPD,
                                                             unsigned nSamples,
                                                             const HcalTimeSlew *hcalTimeSlewDelay) {
  // initialize for every different channel types (HPD vs SiPM)

  if (!(&ps == currentPulseShape_ && isHPD == isCurrentChannelHPD_)) {
    resetPulseShapeTemplate(ps, nSamples);

    // redefine the inverttimeSig2
    if (!isHPD)
      psfPtr_->setinverttimeSig2(1. / (timeSigSiPM_ * timeSigSiPM_));
    else
      psfPtr_->setinverttimeSig2(1. / (timeSigHPD_ * timeSigHPD_));

    currentPulseShape_ = &ps;
    isCurrentChannelHPD_ = isHPD;

    hcalTimeSlewDelay_ = hcalTimeSlewDelay;
    tsDelay1GeV_ = hcalTimeSlewDelay->delay(1.0, slewFlavor_);
  }
}

void PulseShapeFitOOTPileupCorrection::resetPulseShapeTemplate(const HcalPulseShapes::Shape &ps, unsigned nSamples) {
  ++cntsetPulseShape;
  psfPtr_ = std::make_unique<FitterFuncs::PulseShapeFunctor>(
      ps, pedestalConstraint_, timeConstraint_, addPulseJitter_, pulseJitter_, timeMean_, pedMean_, nSamples);
  spfunctor_ =
      std::make_unique<ROOT::Math::Functor>(psfPtr_.get(), &FitterFuncs::PulseShapeFunctor::singlePulseShapeFunc, 3);
  dpfunctor_ =
      std::make_unique<ROOT::Math::Functor>(psfPtr_.get(), &FitterFuncs::PulseShapeFunctor::doublePulseShapeFunc, 5);
  tpfunctor_ =
      std::make_unique<ROOT::Math::Functor>(psfPtr_.get(), &FitterFuncs::PulseShapeFunctor::triplePulseShapeFunc, 7);
}

constexpr char const *varNames[] = {"time", "energy", "time1", "energy1", "time2", "energy2", "ped"};

int PulseShapeFitOOTPileupCorrection::pulseShapeFit(const double *energyArr,
                                                    const double *pedenArr,
                                                    const double *chargeArr,
                                                    const double *pedArr,
                                                    const double *gainArr,
                                                    const double tsTOTen,
                                                    std::vector<float> &fitParsVec,
                                                    const double *noiseArrSq,
                                                    unsigned int soi) const {
  double tsMAX = 0;
  double tmpx[hcal::constants::maxSamples], tmpy[hcal::constants::maxSamples], tmperry[hcal::constants::maxSamples],
      tmperry2[hcal::constants::maxSamples], tmpslew[hcal::constants::maxSamples];
  double tstrig = 0;  // in fC
  for (unsigned int i = 0; i < hcal::constants::maxSamples; ++i) {
    tmpx[i] = i;
    tmpy[i] = energyArr[i] - pedenArr[i];
    //Add Time Slew !!! does this need to be pedestal subtracted
    tmpslew[i] = 0;
    if (applyTimeSlew_) {
      if (chargeArr[i] <= 1.0)
        tmpslew[i] = tsDelay1GeV_;
      else
        tmpslew[i] = hcalTimeSlewDelay_->delay(chargeArr[i], slewFlavor_);
    }

    // add the noise components
    tmperry2[i] = noiseArrSq[i];

    //Propagate it through
    tmperry2[i] *= (gainArr[i] * gainArr[i]);  //Convert from fC to GeV
    tmperry[i] = sqrt(tmperry2[i]);

    if (std::abs(energyArr[i]) > tsMAX)
      tsMAX = std::abs(tmpy[i]);
    if (i == soi || i == (soi + 1)) {
      tstrig += chargeArr[i] - pedArr[i];
    }
  }
  psfPtr_->setpsFitx(tmpx);
  psfPtr_->setpsFity(tmpy);
  psfPtr_->setpsFiterry(tmperry);
  psfPtr_->setpsFiterry2(tmperry2);
  psfPtr_->setpsFitslew(tmpslew);

  //Fit 1 single pulse
  float timevalfit = 0;
  float chargevalfit = 0;
  float pedvalfit = 0;
  float chi2 = 999;  //cannot be zero
  bool fitStatus = false;
  bool useTriple = false;

  unsigned BX[3] = {soi, soi + 1, soi - 1};
  if (ts4Chi2_ != 0)
    fit(1, timevalfit, chargevalfit, pedvalfit, chi2, fitStatus, tsMAX, tsTOTen, tmpy, BX);
  // Based on the pulse shape ( 2. likely gives the same performance )
  if (tmpy[soi - 2] > 3. * tmpy[soi - 1])
    BX[2] = soi - 2;
  // Only do three-pulse fit when tstrig < ts4Max_, otherwise one-pulse fit is used (above)
  if (chi2 > ts4Chi2_ && tstrig < ts4Max_) {  //fails chi2 cut goes straight to 3 Pulse fit
    fit(3, timevalfit, chargevalfit, pedvalfit, chi2, fitStatus, tsMAX, tsTOTen, tmpy, BX);
    useTriple = true;
  }

  timevalfit -= (int(soi) - hcal::constants::shiftTS) * hcal::constants::nsPerBX;

  /*
   if(chi2 > ts345Chi2_)   { //fails do two pulse chi2 for TS5 
     BX[1] = 5;
     fit(3,timevalfit,chargevalfit,pedvalfit,chi2,fitStatus,tsMAX,tsTOTen,BX);
   }
   */
  //Fix back the timeslew
  //if(applyTimeSlew_) timevalfit+=HcalTimeSlew::delay(std::max(1.0,chargeArr[4]),slewFlavor_);
  int outfitStatus = (fitStatus ? 1 : 0);
  fitParsVec.clear();
  fitParsVec.push_back(chargevalfit);
  fitParsVec.push_back(timevalfit);
  fitParsVec.push_back(pedvalfit);
  fitParsVec.push_back(chi2);
  fitParsVec.push_back(useTriple);
  return outfitStatus;
}

void PulseShapeFitOOTPileupCorrection::fit(int iFit,
                                           float &timevalfit,
                                           float &chargevalfit,
                                           float &pedvalfit,
                                           float &chi2,
                                           bool &fitStatus,
                                           double &iTSMax,
                                           const double &iTSTOTEn,
                                           double *iEnArr,
                                           unsigned (&iBX)[3]) const {
  int n = 3;
  if (iFit == 2)
    n = 5;  //Two   Pulse Fit
  if (iFit == 3)
    n = 7;                //Three Pulse Fit
                          //Step 1 Single Pulse fit
  float pedMax = iTSMax;  //=> max timeslice
  float tMin = TSMin_;    //Fitting Time Min
  float tMax = TSMax_;    //Fitting Time Max
  //Checks to make sure fitting happens
  if (pedMax < 1.)
    pedMax = 1.;
  // Set starting values andf step sizes for parameters
  double vstart[n];
  for (int i = 0; i < int((n - 1) / 2); i++) {
    vstart[2 * i + 0] = iniTimesArr[iBX[i]] + timeMean_;
    vstart[2 * i + 1] = iEnArr[iBX[i]];
  }
  vstart[n - 1] = pedMean_;

  double step[n];
  for (int i = 0; i < n; i++)
    step[i] = 0.1;

  if (iFit == 1)
    hybridfitter->SetFunction(*spfunctor_);
  if (iFit == 2)
    hybridfitter->SetFunction(*dpfunctor_);
  if (iFit == 3)
    hybridfitter->SetFunction(*tpfunctor_);
  hybridfitter->Clear();
  //Times and amplitudes
  for (int i = 0; i < int((n - 1) / 2); i++) {
    hybridfitter->SetLimitedVariable(0 + i * 2,
                                     varNames[2 * i + 0],
                                     vstart[0 + i * 2],
                                     step[0 + i * 2],
                                     iniTimesArr[iBX[i]] + tMin,
                                     iniTimesArr[iBX[i]] + tMax);
    hybridfitter->SetLimitedVariable(
        1 + i * 2, varNames[2 * i + 1], vstart[1 + i * 2], step[1 + i * 2], 0, 1.2 * iTSTOTEn);
    //Secret Option to fix the time
    if (timeSig_ < 0)
      hybridfitter->SetFixedVariable(0 + i * 2, varNames[2 * i + 0], vstart[0 + i * 2]);
  }
  //Pedestal
  if (vstart[n - 1] > std::abs(pedMax))
    vstart[n - 1] = pedMax;
  hybridfitter->SetLimitedVariable(n - 1, varNames[n - 1], vstart[n - 1], step[n - 1], -pedMax, pedMax);
  //Secret Option to fix the pedestal
  if (pedSig_ < 0)
    hybridfitter->SetFixedVariable(n - 1, varNames[n - 1], vstart[n - 1]);
  //a special number to label the initial condition
  chi2 = -1;
  //3 fits why?!
  const double *results = nullptr;
  for (int tries = 0; tries <= 3; ++tries) {
    if (fitTimes_ != 2 || tries != 1) {
      hybridfitter->SetMinimizerType(PSFitter::HybridMinimizer::kMigrad);
      fitStatus = hybridfitter->Minimize();
    }
    double chi2valfit = hybridfitter->MinValue();
    const double *newresults = hybridfitter->X();
    if (chi2 == -1 || chi2 > chi2valfit + 0.01) {
      results = newresults;
      chi2 = chi2valfit;
      if (tries == 0 && fitTimes_ == 1)
        break;
      if (tries == 1 && (fitTimes_ == 2 || fitTimes_ == 3))
        break;
      if (tries == 2 && fitTimes_ == 4)
        break;
      if (tries == 3 && fitTimes_ == 5)
        break;
      //Secret option to speed up the fit => perhaps we should drop this
      if (timeSig_ < 0 || pedSig_ < 0)
        break;
      if (tries == 0) {
        hybridfitter->SetMinimizerType(PSFitter::HybridMinimizer::kScan);
        fitStatus = hybridfitter->Minimize();
      } else if (tries == 1) {
        hybridfitter->SetStrategy(1);
      } else if (tries == 2) {
        hybridfitter->SetStrategy(2);
      }
    } else {
      break;
    }
  }
  assert(results);

  timevalfit = results[0];
  chargevalfit = results[1];
  pedvalfit = results[n - 1];
}

void PulseShapeFitOOTPileupCorrection::phase1Apply(const HBHEChannelInfo &channelData,
                                                   float &reconstructedEnergy,
                                                   float &reconstructedTime,
                                                   bool &useTriple,
                                                   float &chi2) const {
  psfPtr_->setDefaultcntNANinfit();

  const unsigned cssize = channelData.nSamples();
  const unsigned int soi = channelData.soi();

  // initialize arrays to be zero
  double chargeArr[hcal::constants::maxSamples] = {}, pedArr[hcal::constants::maxSamples] = {},
         gainArr[hcal::constants::maxSamples] = {};
  double energyArr[hcal::constants::maxSamples] = {}, pedenArr[hcal::constants::maxSamples] = {};
  double noiseADCArr[hcal::constants::maxSamples] = {};
  double noiseArrSq[hcal::constants::maxSamples] = {};
  double noisePHArr[hcal::constants::maxSamples] = {};
  double tsTOT = 0, tstrig = 0;  // in fC
  double tsTOTen = 0;            // in GeV

  // go over the time slices
  for (unsigned int ip = 0; ip < cssize; ++ip) {
    if (ip >= (unsigned)hcal::constants::maxSamples)
      continue;  // Too many samples than what we wanna fit (10 is enough...) -> skip them

    //      const int capid = channelData.capid(); // not needed
    double charge = channelData.tsRawCharge(ip);
    double ped = channelData.tsPedestal(ip);
    double gain = channelData.tsGain(ip);

    double energy = charge * gain;
    double peden = ped * gain;

    chargeArr[ip] = charge;
    pedArr[ip] = ped;
    gainArr[ip] = gain;
    energyArr[ip] = energy;
    pedenArr[ip] = peden;

    // quantization noise from the ADC (QIE8 or QIE10/11)
    noiseADCArr[ip] = (1. / sqrt(12)) * channelData.tsDFcPerADC(ip);

    // Photo statistics uncertainties
    //      sigmaFC/FC = 1/sqrt(Ne);
    // Note2. (from kPedro): the output number of photoelectrons after smearing is treated very differently for SiPMs: *each* pe is assigned a different time based on a random generation from the Y11 pulse plus the SimHit time. In HPDs, the overall pulse is shaped all at once using just the SimHit time.

    noisePHArr[ip] = 0;
    if ((charge - ped) > channelData.tsPedestalWidth(ip)) {
      noisePHArr[ip] = sqrt((charge - ped) * channelData.fcByPE());
    }

    // sum all in quadrature
    noiseArrSq[ip] = noiseADCArr[ip] * noiseADCArr[ip] +
                     channelData.tsPedestalWidth(ip) * channelData.tsPedestalWidth(ip) +
                     noisePHArr[ip] * noisePHArr[ip];

    tsTOT += charge - ped;
    tsTOTen += energy - peden;
    if (ip == soi || ip == soi + 1) {
      tstrig += charge - ped;
    }
  }

  double averagePedSig2GeV =
      0.25 *
      (channelData.tsPedestalWidth(0) * channelData.tsPedestalWidth(0) * channelData.tsGain(0) * channelData.tsGain(0) +
       channelData.tsPedestalWidth(1) * channelData.tsPedestalWidth(1) * channelData.tsGain(1) * channelData.tsGain(1) +
       channelData.tsPedestalWidth(2) * channelData.tsPedestalWidth(2) * channelData.tsGain(2) * channelData.tsGain(2) +
       channelData.tsPedestalWidth(3) * channelData.tsPedestalWidth(3) * channelData.tsGain(3) * channelData.tsGain(3));

  // redefine the invertpedSig2
  psfPtr_->setinvertpedSig2(1. / (averagePedSig2GeV));

  if (channelData.hasTimeInfo()) {
    ts4Chi2_ = vts4Chi2_[1];
    if (channelData.id().depth() == 1)
      ts4Max_ = vts4Max_[1];
    else
      ts4Max_ = vts4Max_[2];

  } else {
    ts4Max_ = vts4Max_[0];
    ts4Chi2_ = vts4Chi2_[0];
  }

  std::vector<float> fitParsVec;
  if (tstrig >= ts4Min_ && tsTOTen > 0.) {  //Two sigma from 0
    pulseShapeFit(energyArr, pedenArr, chargeArr, pedArr, gainArr, tsTOTen, fitParsVec, noiseArrSq, channelData.soi());
  } else {
    fitParsVec.clear();
    fitParsVec.push_back(0.);     //charge
    fitParsVec.push_back(-9999);  // time
    fitParsVec.push_back(0.);     // ped
    fitParsVec.push_back(-9999);  // chi2
    fitParsVec.push_back(false);  // triple
  }

  reconstructedEnergy = fitParsVec[0];
  reconstructedTime = fitParsVec[1];
  chi2 = fitParsVec[3];
  useTriple = fitParsVec[4];
}
