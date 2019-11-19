#define EIGEN_NO_DEBUG  // kill throws in eigen code
#include "RecoLocalCalo/HcalRecAlgos/interface/MahiFit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

MahiFit::MahiFit() {}

void MahiFit::setParameters(bool iDynamicPed,
                            double iTS4Thresh,
                            double chiSqSwitch,
                            bool iApplyTimeSlew,
                            HcalTimeSlew::BiasSetting slewFlavor,
                            bool iCalculateArrivalTime,
                            double iMeanTime,
                            double iTimeSigmaHPD,
                            double iTimeSigmaSiPM,
                            const std::vector<int>& iActiveBXs,
                            int iNMaxItersMin,
                            int iNMaxItersNNLS,
                            double iDeltaChiSqThresh,
                            double iNnlsThresh) {
  dynamicPed_ = iDynamicPed;

  ts4Thresh_ = iTS4Thresh;
  chiSqSwitch_ = chiSqSwitch;

  applyTimeSlew_ = iApplyTimeSlew;
  slewFlavor_ = slewFlavor;

  calculateArrivalTime_ = iCalculateArrivalTime;
  meanTime_ = iMeanTime;
  timeSigmaHPD_ = iTimeSigmaHPD;
  timeSigmaSiPM_ = iTimeSigmaSiPM;

  activeBXs_ = iActiveBXs;

  nMaxItersMin_ = iNMaxItersMin;
  nMaxItersNNLS_ = iNMaxItersNNLS;

  deltaChiSqThresh_ = iDeltaChiSqThresh;
  nnlsThresh_ = iNnlsThresh;

  bxOffsetConf_ = -(*std::min_element(activeBXs_.begin(), activeBXs_.end()));
  bxSizeConf_ = activeBXs_.size();
}

void MahiFit::phase1Apply(const HBHEChannelInfo& channelData,
                          float& reconstructedEnergy,
                          float& reconstructedTime,
                          bool& useTriple,
                          float& chi2) const {
  assert(channelData.nSamples() == 8 || channelData.nSamples() == 10);

  resetWorkspace();

  nnlsWork_.tsOffset = channelData.soi();

  std::array<float, 3> reconstructedVals{{0.0f, -9999.f, -9999.f}};

  double tsTOT = 0, tstrig = 0;  // in GeV
  for (unsigned int iTS = 0; iTS < nnlsWork_.tsSize; ++iTS) {
    auto const amplitude = channelData.tsRawCharge(iTS) - channelData.tsPedestal(iTS);

    nnlsWork_.amplitudes.coeffRef(iTS) = amplitude;

    //ADC granularity
    auto const noiseADC = norm_ * channelData.tsDFcPerADC(iTS);

    //Electronic pedestal
    auto const pedWidth = channelData.tsPedestalWidth(iTS);

    //Photostatistics
    auto const noisePhoto = (amplitude > pedWidth) ? std::sqrt(amplitude * channelData.fcByPE()) : 0.f;

    //Total uncertainty from all sources
    nnlsWork_.noiseTerms.coeffRef(iTS) = noiseADC * noiseADC + noisePhoto * noisePhoto + pedWidth * pedWidth;

    tsTOT += amplitude;
    if (iTS == nnlsWork_.tsOffset)
      tstrig += amplitude;
  }

  tsTOT *= channelData.tsGain(0);
  tstrig *= channelData.tsGain(0);

  useTriple = false;
  if (tstrig >= ts4Thresh_ && tsTOT > 0) {
    //Average pedestal width (for covariance matrix constraint)
    nnlsWork_.pedVal = 0.25f * (channelData.tsPedestalWidth(0) * channelData.tsPedestalWidth(0) +
                                channelData.tsPedestalWidth(1) * channelData.tsPedestalWidth(1) +
                                channelData.tsPedestalWidth(2) * channelData.tsPedestalWidth(2) +
                                channelData.tsPedestalWidth(3) * channelData.tsPedestalWidth(3));

    // only do pre-fit with 1 pulse if chiSq threshold is positive
    if (chiSqSwitch_ > 0) {
      doFit(reconstructedVals, 1);
      if (reconstructedVals[2] > chiSqSwitch_) {
        doFit(reconstructedVals, 0);  //nbx=0 means use configured BXs
        useTriple = true;
      }
    } else {
      doFit(reconstructedVals, 0);
      useTriple = true;
    }
  } else {
    reconstructedVals.at(0) = 0.f;      //energy
    reconstructedVals.at(1) = -9999.f;  //time
    reconstructedVals.at(2) = -9999.f;  //chi2
  }

  reconstructedEnergy = reconstructedVals[0] * channelData.tsGain(0);
  reconstructedTime = reconstructedVals[1];
  chi2 = reconstructedVals[2];
}

void MahiFit::doFit(std::array<float, 3>& correctedOutput, int nbx) const {
  unsigned int bxSize = 1;

  if (nbx == 1) {
    nnlsWork_.bxOffset = 0;
  } else {
    bxSize = bxSizeConf_;
    nnlsWork_.bxOffset = static_cast<int>(nnlsWork_.tsOffset) >= bxOffsetConf_ ? bxOffsetConf_ : nnlsWork_.tsOffset;
  }

  nnlsWork_.nPulseTot = bxSize;

  if (dynamicPed_)
    nnlsWork_.nPulseTot++;
  nnlsWork_.bxs.setZero(nnlsWork_.nPulseTot);

  if (nbx == 1) {
    nnlsWork_.bxs.coeffRef(0) = 0;
  } else {
    for (unsigned int iBX = 0; iBX < bxSize; ++iBX) {
      nnlsWork_.bxs.coeffRef(iBX) =
          activeBXs_[iBX] -
          ((static_cast<int>(nnlsWork_.tsOffset) + activeBXs_[0]) >= 0 ? 0 : (nnlsWork_.tsOffset + activeBXs_[0]));
    }
  }

  nnlsWork_.maxoffset = nnlsWork_.bxs.coeff(bxSize - 1);
  if (dynamicPed_)
    nnlsWork_.bxs[nnlsWork_.nPulseTot - 1] = pedestalBX_;

  nnlsWork_.pulseMat.setZero(nnlsWork_.tsSize, nnlsWork_.nPulseTot);
  nnlsWork_.pulseDerivMat.setZero(nnlsWork_.tsSize, nnlsWork_.nPulseTot);

  FullSampleVector pulseShapeArray;
  FullSampleVector pulseDerivArray;
  FullSampleMatrix pulseCov;

  int offset = 0;
  for (unsigned int iBX = 0; iBX < nnlsWork_.nPulseTot; ++iBX) {
    offset = nnlsWork_.bxs.coeff(iBX);

    if (offset == pedestalBX_) {
      nnlsWork_.pulseMat.col(iBX) = SampleVector::Ones(nnlsWork_.tsSize);
      nnlsWork_.pulseDerivMat.col(iBX) = SampleVector::Zero(nnlsWork_.tsSize);
    } else {
      pulseShapeArray.setZero(nnlsWork_.tsSize + nnlsWork_.maxoffset + nnlsWork_.bxOffset);
      if (calculateArrivalTime_)
        pulseDerivArray.setZero(nnlsWork_.tsSize + nnlsWork_.maxoffset + nnlsWork_.bxOffset);
      pulseCov.setZero(nnlsWork_.tsSize + nnlsWork_.maxoffset + nnlsWork_.bxOffset,
                       nnlsWork_.tsSize + nnlsWork_.maxoffset + nnlsWork_.bxOffset);
      nnlsWork_.pulseCovArray[iBX].setZero(nnlsWork_.tsSize, nnlsWork_.tsSize);

      updatePulseShape(
          nnlsWork_.amplitudes.coeff(nnlsWork_.tsOffset + offset), pulseShapeArray, pulseDerivArray, pulseCov);

      nnlsWork_.pulseMat.col(iBX) = pulseShapeArray.segment(nnlsWork_.maxoffset - offset, nnlsWork_.tsSize);
      if (calculateArrivalTime_)
        nnlsWork_.pulseDerivMat.col(iBX) = pulseDerivArray.segment(nnlsWork_.maxoffset - offset, nnlsWork_.tsSize);
      nnlsWork_.pulseCovArray[iBX] = pulseCov.block(
          nnlsWork_.maxoffset - offset, nnlsWork_.maxoffset - offset, nnlsWork_.tsSize, nnlsWork_.tsSize);
    }
  }

  const float chiSq = minimize();

  bool foundintime = false;
  unsigned int ipulseintime = 0;

  for (unsigned int iBX = 0; iBX < nnlsWork_.nPulseTot; ++iBX) {
    if (nnlsWork_.bxs.coeff(iBX) == 0) {
      ipulseintime = iBX;
      foundintime = true;
      break;
    }
  }

  if (foundintime) {
    correctedOutput.at(0) = nnlsWork_.ampVec.coeff(ipulseintime);  //charge
    if (correctedOutput.at(0) != 0) {
      // fixME store the timeslew
      float arrivalTime = 0.f;
      if (calculateArrivalTime_)
        arrivalTime = calculateArrivalTime(ipulseintime);
      correctedOutput.at(1) = arrivalTime;  //time
    } else
      correctedOutput.at(1) = -9999.f;  //time

    correctedOutput.at(2) = chiSq;  //chi2
  }
}

const float MahiFit::minimize() const {
  nnlsWork_.invcovp.setZero(nnlsWork_.tsSize, nnlsWork_.nPulseTot);
  nnlsWork_.ampVec.setZero(nnlsWork_.nPulseTot);

  SampleMatrix invCovMat;
  invCovMat.setConstant(nnlsWork_.tsSize, nnlsWork_.tsSize, nnlsWork_.pedVal);
  invCovMat += nnlsWork_.noiseTerms.asDiagonal();

  float oldChiSq = 9999;
  float chiSq = oldChiSq;

  for (int iter = 1; iter < nMaxItersMin_; ++iter) {
    updateCov(invCovMat);

    if (nnlsWork_.nPulseTot > 1) {
      nnls();
    } else {
      onePulseMinimize();
    }

    const float newChiSq = calculateChiSq();
    const float deltaChiSq = newChiSq - chiSq;

    if (newChiSq == oldChiSq && newChiSq < chiSq) {
      break;
    }
    oldChiSq = chiSq;
    chiSq = newChiSq;

    if (std::abs(deltaChiSq) < deltaChiSqThresh_)
      break;
  }

  return chiSq;
}

void MahiFit::updatePulseShape(const float itQ,
                               FullSampleVector& pulseShape,
                               FullSampleVector& pulseDeriv,
                               FullSampleMatrix& pulseCov) const {
  float t0 = meanTime_;

  if (applyTimeSlew_) {
    if (itQ <= 1.f)
      t0 += tsDelay1GeV_;
    else
      t0 += hcalTimeSlewDelay_->delay(float(itQ), slewFlavor_);
  }

  std::array<double, HcalConst::maxSamples> pulseN;
  std::array<double, HcalConst::maxSamples> pulseM;
  std::array<double, HcalConst::maxSamples> pulseP;

  const float xx = t0;
  const float xxm = -nnlsWork_.dt + t0;
  const float xxp = nnlsWork_.dt + t0;

  psfPtr_->singlePulseShapeFuncMahi(&xx);
  psfPtr_->getPulseShape(pulseN);

  psfPtr_->singlePulseShapeFuncMahi(&xxm);
  psfPtr_->getPulseShape(pulseM);

  psfPtr_->singlePulseShapeFuncMahi(&xxp);
  psfPtr_->getPulseShape(pulseP);

  //in the 2018+ case where the sample of interest (SOI) is in TS3, add an extra offset to align
  //with previous SOI=TS4 case assumed by psfPtr_->getPulseShape()
  int delta = 4 - nnlsWork_.tsOffset;

  auto invDt = 0.5 / nnlsWork_.dt;

  for (unsigned int iTS = 0; iTS < nnlsWork_.tsSize; ++iTS) {
    pulseShape(iTS + nnlsWork_.maxoffset) = pulseN[iTS + delta];
    if (calculateArrivalTime_)
      pulseDeriv(iTS + nnlsWork_.maxoffset) = (pulseM[iTS + delta] - pulseP[iTS + delta]) * invDt;

    pulseM[iTS + delta] -= pulseN[iTS + delta];
    pulseP[iTS + delta] -= pulseN[iTS + delta];
  }

  for (unsigned int iTS = 0; iTS < nnlsWork_.tsSize; ++iTS) {
    for (unsigned int jTS = 0; jTS < iTS + 1; ++jTS) {
      auto const tmp = 0.5 * (pulseP[iTS + delta] * pulseP[jTS + delta] + pulseM[iTS + delta] * pulseM[jTS + delta]);

      pulseCov(jTS + nnlsWork_.maxoffset, iTS + nnlsWork_.maxoffset) = tmp;
      if (iTS != jTS)
        pulseCov(iTS + nnlsWork_.maxoffset, jTS + nnlsWork_.maxoffset) = tmp;
    }
  }
}

void MahiFit::updateCov(const SampleMatrix& samplecov) const {
  SampleMatrix invCovMat = samplecov;

  for (unsigned int iBX = 0; iBX < nnlsWork_.nPulseTot; ++iBX) {
    auto const amp = nnlsWork_.ampVec.coeff(iBX);
    if (amp == 0)
      continue;

    int offset = nnlsWork_.bxs.coeff(iBX);

    if (offset == pedestalBX_)
      continue;
    else {
      auto const ampsq = amp * amp;
      invCovMat += ampsq * nnlsWork_.pulseCovArray[offset + nnlsWork_.bxOffset];
    }
  }

  nnlsWork_.covDecomp.compute(invCovMat);
}

float MahiFit::calculateArrivalTime(unsigned int itIndex) const {
  if (nnlsWork_.nPulseTot > 1) {
    SamplePulseMatrix pulseDerivMatTMP = nnlsWork_.pulseDerivMat;
    for (unsigned int iBX = 0; iBX < nnlsWork_.nPulseTot; ++iBX) {
      nnlsWork_.pulseDerivMat.col(iBX) = pulseDerivMatTMP.col(nnlsWork_.bxs.coeff(iBX) + nnlsWork_.bxOffset);
    }
  }

  for (unsigned int iBX = 0; iBX < nnlsWork_.nPulseTot; ++iBX) {
    nnlsWork_.pulseDerivMat.col(iBX) *= nnlsWork_.ampVec.coeff(iBX);
  }

  //FIXME: avoid solve full equation for one element
  SampleVector residuals = nnlsWork_.pulseMat * nnlsWork_.ampVec - nnlsWork_.amplitudes;
  PulseVector solution = nnlsWork_.pulseDerivMat.colPivHouseholderQr().solve(residuals);
  float t = std::clamp((float)solution.coeff(itIndex), -timeLimit_, timeLimit_);

  return t;
}

void MahiFit::nnls() const {
  const unsigned int npulse = nnlsWork_.nPulseTot;
  const unsigned int nsamples = nnlsWork_.tsSize;

  PulseVector updateWork;

  nnlsWork_.invcovp = nnlsWork_.covDecomp.matrixL().solve(nnlsWork_.pulseMat);
  nnlsWork_.aTaMat = nnlsWork_.invcovp.transpose() * nnlsWork_.invcovp;
  nnlsWork_.aTbVec = nnlsWork_.invcovp.transpose() * (nnlsWork_.covDecomp.matrixL().solve(nnlsWork_.amplitudes));

  int iter = 0;
  Index idxwmax = 0;
  float wmax = 0.0f;
  float threshold = nnlsThresh_;

  while (true) {
    if (iter > 0 || nnlsWork_.nP == 0) {
      if (nnlsWork_.nP == std::min(npulse, nsamples))
        break;

      const unsigned int nActive = npulse - nnlsWork_.nP;
      // exit if there are no more pulses to constrain
      if (nActive == 0)
        break;

      updateWork = nnlsWork_.aTbVec - nnlsWork_.aTaMat * nnlsWork_.ampVec;

      Index idxwmaxprev = idxwmax;
      float wmaxprev = wmax;
      wmax = updateWork.tail(nActive).maxCoeff(&idxwmax);

      if (wmax < threshold || (idxwmax == idxwmaxprev && wmax == wmaxprev)) {
        break;
      }

      if (iter >= nMaxItersNNLS_) {
        break;
      }

      //unconstrain parameter
      idxwmax += nnlsWork_.nP;
      nnlsUnconstrainParameter(idxwmax);
    }

    while (true) {
      if (nnlsWork_.nP == 0)
        break;

      PulseVector ampvecpermtest = nnlsWork_.ampVec.head(nnlsWork_.nP);

      solveSubmatrix(nnlsWork_.aTaMat, nnlsWork_.aTbVec, ampvecpermtest, nnlsWork_.nP);

      //check solution
      if (ampvecpermtest.head(nnlsWork_.nP).minCoeff() > 0.f) {
        nnlsWork_.ampVec.head(nnlsWork_.nP) = ampvecpermtest.head(nnlsWork_.nP);
        break;
      }

      //update parameter vector
      Index minratioidx = 0;

      // no realizable optimization here (because it autovectorizes!)
      float minratio = std::numeric_limits<float>::max();
      for (unsigned int ipulse = 0; ipulse < nnlsWork_.nP; ++ipulse) {
        if (ampvecpermtest.coeff(ipulse) <= 0.f) {
          const float c_ampvec = nnlsWork_.ampVec.coeff(ipulse);
          const float ratio = c_ampvec / (c_ampvec - ampvecpermtest.coeff(ipulse));
          if (ratio < minratio) {
            minratio = ratio;
            minratioidx = ipulse;
          }
        }
      }
      nnlsWork_.ampVec.head(nnlsWork_.nP) +=
          minratio * (ampvecpermtest.head(nnlsWork_.nP) - nnlsWork_.ampVec.head(nnlsWork_.nP));

      //avoid numerical problems with later ==0. check
      nnlsWork_.ampVec.coeffRef(minratioidx) = 0.f;

      nnlsConstrainParameter(minratioidx);
    }

    ++iter;

    //adaptive convergence threshold to avoid infinite loops but still
    //ensure best value is used
    if (iter % 10 == 0) {
      threshold *= 10.;
    }
  }
}

void MahiFit::onePulseMinimize() const {
  nnlsWork_.invcovp = nnlsWork_.covDecomp.matrixL().solve(nnlsWork_.pulseMat);

  float aTaCoeff = (nnlsWork_.invcovp.transpose() * nnlsWork_.invcovp).coeff(0);
  float aTbCoeff =
      (nnlsWork_.invcovp.transpose() * (nnlsWork_.covDecomp.matrixL().solve(nnlsWork_.amplitudes))).coeff(0);

  nnlsWork_.ampVec.coeffRef(0) = std::max(0.f, aTbCoeff / aTaCoeff);
}

float MahiFit::calculateChiSq() const {
  return (nnlsWork_.covDecomp.matrixL().solve(nnlsWork_.pulseMat * nnlsWork_.ampVec - nnlsWork_.amplitudes))
      .squaredNorm();
}

void MahiFit::setPulseShapeTemplate(const HcalPulseShapes::Shape& ps,
                                    bool hasTimeInfo,
                                    const HcalTimeSlew* hcalTimeSlewDelay,
                                    unsigned int nSamples) {
  if (!(&ps == currentPulseShape_)) {
    hcalTimeSlewDelay_ = hcalTimeSlewDelay;
    tsDelay1GeV_ = hcalTimeSlewDelay->delay(1.0, slewFlavor_);

    resetPulseShapeTemplate(ps, hasTimeInfo, nSamples);
    currentPulseShape_ = &ps;
  }
}

void MahiFit::resetPulseShapeTemplate(const HcalPulseShapes::Shape& ps, bool hasTimeInfo, unsigned int nSamples) {
  ++cntsetPulseShape_;

  // only the pulse shape itself from PulseShapeFunctor is used for Mahi
  // the uncertainty terms calculated inside PulseShapeFunctor are used for Method 2 only
  psfPtr_.reset(new FitterFuncs::PulseShapeFunctor(ps, false, false, false, 1, 0, 0, HcalConst::maxSamples));

  // 1 sigma time constraint
  nnlsWork_.dt = hasTimeInfo ? timeSigmaSiPM_ : timeSigmaHPD_;

  nnlsWork_.tsSize = nSamples;
  nnlsWork_.amplitudes.resize(nnlsWork_.tsSize);
  nnlsWork_.noiseTerms.resize(nnlsWork_.tsSize);
}

void MahiFit::nnlsUnconstrainParameter(Index idxp) const {
  if (idxp != nnlsWork_.nP) {
    nnlsWork_.aTaMat.col(nnlsWork_.nP).swap(nnlsWork_.aTaMat.col(idxp));
    nnlsWork_.aTaMat.row(nnlsWork_.nP).swap(nnlsWork_.aTaMat.row(idxp));
    nnlsWork_.pulseMat.col(nnlsWork_.nP).swap(nnlsWork_.pulseMat.col(idxp));
    Eigen::numext::swap(nnlsWork_.aTbVec.coeffRef(nnlsWork_.nP), nnlsWork_.aTbVec.coeffRef(idxp));
    Eigen::numext::swap(nnlsWork_.ampVec.coeffRef(nnlsWork_.nP), nnlsWork_.ampVec.coeffRef(idxp));
    Eigen::numext::swap(nnlsWork_.bxs.coeffRef(nnlsWork_.nP), nnlsWork_.bxs.coeffRef(idxp));
  }
  ++nnlsWork_.nP;
}

void MahiFit::nnlsConstrainParameter(Index minratioidx) const {
  if (minratioidx != (nnlsWork_.nP - 1)) {
    nnlsWork_.aTaMat.col(nnlsWork_.nP - 1).swap(nnlsWork_.aTaMat.col(minratioidx));
    nnlsWork_.aTaMat.row(nnlsWork_.nP - 1).swap(nnlsWork_.aTaMat.row(minratioidx));
    nnlsWork_.pulseMat.col(nnlsWork_.nP - 1).swap(nnlsWork_.pulseMat.col(minratioidx));
    Eigen::numext::swap(nnlsWork_.aTbVec.coeffRef(nnlsWork_.nP - 1), nnlsWork_.aTbVec.coeffRef(minratioidx));
    Eigen::numext::swap(nnlsWork_.ampVec.coeffRef(nnlsWork_.nP - 1), nnlsWork_.ampVec.coeffRef(minratioidx));
    Eigen::numext::swap(nnlsWork_.bxs.coeffRef(nnlsWork_.nP - 1), nnlsWork_.bxs.coeffRef(minratioidx));
  }
  --nnlsWork_.nP;
}

void MahiFit::phase1Debug(const HBHEChannelInfo& channelData, MahiDebugInfo& mdi) const {
  float recoEnergy, recoTime, chi2;
  bool use3;
  phase1Apply(channelData, recoEnergy, recoTime, use3, chi2);

  mdi.nSamples = channelData.nSamples();
  mdi.soi = channelData.soi();

  mdi.use3 = use3;

  mdi.inTimeConst = nnlsWork_.dt;
  mdi.inPedAvg = 0.25 * (channelData.tsPedestalWidth(0) * channelData.tsPedestalWidth(0) +
                         channelData.tsPedestalWidth(1) * channelData.tsPedestalWidth(1) +
                         channelData.tsPedestalWidth(2) * channelData.tsPedestalWidth(2) +
                         channelData.tsPedestalWidth(3) * channelData.tsPedestalWidth(3));
  mdi.inGain = channelData.tsGain(0);

  for (unsigned int iTS = 0; iTS < channelData.nSamples(); ++iTS) {
    double charge = channelData.tsRawCharge(iTS);
    double ped = channelData.tsPedestal(iTS);

    mdi.inNoiseADC[iTS] = norm_ * channelData.tsDFcPerADC(iTS);

    if ((charge - ped) > channelData.tsPedestalWidth(iTS)) {
      mdi.inNoisePhoto[iTS] = sqrt((charge - ped) * channelData.fcByPE());
    } else {
      mdi.inNoisePhoto[iTS] = 0;
    }

    mdi.inPedestal[iTS] = channelData.tsPedestalWidth(iTS);
    mdi.totalUCNoise[iTS] = nnlsWork_.noiseTerms.coeffRef(iTS);

    if (channelData.hasTimeInfo()) {
      mdi.inputTDC[iTS] = channelData.tsRiseTime(iTS);
    } else {
      mdi.inputTDC[iTS] = -1;
    }
  }

  mdi.arrivalTime = recoTime;
  mdi.chiSq = chi2;

  for (unsigned int iBX = 0; iBX < nnlsWork_.nPulseTot; ++iBX) {
    if (nnlsWork_.bxs.coeff(iBX) == 0) {
      mdi.mahiEnergy = nnlsWork_.ampVec.coeff(iBX);
      for (unsigned int iTS = 0; iTS < nnlsWork_.tsSize; ++iTS) {
        mdi.count[iTS] = iTS;
        mdi.inputTS[iTS] = nnlsWork_.amplitudes.coeff(iTS);
        mdi.itPulse[iTS] = nnlsWork_.pulseMat.col(iBX).coeff(iTS);
      }
    } else if (nnlsWork_.bxs.coeff(iBX) == pedestalBX_) {
      mdi.pedEnergy = nnlsWork_.ampVec.coeff(iBX);
    } else if (nnlsWork_.bxs.coeff(iBX) >= -3 && nnlsWork_.bxs.coeff(iBX) <= 4) {
      int ootIndex = nnlsWork_.bxs.coeff(iBX);
      if (ootIndex > 0)
        ootIndex += 2;
      else
        ootIndex += 3;
      mdi.ootEnergy[ootIndex] = nnlsWork_.ampVec.coeff(iBX);
      for (unsigned int iTS = 0; iTS < nnlsWork_.tsSize; ++iTS) {
        mdi.ootPulse[ootIndex][iTS] = nnlsWork_.pulseMat.col(iBX).coeff(iTS);
      }
    }
  }
}

void MahiFit::solveSubmatrix(PulseMatrix& mat, PulseVector& invec, PulseVector& outvec, unsigned nP) const {
  using namespace Eigen;
  switch (nP) {  // pulse matrix is always square.
    case 10: {
      Matrix<float, 10, 10> temp = mat;
      outvec.head<10>() = temp.ldlt().solve(invec.head<10>());
    } break;
    case 9: {
      Matrix<float, 9, 9> temp = mat.topLeftCorner<9, 9>();
      outvec.head<9>() = temp.ldlt().solve(invec.head<9>());
    } break;
    case 8: {
      Matrix<float, 8, 8> temp = mat.topLeftCorner<8, 8>();
      outvec.head<8>() = temp.ldlt().solve(invec.head<8>());
    } break;
    case 7: {
      Matrix<float, 7, 7> temp = mat.topLeftCorner<7, 7>();
      outvec.head<7>() = temp.ldlt().solve(invec.head<7>());
    } break;
    case 6: {
      Matrix<float, 6, 6> temp = mat.topLeftCorner<6, 6>();
      outvec.head<6>() = temp.ldlt().solve(invec.head<6>());
    } break;
    case 5: {
      Matrix<float, 5, 5> temp = mat.topLeftCorner<5, 5>();
      outvec.head<5>() = temp.ldlt().solve(invec.head<5>());
    } break;
    case 4: {
      Matrix<float, 4, 4> temp = mat.topLeftCorner<4, 4>();
      outvec.head<4>() = temp.ldlt().solve(invec.head<4>());
    } break;
    case 3: {
      Matrix<float, 3, 3> temp = mat.topLeftCorner<3, 3>();
      outvec.head<3>() = temp.ldlt().solve(invec.head<3>());
    } break;
    case 2: {
      Matrix<float, 2, 2> temp = mat.topLeftCorner<2, 2>();
      outvec.head<2>() = temp.ldlt().solve(invec.head<2>());
    } break;
    case 1: {
      Matrix<float, 1, 1> temp = mat.topLeftCorner<1, 1>();
      outvec.head<1>() = temp.ldlt().solve(invec.head<1>());
    } break;
    default:
      throw cms::Exception("HcalMahiWeirdState")
          << "Weird number of pulses encountered in Mahi, module is configured incorrectly!";
  }
}

void MahiFit::resetWorkspace() const {
  nnlsWork_.nPulseTot = 0;
  nnlsWork_.tsOffset = 0;
  nnlsWork_.bxOffset = 0;
  nnlsWork_.maxoffset = 0;
  nnlsWork_.nP = 0;

  // NOT SURE THIS IS NEEDED
  nnlsWork_.amplitudes.setZero();
  nnlsWork_.noiseTerms.setZero();
}
