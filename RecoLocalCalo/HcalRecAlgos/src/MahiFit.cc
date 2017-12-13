#include "RecoLocalCalo/HcalRecAlgos/interface/MahiFit.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <fstream> 

MahiFit::MahiFit() :
  fullTSSize_(19), 
  fullTSofInterest_(8)
{}

double MahiFit::getSiPMDarkCurrent(double darkCurrent, double fcByPE, double lambda) {
  double mu = darkCurrent * 25 / fcByPE;
  return sqrt(mu/pow(1-lambda,3)) * fcByPE;
}

void MahiFit::setParameters(double iTS4Thresh, double chiSqSwitch, bool iApplyTimeSlew, HcalTimeSlew::BiasSetting slewFlavor,
			       double iMeanTime, double iTimeSigmaHPD, double iTimeSigmaSiPM, 
			       const std::vector <int> &iActiveBXs, int iNMaxItersMin, int iNMaxItersNNLS,
			       double iDeltaChiSqThresh, double iNnlsThresh) {

  ts4Thresh_     = iTS4Thresh;
  chiSqSwitch_   = chiSqSwitch;

  applyTimeSlew_ = iApplyTimeSlew;
  slewFlavor_    = slewFlavor;

  meanTime_      = iMeanTime;
  timeSigmaHPD_  = iTimeSigmaHPD;
  timeSigmaSiPM_ = iTimeSigmaSiPM;

  activeBXs_ = iActiveBXs;

  nMaxItersMin_  = iNMaxItersMin;
  nMaxItersNNLS_ = iNMaxItersNNLS;

  deltaChiSqThresh_ = iDeltaChiSqThresh;
  nnlsThresh_    = iNnlsThresh;

  bxOffsetConf_ = -(*std::min_element(activeBXs_.begin(), activeBXs_.end()));
  bxSizeConf_   = activeBXs_.size();
}

void MahiFit::phase1Apply(const HBHEChannelInfo& channelData,
			     float& reconstructedEnergy,
			     float& reconstructedTime,
			     float& chi2) {

  tsSize_ = channelData.nSamples();
  tsOffset_ = channelData.soi();
  fullTSOffset_ = fullTSofInterest_ - tsOffset_;

  // 1 sigma time constraint
  if (channelData.hasTimeInfo()) dt_=timeSigmaSiPM_;
  else dt_=timeSigmaHPD_;

  //fC to photo-electron scale factor (for P.E. uncertainties)
  fcByPe_ = channelData.fcByPE();

  //Dark current value for this channel (SiPM only)
  darkCurrent_ =  getSiPMDarkCurrent(channelData.darkCurrent(), 
				     channelData.fcByPE(),
				     channelData.lambda());

  //Average pedestal width (for covariance matrix constraint)
  pedConstraint_ = 0.25*( channelData.tsPedestalWidth(0)*channelData.tsPedestalWidth(0)+
			  channelData.tsPedestalWidth(1)*channelData.tsPedestalWidth(1)+
			  channelData.tsPedestalWidth(2)*channelData.tsPedestalWidth(2)+
			  channelData.tsPedestalWidth(3)*channelData.tsPedestalWidth(3) );

  std::vector<float> reconstructedVals;
  SampleVector charges;
  
  double tsTOT = 0, tstrig = 0; // in GeV
  for(unsigned int iTS=0; iTS<tsSize_; ++iTS){
    double charge = channelData.tsRawCharge(iTS);
    double ped = channelData.tsPedestal(iTS);

    charges.coeffRef(iTS) = charge - ped;

    //ADC granularity
    double noiseADC = (1./sqrt(12))*channelData.tsDFcPerADC(iTS);

    //Dark current (for SiPMs)
    double noiseDC=0;
    if(channelData.hasTimeInfo() && !channelData.hasEffectivePedestals() && (charge-ped)>channelData.tsPedestalWidth(iTS)) {
      noiseDC = darkCurrent_;
    }

    //Photostatistics
    double noisePhoto = 0;
    if ( (charge-ped)>channelData.tsPedestalWidth(iTS)) {
      noisePhoto = sqrt((charge-ped)*channelData.fcByPE());
    }

    //Electronic pedestal
    double pedWidth = channelData.tsPedestalWidth(iTS);

    //Total uncertainty from all sources
    noiseTerms_.coeffRef(iTS) = noiseADC*noiseADC + noiseDC*noiseDC + noisePhoto*noisePhoto + pedWidth*pedWidth;

    tsTOT += charge - ped;
    if( iTS==tsOffset_ ){
      tstrig += (charge - ped)*channelData.tsGain(0);
    }
  }

  bool status =false;
  if(tstrig >= ts4Thresh_) {
    if (chiSqSwitch_>0) {
      status = doFit(charges,reconstructedVals,1);
      if (reconstructedVals[1]>chiSqSwitch_) {
	status = doFit(charges,reconstructedVals,0);
      }
    }
    else {
      status = doFit(charges,reconstructedVals,0);
    }
  }
  
  if (!status) {
    reconstructedVals.clear();
    reconstructedVals.push_back(0.);
    reconstructedVals.push_back(888.);
    reconstructedVals.push_back(888.);
  }
  
  reconstructedEnergy = reconstructedVals[0]*channelData.tsGain(0);
  reconstructedTime = reconstructedVals[1];
  chi2 = reconstructedVals[2];

}

bool MahiFit::doFit(SampleVector amplitudes, std::vector<float> &correctedOutput, int nbx) {

  if (nbx==1) {
    bxSize_ = 1;
    bxOffset_ = 0;
  }
  else {
    bxSize_ = bxSizeConf_;
    bxOffset_ = bxOffsetConf_;
  }

  bxs_.resize(bxSize_);

  if (nbx==1) {
    bxs_.coeffRef(0) = 0;
  }
  else {
    for (unsigned int iBX=0; iBX<bxSize_; iBX++) {
      bxs_.coeffRef(iBX) = activeBXs_[iBX];
    }
  }

  nPulseTot_ = bxSize_;
  nPulseTot_++;
  bxs_.resize(nPulseTot_);
  bxs_[nPulseTot_-1] = 100;

  amplitudes_ = amplitudes;

  nP_ = 0;
  
  pulseMat_.resize(tsSize_,nPulseTot_);
  ampVec_ = PulseVector::Zero(nPulseTot_);
  errVec_ = PulseVector::Zero(nPulseTot_);

  bool status = true;

  int offset=0;
  for (unsigned int iBX=0; iBX<nPulseTot_; iBX++) {
    offset=bxs_.coeff(iBX);

    pulseShapeArray_[iBX] = FullSampleVector::Zero(MaxFSVSize);
    pulseDerivArray_[iBX] = FullSampleVector::Zero(MaxFSVSize);
    pulseCovArray_[iBX]   = FullSampleMatrix::Constant(0);

    if (offset==100) {
      ampVec_.coeffRef(iBX) = sqrt(pedConstraint_);
    }
    else {
      status = updatePulseShape(amplitudes_.coeff(tsOffset_ + offset), 
				pulseShapeArray_[iBX], 
				pulseDerivArray_[iBX],
				pulseCovArray_[iBX]);
      
      if (offset==0) {
      	ampVec_.coeffRef(iBX)= amplitudes_.coeff(tsOffset_ + offset)/double(pulseShapeArray_[iBX].coeff(fullTSofInterest_));
      }
      else {
	ampVec_.coeffRef(iBX)=0;
      }

      pulseMat_.col(iBX) = pulseShapeArray_[iBX].segment(fullTSOffset_ - offset, tsSize_);

    }
  }

  SampleVector ped = SampleVector::Ones();
  pulseMat_.col(nPulseTot_-1) = ped;

  chiSq_ = 999;

  aTaMat_.resize(nPulseTot_, nPulseTot_);
  aTbVec_.resize(nPulseTot_);

  status = minimize(); 
  ampVecMin_ = ampVec_;
  bxsMin_ = bxs_;
  residuals_ = pulseMat_*ampVec_ - amplitudes_;

  double arrivalTime = calculateArrivalTime();

  if (!status) return status;

  bool foundintime = false;
  unsigned int ipulseintime = 0;

  for (unsigned int iBX=0; iBX<nPulseTot_; ++iBX) {
    if (bxs_.coeff(iBX)==0) {
      ipulseintime = iBX;
      foundintime = true;
    }
  }
  if (!foundintime) return status;

  correctedOutput.clear();
  correctedOutput.push_back(ampVec_.coeff(ipulseintime)); //charge
  correctedOutput.push_back(arrivalTime); //time
  correctedOutput.push_back(chiSq_); //chi2
  
  
  return status;
}

bool MahiFit::minimize() {

  int iter = 0;
  bool status = false;

  double oldChiSq=chiSq_;

  while (true) {
    if (iter>=nMaxItersMin_) {
      break;
    }
    
    status=updateCov();
    if (!status) break;

    if (nPulseTot_>1) {
      status = nnls();
    }
    else {
      status = onePulseMinimize();
    }
    if (!status) break;
    
    double newChiSq=calculateChiSq();
    double deltaChiSq = newChiSq - chiSq_;

    if (newChiSq==oldChiSq && newChiSq<chiSq_) {
      break;
    }
    oldChiSq=chiSq_;
    chiSq_ = newChiSq;

    if (std::abs(deltaChiSq)<deltaChiSqThresh_) break;

    iter++;
    
  }

  return status;
}

bool MahiFit::updatePulseShape(double itQ, FullSampleVector &pulseShape, FullSampleVector &pulseDeriv,
				  FullSampleMatrix &pulseCov) {

  float t0=meanTime_;
  if (applyTimeSlew_) 
    t0=HcalTimeSlew::delay(std::max(1.0, itQ), slewFlavor_);

  pulseN_.fill(0);
  pulseM_.fill(0);
  pulseP_.fill(0);

  const double xx[4]={t0, 1.0, 0.0, 3};
  const double xxm[4]={-dt_+t0, 1.0, 0.0, 3};
  const double xxp[4]={ dt_+t0, 1.0, 0.0, 3};

  (*pfunctor_)(&xx[0]);
  psfPtr_->getPulseShape(pulseN_);

  (*pfunctor_)(&xxm[0]);
  psfPtr_->getPulseShape(pulseM_);
  
  (*pfunctor_)(&xxp[0]);
  psfPtr_->getPulseShape(pulseP_);

  for (unsigned int iTS=fullTSOffset_; iTS<fullTSOffset_ + tsSize_; iTS++) {
    pulseShape.coeffRef(iTS) = pulseN_[iTS-fullTSOffset_];
    pulseDeriv.coeffRef(iTS) = 0.5*(pulseM_[iTS-fullTSOffset_]+pulseP_[iTS-fullTSOffset_])/(2*dt_);

    pulseM_[iTS-fullTSOffset_] -= pulseN_[iTS-fullTSOffset_];
    pulseP_[iTS-fullTSOffset_] -= pulseN_[iTS-fullTSOffset_];
  }

  for (unsigned int iTS=fullTSOffset_; iTS<fullTSOffset_+tsSize_; iTS++) {
    for (unsigned int jTS=fullTSOffset_; jTS<iTS+1; jTS++) {
      
      double tmp=0.5*( pulseP_[iTS-fullTSOffset_]*pulseP_[jTS-fullTSOffset_] +
		       pulseM_[iTS-fullTSOffset_]*pulseM_[jTS-fullTSOffset_] );
      
      pulseCov(iTS,jTS) += tmp;
      pulseCov(jTS,iTS) += tmp;
      
    }
  }
  
  return true;  
}

bool MahiFit::updateCov() {
  
  bool status=true;
  
  invCovMat_ = noiseTerms_.asDiagonal();
  invCovMat_ +=SampleMatrix::Constant(pedConstraint_);

  for (unsigned int iBX=0; iBX<nPulseTot_; iBX++) {
    if (ampVec_.coeff(iBX)==0) continue;
    
    unsigned int offset=bxs_.coeff(iBX);

    if (offset==100) continue;		       
    else { 
      invCovMat_ += ampVec_.coeff(iBX)*ampVec_.coeff(iBX)
	*pulseCovArray_.at(offset+bxOffset_).block(fullTSOffset_-offset, fullTSOffset_-offset, tsSize_, tsSize_);
    }
  }
  
  covDecomp_.compute(invCovMat_);
  
  return status;
}

double MahiFit::calculateArrivalTime() {

  pulseDerivMat_.resize(tsSize_,nPulseTot_);

  int itIndex=0;

  for (unsigned int iBX=0; iBX<nPulseTot_; iBX++) {
    int offset=bxs_.coeff(iBX);
    if (offset==0) itIndex=iBX;

    if (offset==100) {
      pulseDerivMat_.col(iBX) = SampleVector::Zero(tsSize_);
    }
    else {
      pulseDerivMat_.col(iBX) = pulseDerivArray_.at(offset+bxOffset_).segment(fullTSOffset_-offset, tsSize_);
    }
  }

  PulseVector solution = pulseDerivMat_.colPivHouseholderQr().solve(residuals_);
  return solution.coeff(itIndex)/ampVec_.coeff(itIndex);

}
  

bool MahiFit::nnls() {
  const unsigned int npulse = nPulseTot_;
  
  for (unsigned int iBX=0; iBX<npulse; iBX++) {
    int offset=bxs_.coeff(iBX);
    if (offset==100) {
      pulseMat_.col(iBX) = SampleVector::Ones();
    }
    else {
      pulseMat_.col(iBX) = pulseShapeArray_.at(offset+bxOffset_).segment(fullTSOffset_-offset, tsSize_);
    }
  }

  invcovp_ = covDecomp_.matrixL().solve(pulseMat_);
  aTaMat_ = invcovp_.transpose().lazyProduct(invcovp_);
  aTbVec_ = invcovp_.transpose().lazyProduct(covDecomp_.matrixL().solve(amplitudes_));
  
  int iter = 0;
  Index idxwmax = 0;
  double wmax = 0.0;
  double threshold = nnlsThresh_;
  
  while (true) {    
    if (iter>0 || nP_==0) {
      if ( nP_==std::min(npulse, tsSize_)) break;
      
      const unsigned int nActive = npulse - nP_;
      updateWork_ = aTbVec_ - aTaMat_*ampVec_;
      
      Index idxwmaxprev = idxwmax;
      double wmaxprev = wmax;
      wmax = updateWork_.tail(nActive).maxCoeff(&idxwmax);
      
      if (wmax<threshold || (idxwmax==idxwmaxprev && wmax==wmaxprev)) {
	break;
      }
      
      if (iter>=nMaxItersNNLS_) {
	break;
      }

      //unconstrain parameter
      Index idxp = nP_ + idxwmax;
      nnlsUnconstrainParameter(idxp);

    }

    while (true) {
      if (nP_==0) break;     

      ampvecpermtest_ = ampVec_;
      
      eigenSolveSubmatrix(aTaMat_,aTbVec_,ampvecpermtest_,nP_);

      //check solution
      bool positive = true;
      for (unsigned int i = 0; i < nP_; ++i)
        positive &= (ampvecpermtest_(i) > 0);
      if (positive) {
        ampVec_.head(nP_) = ampvecpermtest_.head(nP_);
        break;
      } 

      //update parameter vector
      Index minratioidx=0;
      
      // no realizable optimization here (because it autovectorizes!)
      double minratio = std::numeric_limits<double>::max();
      for (unsigned int ipulse=0; ipulse<nP_; ++ipulse) {
	if (ampvecpermtest_.coeff(ipulse)<=0.) {
	  const double c_ampvec = ampVec_.coeff(ipulse);
	  const double ratio = c_ampvec/(c_ampvec-ampvecpermtest_.coeff(ipulse));
	  if (ratio<minratio) {
	    minratio = ratio;
	    minratioidx = ipulse;
	  }
	}
      }
      ampVec_.head(nP_) += minratio*(ampvecpermtest_.head(nP_) - ampVec_.head(nP_));
      
      //avoid numerical problems with later ==0. check
      ampVec_.coeffRef(minratioidx) = 0.;
      
      nnlsConstrainParameter(minratioidx);
    }
   
    ++iter;

    //adaptive convergence threshold to avoid infinite loops but still
    //ensure best value is used
    if (iter%10==0) {
      threshold *= 10.;
    }
    
    break;
  }
  
  return true;
}

bool MahiFit::onePulseMinimize() {

  invcovp_ = covDecomp_.matrixL().solve(pulseMat_);

  SingleMatrix aTamatval = invcovp_.transpose()*invcovp_;
  SingleVector aTbvecval = invcovp_.transpose()*covDecomp_.matrixL().solve(amplitudes_);

  ampVec_.coeffRef(0) = std::max(0., aTbvecval.coeff(0)/aTamatval.coeff(0));

  return true;

}

double MahiFit::calculateChiSq() {

  return (covDecomp_.matrixL().solve(pulseMat_*ampVec_ - amplitudes_)).squaredNorm();
}

void MahiFit::setPulseShapeTemplate(const HcalPulseShapes::Shape& ps) {

  if (!(&ps == currentPulseShape_ ))
    {
      resetPulseShapeTemplate(ps);
      currentPulseShape_ = &ps;
    }
}

void MahiFit::resetPulseShapeTemplate(const HcalPulseShapes::Shape& ps) { 
  ++ cntsetPulseShape_;

  // only the pulse shape itself from PulseShapeFunctor is used for Mahi
  // the uncertainty terms calculated inside PulseShapeFunctor are used for Method 2 only
  psfPtr_.reset(new FitterFuncs::PulseShapeFunctor(ps,false,false,false,false,
						   1,0,2.5,0,0.00065,1,10));
  pfunctor_    = std::unique_ptr<ROOT::Math::Functor>( new ROOT::Math::Functor(psfPtr_.get(),&FitterFuncs::PulseShapeFunctor::singlePulseShapeFunc, 3) );

}

void MahiFit::nnlsUnconstrainParameter(Index idxp) {
  aTaMat_.col(nP_).swap(aTaMat_.col(idxp));
  aTaMat_.row(nP_).swap(aTaMat_.row(idxp));
  pulseMat_.col(nP_).swap(pulseMat_.col(idxp));
  std::swap(aTbVec_.coeffRef(nP_),aTbVec_.coeffRef(idxp));
  std::swap(ampVec_.coeffRef(nP_),ampVec_.coeffRef(idxp));
  std::swap(bxs_.coeffRef(nP_),bxs_.coeffRef(idxp));
  ++nP_;
}

void MahiFit::nnlsConstrainParameter(Index minratioidx) {
  aTaMat_.col(nP_-1).swap(aTaMat_.col(minratioidx));
  aTaMat_.row(nP_-1).swap(aTaMat_.row(minratioidx));
  pulseMat_.col(nP_-1).swap(pulseMat_.col(minratioidx));
  std::swap(aTbVec_.coeffRef(nP_-1),aTbVec_.coeffRef(minratioidx));
  std::swap(ampVec_.coeffRef(nP_-1),ampVec_.coeffRef(minratioidx));
  std::swap(bxs_.coeffRef(nP_-1),bxs_.coeffRef(minratioidx));
  --nP_;

}

void MahiFit::eigenSolveSubmatrix(PulseMatrix& mat, PulseVector& invec, PulseVector& outvec, unsigned NP) {
  using namespace Eigen;
  switch( NP ) { // pulse matrix is always square.
  case 10:
    {
      Matrix<double,10,10> temp = mat;
      outvec.head<10>() = temp.ldlt().solve(invec.head<10>());
    }
    break;
  case 9:
    {
      Matrix<double,9,9> temp = mat.topLeftCorner<9,9>();
      outvec.head<9>() = temp.ldlt().solve(invec.head<9>());
    }
    break;
  case 8:
    {
      Matrix<double,8,8> temp = mat.topLeftCorner<8,8>();
      outvec.head<8>() = temp.ldlt().solve(invec.head<8>());
    }
    break;
  case 7:
    {
      Matrix<double,7,7> temp = mat.topLeftCorner<7,7>();
      outvec.head<7>() = temp.ldlt().solve(invec.head<7>());
    }
    break;
  case 6:
    {
      Matrix<double,6,6> temp = mat.topLeftCorner<6,6>();
      outvec.head<6>() = temp.ldlt().solve(invec.head<6>());
    }
    break;
  case 5:
    {
      Matrix<double,5,5> temp = mat.topLeftCorner<5,5>();
      outvec.head<5>() = temp.ldlt().solve(invec.head<5>());
    }
    break;
  case 4:
    {
      Matrix<double,4,4> temp = mat.topLeftCorner<4,4>();
      outvec.head<4>() = temp.ldlt().solve(invec.head<4>());
    }
    break;
  case 3: 
    {
      Matrix<double,3,3> temp = mat.topLeftCorner<3,3>();
      outvec.head<3>() = temp.ldlt().solve(invec.head<3>());
    }
    break;
  case 2:
    {
      Matrix<double,2,2> temp = mat.topLeftCorner<2,2>();
      outvec.head<2>() = temp.ldlt().solve(invec.head<2>());
    }
    break;
  case 1:
    {
      Matrix<double,1,1> temp = mat.topLeftCorner<1,1>();
      outvec.head<1>() = temp.ldlt().solve(invec.head<1>());
    }
    break;
  default:
    throw cms::Exception("HcalMahiWeirdState") 
      << "Weird number of pulses encountered in Mahi, module is configured incorrectly!";
  }
}
