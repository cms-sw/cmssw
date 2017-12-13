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

void MahiFit::setParameters(bool iDynamicPed, double iTS4Thresh, double chiSqSwitch, 
			    bool iApplyTimeSlew, HcalTimeSlew::BiasSetting slewFlavor,
			    double iMeanTime, double iTimeSigmaHPD, double iTimeSigmaSiPM, 
			    const std::vector <int> &iActiveBXs, int iNMaxItersMin, int iNMaxItersNNLS,
			    double iDeltaChiSqThresh, double iNnlsThresh) {

  dynamicPed_    = iDynamicPed;
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
			  bool& useTriple, 
			  float& chi2) {

  resetWorkspace();

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
    nnlsWork_.noiseTerms.coeffRef(iTS) = noiseADC*noiseADC + noiseDC*noiseDC + noisePhoto*noisePhoto + pedWidth*pedWidth;

    tsTOT += charge - ped;
    if( iTS==tsOffset_ ){
      tstrig += (charge - ped)*channelData.tsGain(0);
    }
  }

  if(tstrig >= ts4Thresh_) {

    useTriple=false;

    if (chiSqSwitch_>0) {
      doFit(charges,reconstructedVals,1);
      if (reconstructedVals[1]>chiSqSwitch_) {
	doFit(charges,reconstructedVals,0);
	useTriple=true;
      }
    }
    else {
      doFit(charges,reconstructedVals,0);
      useTriple=true;
    }
  }
  else{
    reconstructedVals.clear();
    reconstructedVals.push_back(0.);
    reconstructedVals.push_back(888.);
    reconstructedVals.push_back(888.);
  }
  
  reconstructedEnergy = reconstructedVals[0]*channelData.tsGain(0);
  reconstructedTime = reconstructedVals[1];
  chi2 = reconstructedVals[2];

}

void MahiFit::doFit(SampleVector amplitudes, std::vector<float> &correctedOutput, int nbx) {

  unsigned int bxSize=1;

  if (nbx==1) {
    nnlsWork_.bxOffset = 0;
  }
  else {
    bxSize = bxSizeConf_;
    nnlsWork_.bxOffset = bxOffsetConf_;
  }

  nnlsWork_.bxs.resize(bxSize);

  if (nbx==1) {
    nnlsWork_.bxs.coeffRef(0) = 0;
  }
  else {
    for (unsigned int iBX=0; iBX<bxSize; iBX++) {
      nnlsWork_.bxs.coeffRef(iBX) = activeBXs_[iBX];
    }
  }

  nnlsWork_.nPulseTot = bxSize;

  if (dynamicPed_) {
    nnlsWork_.nPulseTot++;
    nnlsWork_.bxs.resize(nnlsWork_.nPulseTot);
    nnlsWork_.bxs[nnlsWork_.nPulseTot-1] = 100;
  }

  nnlsWork_.amplitudes = amplitudes;

  nP_ = 0;
  
  nnlsWork_.pulseMat.resize(tsSize_,nnlsWork_.nPulseTot);
  nnlsWork_.ampVec = PulseVector::Zero(nnlsWork_.nPulseTot);
  nnlsWork_.errVec = PulseVector::Zero(nnlsWork_.nPulseTot);

  int offset=0;
  for (unsigned int iBX=0; iBX<nnlsWork_.nPulseTot; iBX++) {
    offset=nnlsWork_.bxs.coeff(iBX);

    nnlsWork_.pulseShapeArray[iBX] = FullSampleVector::Zero(MaxFSVSize);
    nnlsWork_.pulseDerivArray[iBX] = FullSampleVector::Zero(MaxFSVSize);
    nnlsWork_.pulseCovArray[iBX]   = FullSampleMatrix::Constant(0);

    if (offset==100) {
      nnlsWork_.ampVec.coeffRef(iBX) = sqrt(pedConstraint_);
    }
    else {
      updatePulseShape(nnlsWork_.amplitudes.coeff(tsOffset_ + offset), 
		       nnlsWork_.pulseShapeArray[iBX], 
		       nnlsWork_.pulseDerivArray[iBX],
		       nnlsWork_.pulseCovArray[iBX]);
      
      if (offset==0) {
      	nnlsWork_.ampVec.coeffRef(iBX)= nnlsWork_.amplitudes.coeff(tsOffset_ + offset)/double(nnlsWork_.pulseShapeArray[iBX].coeff(fullTSofInterest_));
      }
      else {
	nnlsWork_.ampVec.coeffRef(iBX)=0;
      }

      nnlsWork_.pulseMat.col(iBX) = nnlsWork_.pulseShapeArray[iBX].segment(fullTSOffset_ - offset, tsSize_);

    }
  }

  //SampleVector ped = SampleVector::Ones();
  nnlsWork_.pulseMat.col(nnlsWork_.nPulseTot-1) = SampleVector::Ones();

  chiSq_ = 999;

  nnlsWork_.aTaMat.resize(nnlsWork_.nPulseTot, nnlsWork_.nPulseTot);
  nnlsWork_.aTbVec.resize(nnlsWork_.nPulseTot);

  minimize(); 
  ampVecMin_ = nnlsWork_.ampVec;
  bxsMin_ = nnlsWork_.bxs;
  nnlsWork_.residuals = nnlsWork_.pulseMat*nnlsWork_.ampVec - nnlsWork_.amplitudes;


  bool foundintime = false;
  unsigned int ipulseintime = 0;

  for (unsigned int iBX=0; iBX<nnlsWork_.nPulseTot; ++iBX) {
    if (nnlsWork_.bxs.coeff(iBX)==0) {
      ipulseintime = iBX;
      foundintime = true;
    }
  }

  correctedOutput.clear();
  if (foundintime) {
    correctedOutput.push_back(nnlsWork_.ampVec.coeff(ipulseintime)); //charge
    double arrivalTime = calculateArrivalTime();
    correctedOutput.push_back(arrivalTime); //time
    correctedOutput.push_back(chiSq_); //chi2
  }
  
}

void MahiFit::minimize() {

  int iter = 0;
  double oldChiSq=chiSq_;

  while (true) {
    if (iter>=nMaxItersMin_) {
      break;
    }
    
    updateCov();

    if (nnlsWork_.nPulseTot>1) {
      nnls();
    }
    else {
      onePulseMinimize();
    }
    
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

}

void MahiFit::updatePulseShape(double itQ, FullSampleVector &pulseShape, FullSampleVector &pulseDeriv,
			       FullSampleMatrix &pulseCov) {
  
  float t0=meanTime_;
  if (applyTimeSlew_) 
    t0=HcalTimeSlew::delay(std::max(1.0, itQ), slewFlavor_);
  
  nnlsWork_.pulseN.fill(0);
  nnlsWork_.pulseM.fill(0);
  nnlsWork_.pulseP.fill(0);

  const double xx[4]={t0, 1.0, 0.0, 3};
  const double xxm[4]={-dt_+t0, 1.0, 0.0, 3};
  const double xxp[4]={ dt_+t0, 1.0, 0.0, 3};

  (*pfunctor_)(&xx[0]);
  psfPtr_->getPulseShape(nnlsWork_.pulseN);

  (*pfunctor_)(&xxm[0]);
  psfPtr_->getPulseShape(nnlsWork_.pulseM);
  
  (*pfunctor_)(&xxp[0]);
  psfPtr_->getPulseShape(nnlsWork_.pulseP);

  for (unsigned int iTS=fullTSOffset_; iTS<fullTSOffset_ + tsSize_; iTS++) {
    pulseShape.coeffRef(iTS) = nnlsWork_.pulseN[iTS-fullTSOffset_];
    pulseDeriv.coeffRef(iTS) = 0.5*(nnlsWork_.pulseM[iTS-fullTSOffset_]+nnlsWork_.pulseP[iTS-fullTSOffset_])/(2*dt_);

    nnlsWork_.pulseM[iTS-fullTSOffset_] -= nnlsWork_.pulseN[iTS-fullTSOffset_];
    nnlsWork_.pulseP[iTS-fullTSOffset_] -= nnlsWork_.pulseN[iTS-fullTSOffset_];
  }

  for (unsigned int iTS=fullTSOffset_; iTS<fullTSOffset_+tsSize_; iTS++) {
    for (unsigned int jTS=fullTSOffset_; jTS<iTS+1; jTS++) {
      
      double tmp=0.5*( nnlsWork_.pulseP[iTS-fullTSOffset_]*nnlsWork_.pulseP[jTS-fullTSOffset_] +
		       nnlsWork_.pulseM[iTS-fullTSOffset_]*nnlsWork_.pulseM[jTS-fullTSOffset_] );
      
      pulseCov(iTS,jTS) += tmp;
      pulseCov(jTS,iTS) += tmp;
      
    }
  }
  
}

void MahiFit::updateCov() {
  
  nnlsWork_.invCovMat = nnlsWork_.noiseTerms.asDiagonal();
  nnlsWork_.invCovMat +=SampleMatrix::Constant(pedConstraint_);

  for (unsigned int iBX=0; iBX<nnlsWork_.nPulseTot; iBX++) {
    if (nnlsWork_.ampVec.coeff(iBX)==0) continue;
    
    unsigned int offset=nnlsWork_.bxs.coeff(iBX);

    if (offset==100) continue;		       
    else { 
      nnlsWork_.invCovMat += nnlsWork_.ampVec.coeff(iBX)*nnlsWork_.ampVec.coeff(iBX)
	*nnlsWork_.pulseCovArray.at(offset+nnlsWork_.bxOffset).block(fullTSOffset_-offset, fullTSOffset_-offset, tsSize_, tsSize_);
    }
  }
  
  nnlsWork_.covDecomp.compute(nnlsWork_.invCovMat);
}

double MahiFit::calculateArrivalTime() {

  nnlsWork_.pulseDerivMat.resize(tsSize_,nnlsWork_.nPulseTot);

  int itIndex=0;

  for (unsigned int iBX=0; iBX<nnlsWork_.nPulseTot; iBX++) {
    int offset=nnlsWork_.bxs.coeff(iBX);
    if (offset==0) itIndex=iBX;

    if (offset==100) {
      nnlsWork_.pulseDerivMat.col(iBX) = SampleVector::Zero(tsSize_);
    }
    else {
      nnlsWork_.pulseDerivMat.col(iBX) = nnlsWork_.pulseDerivArray.at(offset+nnlsWork_.bxOffset).segment(fullTSOffset_-offset, tsSize_);
    }
  }

  PulseVector solution = nnlsWork_.pulseDerivMat.colPivHouseholderQr().solve(nnlsWork_.residuals);
  return solution.coeff(itIndex)/nnlsWork_.ampVec.coeff(itIndex);

}
  

void MahiFit::nnls() {
  const unsigned int npulse = nnlsWork_.nPulseTot;
  
  for (unsigned int iBX=0; iBX<npulse; iBX++) {
    int offset=nnlsWork_.bxs.coeff(iBX);
    if (offset==100) {
      nnlsWork_.pulseMat.col(iBX) = SampleVector::Ones();
    }
    else {
      nnlsWork_.pulseMat.col(iBX) = nnlsWork_.pulseShapeArray.at(offset+nnlsWork_.bxOffset).segment(fullTSOffset_-offset, tsSize_);
    }
  }

  nnlsWork_.invcovp = nnlsWork_.covDecomp.matrixL().solve(nnlsWork_.pulseMat);
  nnlsWork_.aTaMat = nnlsWork_.invcovp.transpose().lazyProduct(nnlsWork_.invcovp);
  nnlsWork_.aTbVec = nnlsWork_.invcovp.transpose().lazyProduct(nnlsWork_.covDecomp.matrixL().solve(nnlsWork_.amplitudes));
  
  int iter = 0;
  Index idxwmax = 0;
  double wmax = 0.0;
  double threshold = nnlsThresh_;
  
  while (true) {    
    if (iter>0 || nP_==0) {
      if ( nP_==std::min(npulse, tsSize_)) break;
      
      const unsigned int nActive = npulse - nP_;
      nnlsWork_.updateWork = nnlsWork_.aTbVec - nnlsWork_.aTaMat*nnlsWork_.ampVec;
      
      Index idxwmaxprev = idxwmax;
      double wmaxprev = wmax;
      wmax = nnlsWork_.updateWork.tail(nActive).maxCoeff(&idxwmax);
      
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

      nnlsWork_.ampvecpermtest = nnlsWork_.ampVec;
      
      eigenSolveSubmatrix(nnlsWork_.aTaMat,nnlsWork_.aTbVec,nnlsWork_.ampvecpermtest,nP_);

      //check solution
      bool positive = true;
      for (unsigned int i = 0; i < nP_; ++i)
        positive &= (nnlsWork_.ampvecpermtest(i) > 0);
      if (positive) {
        nnlsWork_.ampVec.head(nP_) = nnlsWork_.ampvecpermtest.head(nP_);
        break;
      } 

      //update parameter vector
      Index minratioidx=0;
      
      // no realizable optimization here (because it autovectorizes!)
      double minratio = std::numeric_limits<double>::max();
      for (unsigned int ipulse=0; ipulse<nP_; ++ipulse) {
	if (nnlsWork_.ampvecpermtest.coeff(ipulse)<=0.) {
	  const double c_ampvec = nnlsWork_.ampVec.coeff(ipulse);
	  const double ratio = c_ampvec/(c_ampvec-nnlsWork_.ampvecpermtest.coeff(ipulse));
	  if (ratio<minratio) {
	    minratio = ratio;
	    minratioidx = ipulse;
	  }
	}
      }
      nnlsWork_.ampVec.head(nP_) += minratio*(nnlsWork_.ampvecpermtest.head(nP_) - nnlsWork_.ampVec.head(nP_));
      
      //avoid numerical problems with later ==0. check
      nnlsWork_.ampVec.coeffRef(minratioidx) = 0.;
      
      nnlsConstrainParameter(minratioidx);
    }
   
    ++iter;

    //adaptive convergence threshold to avoid infinite loops but still
    //ensure best value is used
    if (iter%10==0) {
      threshold *= 10.;
    }
    
  }
  
}

void MahiFit::onePulseMinimize() {

  nnlsWork_.invcovp = nnlsWork_.covDecomp.matrixL().solve(nnlsWork_.pulseMat);

  SingleMatrix aTamatval = nnlsWork_.invcovp.transpose()*nnlsWork_.invcovp;
  SingleVector aTbvecval = nnlsWork_.invcovp.transpose()*nnlsWork_.covDecomp.matrixL().solve(nnlsWork_.amplitudes);

  nnlsWork_.ampVec.coeffRef(0) = std::max(0., aTbvecval.coeff(0)/aTamatval.coeff(0));


}

double MahiFit::calculateChiSq() {
  
  return (nnlsWork_.covDecomp.matrixL().solve(nnlsWork_.pulseMat*nnlsWork_.ampVec - nnlsWork_.amplitudes)).squaredNorm();
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
  pfunctor_ = std::unique_ptr<ROOT::Math::Functor>( new ROOT::Math::Functor(psfPtr_.get(),&FitterFuncs::PulseShapeFunctor::singlePulseShapeFunc, 3) );

}

void MahiFit::nnlsUnconstrainParameter(Index idxp) {
  nnlsWork_.aTaMat.col(nP_).swap(nnlsWork_.aTaMat.col(idxp));
  nnlsWork_.aTaMat.row(nP_).swap(nnlsWork_.aTaMat.row(idxp));
  nnlsWork_.pulseMat.col(nP_).swap(nnlsWork_.pulseMat.col(idxp));
  std::swap(nnlsWork_.aTbVec.coeffRef(nP_),nnlsWork_.aTbVec.coeffRef(idxp));
  std::swap(nnlsWork_.ampVec.coeffRef(nP_),nnlsWork_.ampVec.coeffRef(idxp));
  std::swap(nnlsWork_.bxs.coeffRef(nP_),nnlsWork_.bxs.coeffRef(idxp));
  ++nP_;
}

void MahiFit::nnlsConstrainParameter(Index minratioidx) {
  nnlsWork_.aTaMat.col(nP_-1).swap(nnlsWork_.aTaMat.col(minratioidx));
  nnlsWork_.aTaMat.row(nP_-1).swap(nnlsWork_.aTaMat.row(minratioidx));
  nnlsWork_.pulseMat.col(nP_-1).swap(nnlsWork_.pulseMat.col(minratioidx));
  std::swap(nnlsWork_.aTbVec.coeffRef(nP_-1),nnlsWork_.aTbVec.coeffRef(minratioidx));
  std::swap(nnlsWork_.ampVec.coeffRef(nP_-1),nnlsWork_.ampVec.coeffRef(minratioidx));
  std::swap(nnlsWork_.bxs.coeffRef(nP_-1),nnlsWork_.bxs.coeffRef(minratioidx));
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

void MahiFit::resetWorkspace() {

  nnlsWork_.nPulseTot=0;
  //nnlsWork_.bxSize=0;
  nnlsWork_.bxOffset=0;

  std::fill(std::begin(nnlsWork_.pulseCovArray), std::end(nnlsWork_.pulseCovArray), FullSampleMatrix::Zero());
  std::fill(std::begin(nnlsWork_.pulseShapeArray), std::end(nnlsWork_.pulseShapeArray), FullSampleVector::Zero());
  std::fill(std::begin(nnlsWork_.pulseDerivArray), std::end(nnlsWork_.pulseDerivArray), FullSampleVector::Zero());

  std::fill(std::begin(nnlsWork_.pulseN), std::end(nnlsWork_.pulseN), 0);
  std::fill(std::begin(nnlsWork_.pulseM), std::end(nnlsWork_.pulseM), 0);
  std::fill(std::begin(nnlsWork_.pulseP), std::end(nnlsWork_.pulseP), 0);

  nnlsWork_.amplitudes.setZero();
  nnlsWork_.bxs.setZero();
  nnlsWork_.invCovMat.setZero();
  nnlsWork_.noiseTerms.setZero();
  nnlsWork_.pulseMat.setZero();
  nnlsWork_.pulseDerivMat.setZero();
  nnlsWork_.residuals.setZero();
  nnlsWork_.ampVec.setZero();
  nnlsWork_.errVec.setZero();
  nnlsWork_.ampvecpermtest.setZero();
  nnlsWork_.invcovp.setZero();
  nnlsWork_.aTaMat.setZero();
  nnlsWork_.aTbVec.setZero();
  nnlsWork_.updateWork.setZero();
  //nnlsWork_.covDecomp.setZero();
  nnlsWork_.covDecompLinv.setZero();
  nnlsWork_.topleft_work.setZero();
  //nnlsWork_.pulseDecomp.setZero();



}
