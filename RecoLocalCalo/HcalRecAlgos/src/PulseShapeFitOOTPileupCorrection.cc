#include <iostream>
#include <cmath>
#include <climits>
#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFitOOTPileupCorrection.h"
#include "FWCore/Utilities/interface/isFinite.h"

namespace FitterFuncs{

  //Decalare the Pulse object take it in from Hcal and set some options
  PulseShapeFunctor::PulseShapeFunctor(const HcalPulseShapes::Shape& pulse,
				       bool iPedestalConstraint, bool iTimeConstraint,bool iAddPulseJitter,bool iAddTimeSlew,
				       double iPulseJitter,double iTimeMean,double iTimeSig,double iPedMean,double iPedSig,
				       double iNoise) : 
    cntNANinfit(0),
    acc25nsVec(HcalConst::maxPSshapeBin), diff25nsItvlVec(HcalConst::maxPSshapeBin),
    accVarLenIdxZEROVec(HcalConst::nsPerBX), diffVarItvlIdxZEROVec(HcalConst::nsPerBX), 
    accVarLenIdxMinusOneVec(HcalConst::nsPerBX), diffVarItvlIdxMinusOneVec(HcalConst::nsPerBX) {
    //The raw pulse
    for(int i=0;i<HcalConst::maxPSshapeBin;++i) pulse_hist[i] = pulse(i);
    // Accumulate 25ns for each starting point of 0, 1, 2, 3...
    for(int i=0; i<HcalConst::maxPSshapeBin; ++i){
      for(int j=i; j<i+HcalConst::nsPerBX; ++j){  //sum over HcalConst::nsPerBXns from point i
	acc25nsVec[i] += ( j < HcalConst::maxPSshapeBin? pulse_hist[j] : pulse_hist[HcalConst::maxPSshapeBin-1]);
      }
      diff25nsItvlVec[i] = ( i+HcalConst::nsPerBX < HcalConst::maxPSshapeBin? pulse_hist[i+HcalConst::nsPerBX] - pulse_hist[i] : pulse_hist[HcalConst::maxPSshapeBin-1] - pulse_hist[i]);
    }
    // Accumulate different ns for starting point of index either 0 or -1
    for(int i=0; i<HcalConst::nsPerBX; ++i){
      if( i==0 ){
	accVarLenIdxZEROVec[0] = pulse_hist[0];
	accVarLenIdxMinusOneVec[i] = pulse_hist[0];
      } else{
	accVarLenIdxZEROVec[i] = accVarLenIdxZEROVec[i-1] + pulse_hist[i];
	accVarLenIdxMinusOneVec[i] = accVarLenIdxMinusOneVec[i-1] + pulse_hist[i-1];
      }
      diffVarItvlIdxZEROVec[i] = pulse_hist[i+1] - pulse_hist[0];
      diffVarItvlIdxMinusOneVec[i] = pulse_hist[i] - pulse_hist[0];
    }
    for(int i = 0; i < HcalConst::maxSamples; i++) { 
      psFit_x[i]      = 0;
      psFit_y[i]      = 0;
      psFit_erry[i]   = 1.;
      psFit_erry2[i]  = 1.;
      psFit_slew [i]  = 0.;
    }
    //Constraints
    pedestalConstraint_ = iPedestalConstraint;
    timeConstraint_     = iTimeConstraint;
    addPulseJitter_     = iAddPulseJitter;
    pulseJitter_        = iPulseJitter*iPulseJitter;
    timeMean_           = iTimeMean;
    timeSig_            = iTimeSig;
    pedMean_            = iPedMean;
    pedSig_             = iPedSig;
    noise_              = iNoise;
    timeShift_          = 100.;
    timeShift_ += 12.5;//we are trying to get BX 

    inverttimeSig_ = 1./timeSig_;
    inverttimeSig2_ = inverttimeSig_*inverttimeSig_;
    invertpedSig_ = 1./pedSig_;
    invertpedSig2_ = invertpedSig_*invertpedSig_;
  }

  void PulseShapeFunctor::funcHPDShape(std::array<float,HcalConst::maxSamples> & ntmpbin, const double &pulseTime, const double &pulseHeight,const double &slew) { 
    // pulse shape components over a range of time 0 ns to 255 ns in 1 ns steps
    constexpr int ns_per_bx = HcalConst::nsPerBX;
    constexpr int num_ns = HcalConst::nsPerBX*HcalConst::maxSamples;
    constexpr int num_bx = num_ns/ns_per_bx;
    //Get the starting time
    int i_start         = ( -HcalConst::iniTimeShift - pulseTime - slew >0 ? 0 : (int)std::abs(-HcalConst::iniTimeShift-pulseTime-slew) + 1);
    float offset_start = i_start - HcalConst::iniTimeShift - pulseTime - slew; //-199-2*pars[0]-2.*slew (for pars[0] > 98.5) or just -98.5-pars[0]-slew;
    // zeroing output binned pulse shape
    ntmpbin = { {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };

    if( edm::isNotFinite(offset_start) ){ //Check for nan
      ++ cntNANinfit;
    }else{
      if( offset_start == 1.0 ){ offset_start = 0.; i_start-=1; } //Deal with boundary

      const int bin_start        = (int) offset_start; //bin off to integer
      const int bin_0_start      = ( offset_start < bin_start + 0.5 ? bin_start -1 : bin_start ); //Round it
      const int iTS_start        = i_start/ns_per_bx;         //Time Slice for time shift
      const int distTo25ns_start = HcalConst::nsPerBX - 1 - i_start%ns_per_bx;    //Delta ns 
      const float factor = offset_start - bin_0_start - 0.5; //Small correction?
    
      //Build the new pulse
      ntmpbin[iTS_start] = (bin_0_start == -1 ? // Initial bin (I'm assuming this is ok)
			      accVarLenIdxMinusOneVec[distTo25ns_start] + factor * diffVarItvlIdxMinusOneVec[distTo25ns_start]
			    : accVarLenIdxZEROVec    [distTo25ns_start] + factor * diffVarItvlIdxZEROVec    [distTo25ns_start]);
      //Fill the rest of the bins
      for(int iTS = iTS_start+1; iTS < num_bx; ++iTS){
	int bin_idx = distTo25ns_start + 1 + (iTS-iTS_start-1)*ns_per_bx + bin_0_start;
	ntmpbin[iTS] = acc25nsVec[bin_idx] + factor * diff25nsItvlVec[bin_idx];
      }
      //Scale the pulse 
      for(int i=iTS_start; i < num_bx; ++i) {
	ntmpbin[i]     *= pulseHeight;
      }

    }

    return;
  }

  PulseShapeFunctor::~PulseShapeFunctor() {
  }

  double PulseShapeFunctor::EvalPulse(const double *pars, unsigned int nPars) {
      constexpr unsigned nbins = (unsigned) HcalConst::maxSamples;
      unsigned i =0, j=0;
      //Stop crashes
      for(i =0; i < nPars; ++i ) if( edm::isNotFinite(pars[i]) ){ ++ cntNANinfit; return 1e10; }
      
      //calculate chisquare
      double chisq  = 0;
      unsigned int parBy2=(nPars-1)/2;
      //      std::array<float,HcalConst::maxSamples> pulse_shape_;

      if(addPulseJitter_) {
	int time = (pars[0]+timeShift_-timeMean_)*HcalConst::invertnsPerBx;
	//Interpolate the fit (Quickly)
	funcHPDShape(pulse_shape_, pars[0],pars[1],psFit_slew[time]);
	for (j=0; j<nbins; ++j) {
	  psFit_erry2[j]  = psFit_erry[j]*psFit_erry[j] + pulse_shape_[j]*pulse_shape_[j]*pulseJitter_;
	  pulse_shape_sum_[j] = pulse_shape_[j] + pars[nPars-1];
	}

	i=1;
	while (i<parBy2) {  
	  time = (pars[i*2]+timeShift_-timeMean_)*HcalConst::invertnsPerBx;
	  //Interpolate the fit (Quickly)
	  funcHPDShape(pulse_shape_, pars[i+2],pars[i*2+1],psFit_slew[time]);
	  // add an uncertainty from the pulse (currently noise * pulse height =>Ecal uses full cov)
	 /////
	  for (j=0; j<nbins; ++j) {
	    psFit_erry2[j] += pulse_shape_[j]*pulse_shape_[j]*pulseJitter_;
	    pulse_shape_sum_[j] += pulse_shape_[j];
	  }	    
	  i++;
	}
      }
      else{
	int time = (pars[0]+timeShift_-timeMean_)*HcalConst::invertnsPerBx;
	//Interpolate the fit (Quickly)
	funcHPDShape(pulse_shape_, pars[0],pars[1],psFit_slew[time]);
	for(j=0; j<nbins; ++j)
	  pulse_shape_sum_[j] = pulse_shape_[j] + pars[nPars-1];

	i=1;
	while (i<parBy2) {  
	  time = (pars[i*2]+timeShift_-timeMean_)*HcalConst::invertnsPerBx;
	  //Interpolate the fit (Quickly)
	  funcHPDShape(pulse_shape_, pars[i*2],pars[i*2+1],psFit_slew[time]);
	  // add an uncertainty from the pulse (currently noise * pulse height =>Ecal uses full cov)
	  for(j=0; j<nbins; ++j)
	    pulse_shape_sum_[j] += pulse_shape_[j];
	  i++;
	}
      }

      for (i=0;i<nbins; ++i) 
        chisq += (psFit_y[i]- pulse_shape_sum_[i])*(psFit_y[i]- pulse_shape_sum_[i])/psFit_erry2[i];

      if(pedestalConstraint_) {
	 //Add the pedestal Constraint to chi2
         chisq += invertpedSig2_*(pars[nPars-1] - pedMean_)*(pars[nPars-1]- pedMean_);
      }
        //Add the time Constraint to chi2
      if(timeConstraint_) {
	for(j=0; j< parBy2; ++j ){
	  int time = (pars[j*2]+timeShift_-timeMean_)*(double)HcalConst::invertnsPerBx;
	  double time1 = -100.+time*HcalConst::nsPerBX;
	  chisq += inverttimeSig2_*(pars[j*2] - timeMean_ - time1)*(pars[j*2] - timeMean_ - time1);
	}
      }
      return chisq;
   }

   double PulseShapeFunctor::singlePulseShapeFunc( const double *x ) {
      return EvalPulse(x,3);
   }
  
   double PulseShapeFunctor::doublePulseShapeFunc( const double *x ) {
      return EvalPulse(x,5);
   }
  
   double PulseShapeFunctor::triplePulseShapeFunc( const double *x ) {
      return EvalPulse(x,7);
   }
  //Greg's Hcal Binning => here to keep the const correctness below (channel discretization)
   double PulseShapeFunctor::sigma(double ifC) { 
      if(ifC < 75) return (0.577 + 0.0686*ifC)/3.; 
      return (2.75  + 0.0373*ifC + 3e-6*ifC*ifC)/3.; 
   }
  
}

PulseShapeFitOOTPileupCorrection::PulseShapeFitOOTPileupCorrection() : cntsetPulseShape(0), chargeThreshold_(6.),
								       psfPtr_(nullptr), spfunctor_(nullptr), dpfunctor_(nullptr), tpfunctor_(nullptr),
								       TSMin_(0), TSMax_(0), ts4Chi2_(0), ts3Chi2_(0), ts345Chi2_(0), pedestalConstraint_(0),
								       timeConstraint_(0), addPulseJitter_(0), unConstrainedFit_(0), applyTimeSlew_(0),
								       ts4Min_(0), ts4Max_(0), pulseJitter_(0), timeMean_(0), timeSig_(0), pedMean_(0), pedSig_(0),
								       noise_(0) {
   hybridfitter = new PSFitter::HybridMinimizer(PSFitter::HybridMinimizer::kMigrad);
   iniTimesArr = { {-100,-75,-50,-25,0,25,50,75,100,125} };
}

PulseShapeFitOOTPileupCorrection::~PulseShapeFitOOTPileupCorrection() { 
   if(hybridfitter) delete hybridfitter;
   if(spfunctor_)   delete spfunctor_;
   if(dpfunctor_)   delete dpfunctor_;
   if(tpfunctor_)   delete tpfunctor_;
}

void PulseShapeFitOOTPileupCorrection::setPUParams(bool   iPedestalConstraint, bool iTimeConstraint,bool iAddPulseJitter,
						   bool   iUnConstrainedFit,   bool iApplyTimeSlew,double iTS4Min, double iTS4Max,
						   double iPulseJitter,double iTimeMean,double iTimeSig,double iPedMean,double iPedSig,
						   double iNoise,double iTMin,double iTMax,
						   double its3Chi2,double its4Chi2,double its345Chi2,
						   double iChargeThreshold,HcalTimeSlew::BiasSetting slewFlavor, int iFitTimes) { 

  TSMin_ = iTMin;
  TSMax_ = iTMax;
  ts3Chi2_   = its3Chi2;
  ts4Chi2_   = its4Chi2;
  ts345Chi2_ = its345Chi2;
  pedestalConstraint_ = iPedestalConstraint;
  timeConstraint_     = iTimeConstraint;
  addPulseJitter_     = iAddPulseJitter;
  unConstrainedFit_   = iUnConstrainedFit;
  applyTimeSlew_      = iApplyTimeSlew;
  ts4Min_             = iTS4Min;
  ts4Max_             = iTS4Max;
  pulseJitter_        = iPulseJitter*iPulseJitter;
  timeMean_           = iTimeMean;
  timeSig_            = iTimeSig;
  pedMean_            = iPedMean;
  pedSig_             = iPedSig;
  noise_              = iNoise;
  slewFlavor_         = slewFlavor;
  chargeThreshold_    = iChargeThreshold;
  fitTimes_           = iFitTimes;
  if(unConstrainedFit_) { //Turn off all Constraints
    //pedestalConstraint_ = false; => Leaving this as tunable
    //timeConstraint_     = false;
    TSMin_ = -100.;
    TSMax_ =   75.;
  }
}

void PulseShapeFitOOTPileupCorrection::setPulseShapeTemplate(const HcalPulseShapes::Shape& ps) {

   if( cntsetPulseShape ) return;
   ++ cntsetPulseShape;
   psfPtr_.reset(new FitterFuncs::PulseShapeFunctor(ps,pedestalConstraint_,timeConstraint_,addPulseJitter_,applyTimeSlew_,
								 pulseJitter_,timeMean_,timeSig_,pedMean_,pedSig_,noise_));
   spfunctor_    = new ROOT::Math::Functor(psfPtr_.get(),&FitterFuncs::PulseShapeFunctor::singlePulseShapeFunc, 3);
   dpfunctor_    = new ROOT::Math::Functor(psfPtr_.get(),&FitterFuncs::PulseShapeFunctor::doublePulseShapeFunc, 5);
   tpfunctor_    = new ROOT::Math::Functor(psfPtr_.get(),&FitterFuncs::PulseShapeFunctor::triplePulseShapeFunc, 7);
}

void PulseShapeFitOOTPileupCorrection::resetPulseShapeTemplate(const HcalPulseShapes::Shape& ps) { 
   ++ cntsetPulseShape;
   psfPtr_.reset(new FitterFuncs::PulseShapeFunctor(ps,pedestalConstraint_,timeConstraint_,addPulseJitter_,applyTimeSlew_,
								 pulseJitter_,timeMean_,timeSig_,pedMean_,pedSig_,noise_));
   spfunctor_    = new ROOT::Math::Functor(psfPtr_.get(),&FitterFuncs::PulseShapeFunctor::singlePulseShapeFunc, 3);
   dpfunctor_    = new ROOT::Math::Functor(psfPtr_.get(),&FitterFuncs::PulseShapeFunctor::doublePulseShapeFunc, 5);
   tpfunctor_    = new ROOT::Math::Functor(psfPtr_.get(),&FitterFuncs::PulseShapeFunctor::triplePulseShapeFunc, 7);
}

void PulseShapeFitOOTPileupCorrection::apply(const CaloSamples & cs, const std::vector<int> & capidvec, const HcalCalibrations & calibs, std::vector<double> & correctedOutput) const
{
   psfPtr_->setDefaultcntNANinfit();

   const unsigned int cssize = cs.size();
// initialize arrays to be zero
   double chargeArr[HcalConst::maxSamples]={}, pedArr[HcalConst::maxSamples]={}, gainArr[HcalConst::maxSamples]={};
   double energyArr[HcalConst::maxSamples]={}, pedenArr[HcalConst::maxSamples]={};
   double tsTOT = 0, tstrig = 0; // in fC
   double tsTOTen = 0; // in GeV
   for(unsigned int ip=0; ip<cssize; ++ip){
      if( ip >= (unsigned) HcalConst::maxSamples ) continue; // Too many samples than what we wanna fit (10 is enough...) -> skip them
      const int capid = capidvec[ip];
      double charge = cs[ip];
      double ped = calibs.pedestal(capid);
      double gain = calibs.respcorrgain(capid);

      double energy = charge*gain;
      double peden = ped*gain;

      chargeArr[ip] = charge; pedArr[ip] = ped; gainArr[ip] = gain;
      energyArr[ip] = energy; pedenArr[ip] = peden;
      
      tsTOT += charge - ped;
      tsTOTen += energy - peden;
      if( ip ==4 || ip==5 ){
         tstrig += charge - ped;
      }
   }
   if( tsTOTen < 0. ) tsTOTen = pedSig_;
   std::vector<double> fitParsVec;
   if( tstrig >= ts4Min_ ) { //Two sigma from 0 
     pulseShapeFit(energyArr, pedenArr, chargeArr, pedArr, gainArr, tsTOTen, fitParsVec);
//     double time = fitParsVec[1], ampl = fitParsVec[0], uncorr_ampl = fitParsVec[0];
   }
   correctedOutput.swap(fitParsVec); correctedOutput.push_back(psfPtr_->getcntNANinfit());
}

constexpr char const* varNames[] = {"time", "energy","time1","energy1","time2","energy2", "ped"};

int PulseShapeFitOOTPileupCorrection::pulseShapeFit(const double * energyArr, const double * pedenArr, const double *chargeArr, const double *pedArr, const double *gainArr, const double tsTOTen, std::vector<double> &fitParsVec)  const {
   double tsMAX=0;
   int    nAboveThreshold = 0;
   double tmpx[HcalConst::maxSamples], tmpy[HcalConst::maxSamples], tmperry[HcalConst::maxSamples],tmperry2[HcalConst::maxSamples],tmpslew[HcalConst::maxSamples];
   double tstrig = 0; // in fC
   for(int i=0;i<HcalConst::maxSamples;++i){
      tmpx[i]=i;
      tmpy[i]=energyArr[i]-pedenArr[i];
      //Add Time Slew !!! does this need to be pedestal subtracted
      tmpslew[i] = 0;
      if(applyTimeSlew_) tmpslew[i] = HcalTimeSlew::delay(std::max(1.0,chargeArr[i]),slewFlavor_); 
      //Add Greg's channel discretization
      double sigmaBin =  psfPtr_->sigma(chargeArr[i]);
      tmperry2[i]=noise_*noise_+ sigmaBin*sigmaBin; //Greg's Granularity
      //Propagate it through
      tmperry2[i]*=(gainArr[i]*gainArr[i]); //Convert from fC to GeV
      tmperry [i]=sqrt(tmperry2[i]);
      //Add the Uncosntrained Double Pulse Switch
      if((chargeArr[i])>chargeThreshold_) nAboveThreshold++;
      if(std::abs(energyArr[i])>tsMAX) tsMAX=std::abs(tmpy[i]);
      if( i ==4 || i ==5 ){
         tstrig += chargeArr[i] - pedArr[i];
      }
   }
   psfPtr_->setpsFitx    (tmpx);
   psfPtr_->setpsFity    (tmpy);
   psfPtr_->setpsFiterry (tmperry);
   psfPtr_->setpsFiterry2(tmperry2);
   psfPtr_->setpsFitslew (tmpslew);
   
   //Fit 1 single pulse
   float timevalfit  = 0;
   float chargevalfit= 0;
   float pedvalfit   = 0;
   float chi2        = 999; //cannot be zero
   bool  fitStatus   = false;

   int BX[3] = {4,5,3};
   if(ts4Chi2_ != 0) fit(1,timevalfit,chargevalfit,pedvalfit,chi2,fitStatus,tsMAX,tsTOTen,tmpy,BX);
// Based on the pulse shape ( 2. likely gives the same performance )
   if(tmpy[2] > 3.*tmpy[3]) BX[2] = 2;
// Only do three-pulse fit when tstrig < ts4Max_, otherwise one-pulse fit is used (above)
   if(chi2 > ts4Chi2_ && !unConstrainedFit_ && tstrig < ts4Max_)   { //fails chi2 cut goes straight to 3 Pulse fit
     fit(3,timevalfit,chargevalfit,pedvalfit,chi2,fitStatus,tsMAX,tsTOTen,tmpy,BX);
   }
   if(unConstrainedFit_ && nAboveThreshold > 5) { //For the old method 2 do double pulse fit if values above a threshold
     fit(2,timevalfit,chargevalfit,pedvalfit,chi2,fitStatus,tsMAX,tsTOTen,tmpy,BX); 
   }
   /*
   if(chi2 > ts345Chi2_)   { //fails do two pulse chi2 for TS5 
     BX[1] = 5;
     fit(3,timevalfit,chargevalfit,pedvalfit,chi2,fitStatus,tsMAX,tsTOTen,BX);
   }
   */
   //Fix back the timeslew
   //if(applyTimeSlew_) timevalfit+=HcalTimeSlew::delay(std::max(1.0,chargeArr[4]),slewFlavor_);
   int outfitStatus = (fitStatus ? 1: 0 );
   fitParsVec.clear();
   fitParsVec.push_back(chargevalfit);
   fitParsVec.push_back(timevalfit);
   fitParsVec.push_back(pedvalfit);
   fitParsVec.push_back(chi2);
   return outfitStatus;
}

void PulseShapeFitOOTPileupCorrection::fit(int iFit,float &timevalfit,float &chargevalfit,float &pedvalfit,float &chi2,bool &fitStatus,double &iTSMax,const double &iTSTOTEn,double *iEnArr,int (&iBX)[3]) const { 
  int n = 3;
  if(iFit == 2) n = 5; //Two   Pulse Fit 
  if(iFit == 3) n = 7; //Three Pulse Fit 
  //Step 1 Single Pulse fit
   float pedMax =  iTSMax;   //=> max timeslice
   float tMin   =  TSMin_;   //Fitting Time Min
   float tMax   =  TSMax_;   //Fitting Time Max
   //Checks to make sure fitting happens
   if(pedMax   < 1.) pedMax = 1.;
   // Set starting values andf step sizes for parameters
   double vstart[n];
   for(int i = 0; i < int((n-1)/2); i++) { 
     vstart[2*i+0] = iniTimesArr[iBX[i]]+timeMean_;
     vstart[2*i+1] = iEnArr[iBX[i]];
   }
   vstart[n-1] = pedMean_;

   double step[n];
   for(int i = 0; i < n; i++) step[i] = 0.1;
      
   if(iFit == 1) hybridfitter->SetFunction(*spfunctor_);
   if(iFit == 2) hybridfitter->SetFunction(*dpfunctor_);
   if(iFit == 3) hybridfitter->SetFunction(*tpfunctor_);
   hybridfitter->Clear();
   //Times and amplitudes
   for(int i = 0; i < int((n-1)/2); i++) {
     hybridfitter->SetLimitedVariable(0+i*2, varNames[2*i+0]  , vstart[0+i*2],   step[0+i*2],iniTimesArr[iBX[i]]+tMin, iniTimesArr[ iBX[i] ]+tMax);
     hybridfitter->SetLimitedVariable(1+i*2, varNames[2*i+1]  , vstart[1+i*2],   step[1+i*2],      0, iTSTOTEn);
     //Secret Option to fix the time 
     if(timeSig_ < 0) hybridfitter->SetFixedVariable(0+i*2, varNames[2*i+0],vstart[0+i*2]);
   }
   //Pedestal
   if(vstart[n-1] > std::abs(pedMax)) vstart[n-1] = pedMax;
   hybridfitter->SetLimitedVariable(n-1, varNames[n-1], vstart[n-1], step[n-1],-pedMax,pedMax);
   //Secret Option to fix the pedestal
   if(pedSig_ < 0) hybridfitter->SetFixedVariable(n-1,varNames[n-1],vstart[n-1]);
   //a special number to label the initial condition
   chi2=-1;
   //3 fits why?!
   const double *results = 0;
   for(int tries=0; tries<=3;++tries){
     if( fitTimes_ != 2 || tries !=1 ){
        hybridfitter->SetMinimizerType(PSFitter::HybridMinimizer::kMigrad);
        fitStatus = hybridfitter->Minimize();
     }
     double chi2valfit = hybridfitter->MinValue();
     const double *newresults = hybridfitter->X();
     if(chi2 == -1 || chi2>chi2valfit+0.01) {
       results=newresults;
       chi2=chi2valfit;
       if( tries == 0 && fitTimes_ == 1 ) break;
       if( tries == 1 && (fitTimes_ == 2 || fitTimes_ ==3 ) ) break;
       if( tries == 2 && fitTimes_ == 4 ) break;
       if( tries == 3 && fitTimes_ == 5 ) break; 
       //Secret option to speed up the fit => perhaps we should drop this
       if(timeSig_ < 0 || pedSig_ < 0) break;
       if(tries==0){
	 hybridfitter->SetMinimizerType(PSFitter::HybridMinimizer::kScan);
	 fitStatus = hybridfitter->Minimize();
       } else if(tries==1){
	 hybridfitter->SetStrategy(1);
       } else if(tries==2){
	 hybridfitter->SetStrategy(2);
       }
     } else {
       break;
     }
   }
   assert(results);

   timevalfit   = results[0];
   chargevalfit = results[1];
   pedvalfit    = results[n-1];
   if(!(unConstrainedFit_ && iFit == 2)) return;
   //Add the option of the old method 2
   float timeval2fit   = results[2];
   float chargeval2fit = results[3];
   if(std::abs(timevalfit)>std::abs(timeval2fit)) {// if timeval1fit and timeval2fit are differnt, choose the one which is closer to zero
     timevalfit=timeval2fit;
     chargevalfit=chargeval2fit;
   } else if(timevalfit==timeval2fit) { // if the two times are the same, then for charge we just sum the two  
     timevalfit=(timevalfit+timeval2fit)/2;
     chargevalfit=chargevalfit+chargeval2fit;
   } else {
     timevalfit=-999.;
     chargevalfit=-999.;
   }
}
