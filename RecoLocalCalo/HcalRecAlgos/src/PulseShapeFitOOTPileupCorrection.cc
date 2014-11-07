#include <iostream>
#include <cmath>
#include <climits>
#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFitOOTPileupCorrection.h"

namespace FitterFuncs{

  int cntNANinfit;
  double psFit_x[10], psFit_y[10], psFit_slew[10],psFit_erry[10], psFit_erry2[10];
  //const std::vector<double>& pars,                     
  std::array<float,10> funcHPDShape(
				    const double &pulseTime, const double &pulseHeight,const double &slew,const std::array<float,256>& h1_single, 
				    const std::vector<float> &acc25nsVec,              const std::vector<float> &diff25nsItvlVec, 
				    const std::vector<float> &accVarLenIdxZEROVec,     const std::vector<float> &diffVarItvlIdxZEROVec, 
				    const std::vector<float> &accVarLenIdxMinusOneVec, const std::vector<float>&diffVarItvlIdxMinusOneVec) {
    // pulse shape components over a range of time 0 ns to 255 ns in 1 ns steps
    constexpr int ns_per_bx = 25;
    constexpr int num_ns = 250;
    constexpr int num_bx = num_ns/ns_per_bx;
    // zeroing output binned pulse shape
    std::array<float,num_bx> ntmpbin{ {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };
    //Get the starting time
    int i_start         = ( -98.5f - pulseTime - slew >0 ? 0 : (int)fabs(-98.5f-pulseTime-slew) + 1);
    double offset_start = i_start - 98.5f - pulseTime - slew; //-199-2*pars[0]-2.*slew (for pars[0] > 98.5) or just -98.5-pars[0]-slew;
    if( offset_start == 1.0 ){ offset_start = 0.; i_start-=1; } //Deal with boundary
    const int bin_start        = (int) offset_start; //bin off to integer
    const int bin_0_start      = ( offset_start < bin_start + 0.5 ? bin_start -1 : bin_start ); //Round it
    const int iTS_start        = i_start/ns_per_bx;         //Time Slice for time shift
    const int distTo25ns_start = 24 - i_start%ns_per_bx;    //Delta ns 
    const double factor = offset_start - bin_0_start - 0.5; //Small correction?
    
    if( offset_start != offset_start){ //Check for nan
      cntNANinfit ++;
    }else{
      //Build the new pulse
      ntmpbin[iTS_start] = (bin_0_start == -1 ? // Initial bin (I'm assuming this is ok)
			      accVarLenIdxMinusOneVec[distTo25ns_start] + factor * diffVarItvlIdxMinusOneVec[distTo25ns_start]
			    : accVarLenIdxZEROVec    [distTo25ns_start] + factor * diffVarItvlIdxZEROVec    [distTo25ns_start]);
      //Fill the rest of the bins
      for(int iTS = iTS_start+1; iTS < num_bx; ++iTS){
	int bin_idx = distTo25ns_start + 1 + (iTS-iTS_start-1)*ns_per_bx + bin_0_start;
	ntmpbin[iTS] = acc25nsVec[bin_idx] + factor * diff25nsItvlVec[bin_idx];
      }
    }
    //Scale the pulse and add a pedestal
    for(int i=0; i < num_bx; ++i) {
      ntmpbin[i]     *= pulseHeight;
    }
    return ntmpbin;
  }
  //Decalare the Pulse object take it in from Hcal and set some options
  PulseShapeFunctor::PulseShapeFunctor(const HcalPulseShapes::Shape& pulse,
				       bool iPedestalConstraint, bool iTimeConstraint,bool iAddPulseJitter,
				       double iPulseJitter,double iTimeMean,double iTimeSig,double iPedMean,double iPedSig,
				       double iNoise) :
    acc25nsVec(256), diff25nsItvlVec(256),
    accVarLenIdxZEROVec(25), diffVarItvlIdxZEROVec(25), 
    accVarLenIdxMinusOneVec(25), diffVarItvlIdxMinusOneVec(25) {
    //The raw pulse
    for(int i=0;i<256;++i) pulse_hist[i] = pulse(i);
    //Integrate in 25ns bins for each 1ns bin  0, 1, 2, 3...
    for(int i=0; i<256; ++i){ 
      for(int j=i; j<i+25; ++j){ //sum over 25ns from point i
	acc25nsVec[i] += ( j < 256? pulse_hist[j] : pulse_hist[255]);
      }
      //Take dy/dt (dt = 25ns) 
      diff25nsItvlVec[i] = ( i+25 < 256? pulse_hist[i+25] - pulse_hist[i] : pulse_hist[255] - pulse_hist[i]);
    }
    // Sum over 25  ns for starting point of index either i=  0 or -1
    for(int i=0; i<25; ++i){
      if( i==0 ){
	accVarLenIdxZEROVec[0]     = pulse_hist[0];
	accVarLenIdxMinusOneVec[i] = pulse_hist[0];
      } else{
	//Sum over 25ns
	accVarLenIdxZEROVec[i]     = accVarLenIdxZEROVec    [i-1] + pulse_hist[i];
	accVarLenIdxMinusOneVec[i] = accVarLenIdxMinusOneVec[i-1] + pulse_hist[i-1];
      }
      //Diff from 0 for (i) and (i-1)
      diffVarItvlIdxZEROVec[i]     = pulse_hist[i+1] - pulse_hist[0];
      diffVarItvlIdxMinusOneVec[i] = pulse_hist[i]   - pulse_hist[0];
    }
    //Time slew model
    pedestalConstraint_ = iPedestalConstraint;
    timeConstraint_     = iTimeConstraint;
    addPulseJitter_     = iAddPulseJitter;
    pulseJitter_        = iPulseJitter;
    timeMean_           = iTimeMean;
    timeSig_            = iTimeSig;
    pedMean_            = iPedMean;
    pedSig_             = iPedSig;
    noise_              = iNoise;
    for(int i=0; i < 10; i++) psFit_erry [i] = noise_;
    for(int i=0; i < 10; i++) psFit_erry2[i] = noise_*noise_;
  }
  
  PulseShapeFunctor::~PulseShapeFunctor() {
  }
  double PulseShapeFunctor::EvalSinglePulse(const std::vector<double>& pars) const {
      constexpr unsigned nbins = 10;
      unsigned i =0;

      //calculate chisquare
      double chisq  = 0;
      double delta2 =0;
      int    time   = (pars[0]+113.)/25.; //Please note 13 is approximation for the time slew => we are trying to get BX
      double time1  = -100.+time*25.;     //Get the center time in the BX for the likelihood below
      //Stop crashes
      if(std::isnan(pars[0]) || std::isnan(pars[1]) || std::isnan(pars[2]) ) return 1e10;
      //Interpolate the fit (Quickly)
      std::array<float,nbins> pulse_shape = std::move(funcHPDShape(pars[0],pars[1],psFit_slew[time],pulse_hist,       //Basic Inputs 
								   acc25nsVec,              diff25nsItvlVec,          //Integral and diff of pulse
								   accVarLenIdxZEROVec,     diffVarItvlIdxZEROVec,    //Integral and diff (i)
								   accVarLenIdxMinusOneVec, diffVarItvlIdxMinusOneVec));//Integral and diff (i-1)
								   
      // add an uncertainty from the pulse (currently noise * pulse height =>Ecal uses full cov)
      if(addPulseJitter_) { 
	for (i=0;i<nbins; ++i) {
	  psFit_erry2[i] = pulse_shape[i]*pulse_shape[i]*pulseJitter_*pulseJitter_ + psFit_erry[i]*psFit_erry[i];
	}
      }
      //Add Pedestal outside of interpolate
      for(i = 0; i < nbins; i++)  pulse_shape[i] += pars[2];
      //Compute chi2
      for (i=0;i<nbins; ++i) {
	delta2 = (psFit_y[i]- pulse_shape[i]);
	delta2 = delta2*delta2;
	delta2 = delta2/psFit_erry2[i];
	chisq += delta2;
	//Add the pedestal Constraint to chi2
	if(pedestalConstraint_) {
	  chisq += ((pars[2]-pedMean_)  /pedSig_)*((pars[2]- pedMean_)/pedSig_);
	}
	//Add the time Constraint to chi2
	if(timeConstraint_) {
	  chisq += ((pars[0]-timeMean_-time1)/timeSig_)*((pars[0]-timeMean_-time1)/timeSig_);
	}
      }
      return chisq;
   }

   double PulseShapeFunctor::EvalDoublePulse(const std::vector<double>& pars) const {
      constexpr unsigned nbins = 10;
      unsigned i =0;

      //Stop crashes
      if(std::isnan(pars[0]) || std::isnan(pars[1]) || std::isnan(pars[2]) ||  std::isnan(pars[3]) ||  std::isnan(pars[4])) return 1e10;

      //calculate chisquare
      double chisq = 0;
      double delta2 = 0;
      int    time   = (pars[0]+113.)/25.; //Please note 13 is approximation for the time slew => we are trying to get BX
      double time1  = -100.+time*25.;     //Get the center time in the BX for the likelihood below
      std::array<float,nbins> pulse_shape1 = std::move(funcHPDShape(pars[0],pars[1],psFit_slew[time],pulse_hist,                  //Basic Inputs 
								    acc25nsVec,              diff25nsItvlVec,          //Integral and diff of pulse
								    accVarLenIdxZEROVec,     diffVarItvlIdxZEROVec,    //Integral and diff (i)
								    accVarLenIdxMinusOneVec, diffVarItvlIdxMinusOneVec));//Integral and diff (i-1)
      
      time         = (pars[2]+113.)/25.; //The 13 is to adjust for the slew shift at 0 energy
      double time2 = -100. + time*25.;
      std::array<float,nbins> pulse_shape2 = std::move(funcHPDShape(pars[2],pars[3],psFit_slew[time],pulse_hist,                  //Basic Inputs 
								    acc25nsVec,              diff25nsItvlVec,          //Integral and diff of pulse
								    accVarLenIdxZEROVec,     diffVarItvlIdxZEROVec,    //Integral and diff (i)
								    accVarLenIdxMinusOneVec, diffVarItvlIdxMinusOneVec));//Integral and diff (i-1)
      
      // add an uncertainty from the pulse (currently noise * pulse height =>Ecal uses full cov)
      if(addPulseJitter_) { 
	for (i=0;i<nbins; ++i) {
	  psFit_erry2[i]  = psFit_erry  [i]*psFit_erry  [i];
	  psFit_erry2[i] += pulse_shape1[i]*pulse_shape1[i]*pulseJitter_*pulseJitter_;
	  psFit_erry2[i] += pulse_shape2[i]*pulse_shape2[i]*pulseJitter_*pulseJitter_;
	}
      }
      //Add Pedestal outside of interpolate
      for(i = 0; i < nbins; i++)  pulse_shape1[i] += pulse_shape2[i];
      for(i = 0; i < nbins; i++)  pulse_shape1[i] += pars[4];
      //Compute chi2
      for (i=0;i<nbins; ++i) {
	delta2 = (psFit_y[i]- pulse_shape1[i])*(psFit_y[i]- pulse_shape1[i])/psFit_erry2[i];
	chisq += delta2;
	//Add the pedestal Constraint to chi2
	if(pedestalConstraint_) {
	  chisq += ((pars[4]-pedMean_)  /pedSig_)*((pars[4]- pedMean_)/pedSig_);
	}
	//Add the time Constraint to chi2
	if(timeConstraint_) {
	  chisq += ((pars[0]-timeMean_-time1)/timeSig_)*((pars[0]-timeMean_-time1)/timeSig_);
	  chisq += ((pars[2]-timeMean_-time2)/timeSig_)*((pars[2]-timeMean_-time2)/timeSig_);
	}
      }
      return chisq;
   }

   double PulseShapeFunctor::EvalTriplePulse(const std::vector<double>& pars) const {
      constexpr unsigned nbins = 10;
      unsigned i =0;
      if(std::isnan(pars[0]) || std::isnan(pars[1]) || std::isnan(pars[2]) ||  
	 std::isnan(pars[3]) || std::isnan(pars[4]) || std::isnan(pars[5]) ||  std::isnan(pars[6])   ) return 1e10;
      
      //calculate chisquare
      double chisq  = 0;
      double delta2 = 0;
      //double val[1];
      int    time  = (pars[0]+113.)/25.;
      double time1 = -100. + time*25.;
      std::array<float,nbins> pulse_shape1 = std::move(funcHPDShape(pars[0],pars[1],psFit_slew[time],pulse_hist,      //Basic Inputs 
								    acc25nsVec,              diff25nsItvlVec,          //Integral and diff of pulse
								    accVarLenIdxZEROVec,     diffVarItvlIdxZEROVec,    //Integral and diff (i)
								    accVarLenIdxMinusOneVec, diffVarItvlIdxMinusOneVec));//Integral and diff (i-1)

      time         = (pars[2]+113.)/25.; //The 13 is to adjust for the slew shift at 0 energy
      double time2 = -100. + time*25.;
      std::array<float,nbins> pulse_shape2 = std::move(funcHPDShape(pars[2],pars[3],psFit_slew[time],pulse_hist,      //Basic Inputs 
								    acc25nsVec,              diff25nsItvlVec,          //Integral and diff of pulse
								    accVarLenIdxZEROVec,     diffVarItvlIdxZEROVec,    //Integral and diff (i)
								    accVarLenIdxMinusOneVec, diffVarItvlIdxMinusOneVec));//Integral and diff (i-1)

      time         = (pars[4]+113.)/25.;
      double time3 = -100. + time*25.;
      std::array<float,nbins> pulse_shape3 = std::move(funcHPDShape(pars[4],pars[5],psFit_slew[time],pulse_hist,       //Basic Inputs 
								    acc25nsVec,              diff25nsItvlVec,          //Integral and diff of pulse
								    accVarLenIdxZEROVec,     diffVarItvlIdxZEROVec,    //Integral and diff (i)
								    accVarLenIdxMinusOneVec, diffVarItvlIdxMinusOneVec));//Integral and diff (i-1)

      // add an uncertainty from the pulse (currently noise * pulse height =>Ecal uses full cov)
      if(addPulseJitter_) { 
	for (i=0;i<nbins; ++i) {
	  psFit_erry2[i]  = psFit_erry  [i]*psFit_erry  [i];
	  psFit_erry2[i] += pulse_shape1[i]*pulse_shape1[i]*pulseJitter_*pulseJitter_;
	  psFit_erry2[i] += pulse_shape2[i]*pulse_shape2[i]*pulseJitter_*pulseJitter_;
	  psFit_erry2[i] += pulse_shape3[i]*pulse_shape3[i]*pulseJitter_*pulseJitter_;
	}
      }
      //Add Pedestal and other pulses outside of interpolate
      for(i = 0; i < nbins; i++)  pulse_shape1[i] += pulse_shape2[i];
      for(i = 0; i < nbins; i++)  pulse_shape1[i] += pulse_shape3[i];
      for(i = 0; i < nbins; i++)  pulse_shape1[i] += pars[6];
      //Compute chi2
      for (i=0;i<nbins; ++i) {
	delta2 = (psFit_y[i]- pulse_shape1[i])*(psFit_y[i]- pulse_shape1[i])/psFit_erry2[i];
	chisq += delta2;
	//Add the pedestal Constraint to chi2
	if(pedestalConstraint_) {
	  chisq += ((pars[6]-pedMean_)  /pedSig_)*((pars[6]- pedMean_)/pedSig_);
	}
	//Add the time Constraint to chi2
	if(timeConstraint_) {
	  chisq += ((pars[0]-timeMean_-time1)/timeSig_)*((pars[0]-timeMean_-time1)/timeSig_);
	  chisq += ((pars[2]-timeMean_-time2)/timeSig_)*((pars[2]-timeMean_-time2)/timeSig_);
	  chisq += ((pars[4]-timeMean_-time3)/timeSig_)*((pars[4]-timeMean_-time3)/timeSig_);
	}
      }
      return chisq;
   }
 
   std::auto_ptr<PulseShapeFunctor> psfPtr_;

   double singlePulseShapeFunc( const double *x ) {
      std::vector<double> pars(x, x+3);
      return psfPtr_->EvalSinglePulse(pars);
   }

   double doublePulseShapeFunc( const double *x ) {
      std::vector<double> pars(x, x+5);
      return psfPtr_->EvalDoublePulse(pars);
   }

   double triplePulseShapeFunc( const double *x ) {
      std::vector<double> pars(x, x+7);
      return psfPtr_->EvalTriplePulse(pars);
   }
  //Greg's Hcal Binning => here to keep the const correctness below
  double sigma(double ifC) { 
    if(ifC < 75) return (0.577 + 0.0686*ifC)/3.; 
    return (2.75  + 0.0373*ifC + 3e-6*ifC*ifC)/3.; 
  }

  }

PulseShapeFitOOTPileupCorrection::PulseShapeFitOOTPileupCorrection() : cntsetPulseShape(0), chargeThreshold_(6.)
{
   hybridfitter = new PSFitter::HybridMinimizer(PSFitter::HybridMinimizer::kMigrad);
   spfunctor_    = new ROOT::Math::Functor(&FitterFuncs::singlePulseShapeFunc, 3);
   dpfunctor_    = new ROOT::Math::Functor(&FitterFuncs::doublePulseShapeFunc, 5);
   tpfunctor_    = new ROOT::Math::Functor(&FitterFuncs::triplePulseShapeFunc, 7);
   iniTimesArr = { {-100,-75,-50,-25,0,25,50,75,100,125} };
}

PulseShapeFitOOTPileupCorrection::~PulseShapeFitOOTPileupCorrection()
{ 

   if(hybridfitter) delete hybridfitter;
   if(spfunctor_)   delete spfunctor_;
   if(dpfunctor_)   delete dpfunctor_;
   if(tpfunctor_)   delete tpfunctor_;
}
void PulseShapeFitOOTPileupCorrection::setPUParams(bool   iPedestalConstraint, bool iTimeConstraint,bool iAddPulseJitter,
						   double iPulseJitter,double iTimeMean,double iTimeSig,double iPedMean,double iPedSig,
						   double iNoise,double iTMin,double iTMax,
						   double its3Chi2,double its4Chi2,double its345Chi2,HcalTimeSlew::BiasSetting slewFlavor) { 

  TSMin_ = iTMin;
  TSMax_ = iTMax;
  ts3Chi2_   = its3Chi2;
  ts4Chi2_   = its4Chi2;
  ts345Chi2_ = its345Chi2;
  pedestalConstraint_ = iPedestalConstraint;
  timeConstraint_     = iTimeConstraint;
  addPulseJitter_     = iAddPulseJitter;
  pulseJitter_        = iPulseJitter;
  timeMean_           = iTimeMean;
  timeSig_            = iTimeSig;
  pedMean_            = iPedMean;
  pedSig_             = iPedSig;
  noise_              = iNoise;
  slewFlavor_         = slewFlavor;
}
void PulseShapeFitOOTPileupCorrection::setPulseShapeTemplate(const HcalPulseShapes::Shape& ps) {

   if( cntsetPulseShape ) return;
   ++ cntsetPulseShape;
   FitterFuncs::psfPtr_.reset(new FitterFuncs::PulseShapeFunctor(ps,pedestalConstraint_,timeConstraint_,addPulseJitter_,
								 pulseJitter_,timeMean_,timeSig_,pedMean_,pedSig_,noise_));
}
void PulseShapeFitOOTPileupCorrection::resetPulseShapeTemplate(const HcalPulseShapes::Shape& ps) { 
   ++ cntsetPulseShape;
   FitterFuncs::psfPtr_.reset(new FitterFuncs::PulseShapeFunctor(ps,pedestalConstraint_,timeConstraint_,addPulseJitter_,
								 pulseJitter_,timeMean_,timeSig_,pedMean_,pedSig_,noise_));
}

void PulseShapeFitOOTPileupCorrection::apply(const CaloSamples & cs, const std::vector<int> & capidvec, const HcalCalibrations & calibs, std::vector<double> & correctedOutput) const
{
   FitterFuncs::cntNANinfit = 0;

   const unsigned int cssize = cs.size();
   double chargeArr[cssize], pedArr[cssize];
   double energyArr[cssize], pedenArr[cssize];
   double tsTOT = 0, tstrig = 0; // in fC
   double tsTOTen = 0; // in GeV
   for(unsigned int ip=0; ip<cssize; ++ip){
      const int capid = capidvec[ip];
      double charge = cs[ip];
      double ped = calibs.pedestal(capid);
      double gain = calibs.respcorrgain(capid);

      double energy = charge*gain;
      double peden = ped*gain;

      chargeArr[ip] = charge; pedArr[ip] = ped;
      energyArr[ip] = energy; pedenArr[ip] = peden;
      
      tsTOT += charge - ped;
      tsTOTen += energy - peden;
      if( ip ==4 ){
         tstrig = charge - ped;
      }
   }
   std::vector<double> fitParsVec;
   if( tstrig >= 4 && tsTOT >= 10 ){
     pulseShapeFit(energyArr, pedenArr, chargeArr, pedArr, tsTOTen, fitParsVec);
     //      double time = fitParsVec[1], ampl = fitParsVec[0], uncorr_ampl = fitParsVec[0];
   }
   correctedOutput.swap(fitParsVec); correctedOutput.push_back(FitterFuncs::cntNANinfit);
}

constexpr char const* varNames[] = {"time", "energy","time1","energy1","time2","energy2", "ped"};

int PulseShapeFitOOTPileupCorrection::pulseShapeFit(const double * energyArr, const double * pedenArr, const double *chargeArr, const double *pedArr, const double tsTOTen, std::vector<double> &fitParsVec)  const {
   double tsMAX=0;
   for(int i=0;i<10;++i){
      FitterFuncs::psFit_x[i]=i;
      FitterFuncs::psFit_y[i]=energyArr[i]-pedenArr[i];
      //Add Time Slew
      FitterFuncs::psFit_slew[i] = HcalTimeSlew::delay(std::max(1.0,chargeArr[i]),slewFlavor_); // !!! does this need to be pedestal subtracted
      //Add Greg's channel discretization
      double sigmaBin =  FitterFuncs::sigma(chargeArr[i]);
      FitterFuncs::psFit_erry2[i]=noise_*noise_+ sigmaBin*sigmaBin; //Greg's Granularity
      //Propagate it through
      FitterFuncs::psFit_erry2[i]*=(energyArr[i]/chargeArr[i])*(energyArr[i]/chargeArr[i]); //Convert from fC to GeV
      FitterFuncs::psFit_erry [i]=sqrt(FitterFuncs::psFit_erry2[i]); //Formally, I should take a max of the above instead of quadrature right?
   }
   for(int i=0;i<10;++i){
     if(fabs(energyArr[i])>tsMAX){
       tsMAX=fabs(FitterFuncs::psFit_y[i]);
      }
   }
   
   //Fit 1 single pulse
   float timevalfit  = 0;
   float chargevalfit= 0;
   float pedvalfit   = 0;
   float chi2        = 0;
   bool  fitStatus   = false;
   int BX[3] = {4,5,3};
   fit(  1,timevalfit,chargevalfit,pedvalfit,chi2,fitStatus,tsMAX,tsTOTen,BX);
   if(FitterFuncs::psFit_y[2] > 3.*FitterFuncs::psFit_y[3]) BX[2] = 2;
   if(chi2 > ts4Chi2_)   { //fails chi2 cut goes straight to 3 Pulse fit
     fit(3,timevalfit,chargevalfit,pedvalfit,chi2,fitStatus,tsMAX,tsTOTen,BX);
   }
   /*
   if(chi2 > ts345Chi2_) { //fails do two pulse chi2 for TS3
     fit(2,timevalfit,chargevalfit,pedvalfit,chi2,fitStatus,tsMAX,tsTOTen,BX); 
   }
   if(chi2 > ts345Chi2_)   { //fails do two pulse chi2 for TS5 
     BX[1] = 5;
     fit(3,timevalfit,chargevalfit,pedvalfit,chi2,fitStatus,tsMAX,tsTOTen,BX);
   }
   */
   //Fix back the timeslew
   timevalfit+=HcalTimeSlew::delay(std::max(1.0,chargeArr[4]),slewFlavor_);
   int outfitStatus = (fitStatus ? 1: 0 );
   fitParsVec.clear();
   fitParsVec.push_back(chargevalfit);
   fitParsVec.push_back(timevalfit);
   fitParsVec.push_back(pedvalfit);
   fitParsVec.push_back(chi2);
   return outfitStatus;
}
void PulseShapeFitOOTPileupCorrection::fit(int iFit,float &timevalfit,float &chargevalfit,float &pedvalfit,float &chi2,bool &fitStatus,double &iTSMax,const double &iTSTOTEn,int (&iBX)[3]) const { 
  int n = 3;
  if(iFit == 2) n = 5; //Two   Pulse Fit 
  if(iFit == 3) n = 7; //Three Pulse Fit 
  //Step 1 Single Pulse fit
   float pedMax =  iTSMax;   //=> max timeslice
   float tMin   =  TSMin_;   //Fitting Time Min
   float tMax   =  TSMax_;   //Fitting Time Max
   
   // Set starting values andf step sizes for parameters
   double vstart[n];
   for(int i = 0; i < int((n-1)/2); i++) { 
     vstart[2*i+0] = iniTimesArr[iBX[i] ]+timeMean_;
     vstart[2*i+1] = FitterFuncs::psFit_y[iBX[i]];
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
   }
   //Pedestal
   if(vstart[n-1] > fabs(pedMax)) vstart[n-1] = pedMax;
   hybridfitter->SetLimitedVariable(n-1, varNames[n-1], vstart[n-1], step[n-1],    -pedMax, pedMax);
   chi2=9999.;
   //3 fits why?!
   const double *results = 0;
   for(int tries=0; tries<=3;++tries){
     hybridfitter->SetMinimizerType(PSFitter::HybridMinimizer::kMigrad);
     fitStatus = hybridfitter->Minimize();
     double chi2valfit = hybridfitter->MinValue();
     const double *newresults = hybridfitter->X();
     if(chi2>chi2valfit+0.01) {
       results=newresults;
       chi2=chi2valfit;
       if(tries==0){
	 hybridfitter->SetMinimizerType(PSFitter::HybridMinimizer::kScan);
	 hybridfitter->Minimize();
       } else if(tries==1){
	 hybridfitter->SetStrategy(1);
       } else if(tries==2){
	 hybridfitter->SetStrategy(2);
       }
     } else {
       break;
     }
   }
   timevalfit   = results[0];
   chargevalfit = results[1];
   pedvalfit    = results[n-1];
}
