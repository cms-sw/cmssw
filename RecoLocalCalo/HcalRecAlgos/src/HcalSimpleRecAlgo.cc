#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSimpleRecAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include <algorithm> // for "max"
#include <math.h>

static double MaximumFractionalError = 0.0005; // 0.05% error allowed from this source

HcalSimpleRecAlgo::HcalSimpleRecAlgo(int firstSample, int samplesToAdd, bool correctForTimeslew, bool correctForPulse, float phaseNS) : 
  firstSample_(firstSample), 
  samplesToAdd_(samplesToAdd), 
  correctForTimeslew_(correctForTimeslew) {
  if (correctForPulse) 
    pulseCorr_=std::auto_ptr<HcalPulseContainmentCorrection>(new HcalPulseContainmentCorrection(samplesToAdd_,phaseNS,MaximumFractionalError));
}

HcalSimpleRecAlgo::HcalSimpleRecAlgo(int firstSample, int samplesToAdd) : 
  firstSample_(firstSample), 
  samplesToAdd_(samplesToAdd), 
  correctForTimeslew_(false) {
}

///Timeshift correction for HPDs based on the position of the peak ADC measurement.
///  Allows for an accurate determination of the relative phase of the pulse shape from
///  the HPD.  Calculated based on a weighted sum of the -1,0,+1 samples relative to the peak
///  as follows:  wpksamp = (0*sample[0] + 1*sample[1] + 2*sample[2]) / (sample[0] + sample[1] + sample[2])
///  where sample[1] is the maximum ADC sample value.
static float timeshift_ns_hbheho(float wpksamp);

///Same as above, but for the HF PMTs.
static float timeshift_ns_hf(float wpksamp);


namespace HcalSimpleRecAlgoImpl {
  template<class Digi, class RecHit>
  inline RecHit reco(const Digi& digi, const HcalCoder& coder, const HcalCalibrations& calibs, 
		     int ifirst, int n, bool slewCorrect, const HcalPulseContainmentCorrection* corr, HcalTimeSlew::BiasSetting slewFlavor) {
    CaloSamples tool;
    coder.adc2fC(digi,tool);

    double ampl=0; int maxI = -1; double maxA = -1e10; float ta=0;
    double fc_ampl=0;
    for (int i=ifirst; i<tool.size() && i<n+ifirst; i++) {
      int capid=digi[i].capid();
      ta = (tool[i]-calibs.pedestal(capid)); // pedestal subtraction
      fc_ampl+=ta; 
      ta*= calibs.respcorrgain(capid) ; // fC --> GeV
      ampl+=ta;
      if(ta>maxA){
	maxA=ta;
	maxI=i;
      }
    }

    float time=-9999;
    ////Cannot calculate time value with max ADC sample at first or last position in window....
    if(maxI==0 || maxI==(tool.size()-1)) {      
      LogDebug("HCAL Pulse") << "HcalSimpleRecAlgo::reconstruct :" 
					       << " Invalid max amplitude position, " 
					       << " max Amplitude: "<< maxI
					       << " first: "<<ifirst
					       << " last: "<<(tool.size()-1)
					       << std::endl;
    } else {
      maxA=fabs(maxA);
      int capid=digi[maxI-1].capid();
      float t0 = fabs((tool[maxI-1]-calibs.pedestal(capid))*calibs.respcorrgain(capid) );
      capid=digi[maxI+1].capid();
      float t2 = fabs((tool[maxI+1]-calibs.pedestal(capid))*calibs.respcorrgain(capid) );    
      float wpksamp = (t0 + maxA + t2);
      if (wpksamp!=0) wpksamp=(maxA + 2.0*t2) / wpksamp; 
      time = (maxI - digi.presamples())*25.0 + timeshift_ns_hbheho(wpksamp);

      if (corr!=0) {
	// Apply phase-based amplitude correction:
	ampl *= corr->getCorrection(fc_ampl);
	//      std::cout << fc_ampl << " --> " << corr->getCorrection(fc_ampl) << std::endl;
      }

    
      if (slewCorrect) time-=HcalTimeSlew::delay(std::max(1.0,fc_ampl),slewFlavor);
    }
    return RecHit(digi.id(),ampl,time);    
  }
}

HBHERecHit HcalSimpleRecAlgo::reconstruct(const HBHEDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<HBHEDataFrame,HBHERecHit>(digi,coder,calibs,
							       firstSample_,samplesToAdd_,correctForTimeslew_,
							       pulseCorr_.get(),
							       HcalTimeSlew::Medium);
}

HORecHit HcalSimpleRecAlgo::reconstruct(const HODataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<HODataFrame,HORecHit>(digi,coder,calibs,
							   firstSample_,samplesToAdd_,correctForTimeslew_,
							   pulseCorr_.get(),
							   HcalTimeSlew::Slow);
}

ZDCRecHit HcalSimpleRecAlgo::reconstruct(const ZDCDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<ZDCDataFrame,ZDCRecHit>(digi,coder,calibs,
							     firstSample_,samplesToAdd_,false,
							     0,
							     HcalTimeSlew::Fast);
}

HcalCalibRecHit HcalSimpleRecAlgo::reconstruct(const HcalCalibDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<HcalCalibDataFrame,HcalCalibRecHit>(digi,coder,calibs,
									 firstSample_,samplesToAdd_,correctForTimeslew_,
									 pulseCorr_.get(),
									 HcalTimeSlew::Fast);
}

HFRecHit HcalSimpleRecAlgo::reconstruct(const HFDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  CaloSamples tool;
  coder.adc2fC(digi,tool);
  
  double ampl=0; int maxI = -1; double maxA = -1e10; float ta=0;
  for (int i=firstSample_; i<tool.size() && i<samplesToAdd_+firstSample_; i++) {
    int capid=digi[i].capid();
    ta = (tool[i]-calibs.pedestal(capid))*calibs.respcorrgain(capid);
    ampl+=ta;
    if(ta>maxA){
      maxA=ta;
      maxI=i;
    }
  }

  float time=-9999.0;
  ////Cannot calculate time value with max ADC sample at first or last position in window....
  if(maxI==0 || maxI==(tool.size()-1)) {
      LogDebug("HCAL Pulse") << "HcalSimpleRecAlgo::reconstruct :" 
					       << " Invalid max amplitude position, " 
					       << " max Amplitude: "<< maxI
					       << " first: "<<firstSample_
					       << " last: "<<(tool.size()-1)
					       << std::endl;
  } else {
    maxA=fabs(maxA);  
    int capid=digi[maxI-1].capid();
    float t0 = fabs((tool[maxI-1]-calibs.pedestal(capid))*calibs.respcorrgain(capid) );
    capid=digi[maxI+1].capid();
    float t2 = fabs((tool[maxI+1]-calibs.pedestal(capid))*calibs.respcorrgain(capid) );    
    float wpksamp = (t0 + maxA + t2);
    if (wpksamp!=0) wpksamp=(maxA + 2.0*t2) / wpksamp; 
    time = (maxI - digi.presamples())*25.0 + timeshift_ns_hf(wpksamp);
  }

  return HFRecHit(digi.id(),ampl,time); 
}

// timeshift implementation

static const float wpksamp0_hbheho = 0.680178;
static const float scale_hbheho    = 0.819786;
static const int   num_bins_hbheho = 50;

static const float actual_ns_hbheho[num_bins_hbheho] = {
 0.00000, // 0.000-0.020
 0.41750, // 0.020-0.040
 0.81500, // 0.040-0.060
 1.21000, // 0.060-0.080
 1.59500, // 0.080-0.100
 1.97250, // 0.100-0.120
 2.34750, // 0.120-0.140
 2.71250, // 0.140-0.160
 3.07500, // 0.160-0.180
 3.43500, // 0.180-0.200
 3.79000, // 0.200-0.220
 4.14250, // 0.220-0.240
 4.49250, // 0.240-0.260
 4.84250, // 0.260-0.280
 5.19000, // 0.280-0.300
 5.53750, // 0.300-0.320
 5.89000, // 0.320-0.340
 6.23750, // 0.340-0.360
 6.59250, // 0.360-0.380
 6.95250, // 0.380-0.400
 7.31000, // 0.400-0.420
 7.68000, // 0.420-0.440
 8.05500, // 0.440-0.460
 8.43000, // 0.460-0.480
 8.83250, // 0.480-0.500
 9.23250, // 0.500-0.520
 9.65500, // 0.520-0.540
10.09500, // 0.540-0.560
10.54750, // 0.560-0.580
11.04500, // 0.580-0.600
11.55750, // 0.600-0.620
12.13000, // 0.620-0.640
12.74500, // 0.640-0.660
13.41250, // 0.660-0.680
14.18500, // 0.680-0.700
15.02750, // 0.700-0.720
15.92250, // 0.720-0.740
16.82500, // 0.740-0.760
17.70000, // 0.760-0.780
18.52500, // 0.780-0.800
19.28750, // 0.800-0.820
19.99750, // 0.820-0.840
20.67250, // 0.840-0.860
21.31250, // 0.860-0.880
21.90750, // 0.880-0.900
22.48750, // 0.900-0.920
23.02750, // 0.920-0.940
23.55250, // 0.940-0.960
24.05000, // 0.960-0.980
24.53750, // 0.980-1.000
};

float timeshift_ns_hbheho(float wpksamp) {
  int index=(int)(0.5+num_bins_hbheho*(wpksamp-wpksamp0_hbheho)/scale_hbheho);
  
  if      (index <    0)             return actual_ns_hbheho[0];
  else if (index >= num_bins_hbheho) return actual_ns_hbheho[num_bins_hbheho-1];
  
  return actual_ns_hbheho[index];
}

static const float wpksamp0 = 0.500053;
static const float scale    = 0.999683;
static const int   num_bins = 100;

static const float actual_ns_hf_new[num_bins] = {
 0.00250, // 0.000-0.010
 0.08000, // 0.010-0.020
 0.16000, // 0.020-0.030
 0.23750, // 0.030-0.040
 0.31750, // 0.040-0.050
 0.39500, // 0.050-0.060
 0.47500, // 0.060-0.070
 0.55500, // 0.070-0.080
 0.63000, // 0.080-0.090
 0.70000, // 0.090-0.100
 0.77000, // 0.100-0.110
 0.84000, // 0.110-0.120
 0.91000, // 0.120-0.130
 0.98000, // 0.130-0.140
 1.05000, // 0.140-0.150
 1.12000, // 0.150-0.160
 1.19000, // 0.160-0.170
 1.26000, // 0.170-0.180
 1.33000, // 0.180-0.190
 1.40000, // 0.190-0.200
 1.47000, // 0.200-0.210
 1.54000, // 0.210-0.220
 1.61000, // 0.220-0.230
 1.68000, // 0.230-0.240
 1.75000, // 0.240-0.250
 1.82000, // 0.250-0.260
 1.89000, // 0.260-0.270
 1.96000, // 0.270-0.280
 2.03000, // 0.280-0.290
 2.10000, // 0.290-0.300
 2.17000, // 0.300-0.310
 2.24000, // 0.310-0.320
 2.31000, // 0.320-0.330
 2.38000, // 0.330-0.340
 2.45000, // 0.340-0.350
 2.52000, // 0.350-0.360
 2.59000, // 0.360-0.370
 2.68500, // 0.370-0.380
 2.79250, // 0.380-0.390
 2.90250, // 0.390-0.400
 3.01000, // 0.400-0.410
 3.11750, // 0.410-0.420
 3.22500, // 0.420-0.430
 3.33500, // 0.430-0.440
 3.44250, // 0.440-0.450
 3.55000, // 0.450-0.460
 3.73250, // 0.460-0.470
 4.02000, // 0.470-0.480
 4.30750, // 0.480-0.490
 4.59500, // 0.490-0.500
 6.97500, // 0.500-0.510
10.98750, // 0.510-0.520
13.03750, // 0.520-0.530
14.39250, // 0.530-0.540
15.39500, // 0.540-0.550
16.18250, // 0.550-0.560
16.85250, // 0.560-0.570
17.42750, // 0.570-0.580
17.91500, // 0.580-0.590
18.36250, // 0.590-0.600
18.76500, // 0.600-0.610
19.11250, // 0.610-0.620
19.46000, // 0.620-0.630
19.76500, // 0.630-0.640
20.03500, // 0.640-0.650
20.30250, // 0.650-0.660
20.57250, // 0.660-0.670
20.79250, // 0.670-0.680
21.00250, // 0.680-0.690
21.21000, // 0.690-0.700
21.42000, // 0.700-0.710
21.62750, // 0.710-0.720
21.79000, // 0.720-0.730
21.95250, // 0.730-0.740
22.11500, // 0.740-0.750
22.27750, // 0.750-0.760
22.44000, // 0.760-0.770
22.60500, // 0.770-0.780
22.73250, // 0.780-0.790
22.86000, // 0.790-0.800
22.98500, // 0.800-0.810
23.11250, // 0.810-0.820
23.23750, // 0.820-0.830
23.36500, // 0.830-0.840
23.49000, // 0.840-0.850
23.61750, // 0.850-0.860
23.71500, // 0.860-0.870
23.81250, // 0.870-0.880
23.91250, // 0.880-0.890
24.01000, // 0.890-0.900
24.10750, // 0.900-0.910
24.20750, // 0.910-0.920
24.30500, // 0.920-0.930
24.40500, // 0.930-0.940
24.50250, // 0.940-0.950
24.60000, // 0.950-0.960
24.68250, // 0.960-0.970
24.76250, // 0.970-0.980
24.84000, // 0.980-0.990
24.92000  // 0.990-1.000
};

float timeshift_ns_hf(float wpksamp) {
  float flx = (num_bins_hf*(wpksamp - wpksamp0_hf)/scale_hf);
  int index = (int)flx;
  float yval;
  
  if      (index <    0)        return actual_ns_hf[0];
  else if (index >= num_bins_hf-1) return actual_ns_hf[num_bins_hf-1];

  // else interpolate:
  float y1       = actual_ns_hf[index];
  float y2       = actual_ns_hf[index+1];

  // float delta_x  = 1/(float)num_bins_hf;
  // yval = y1 + (y2-y1)*(flx-(float)index)/delta_x;

  yval = y1 + (y2-y1)*(flx-(float)index);
  return yval;
}
