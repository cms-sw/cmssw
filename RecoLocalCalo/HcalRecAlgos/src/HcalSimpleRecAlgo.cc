#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSimpleRecAlgo.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include <algorithm> // for "max"

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
      ta*=calibs.gain(capid); // fC --> GeV
      ampl+=ta;
      if(ta>maxA){
	maxA=ta;
	maxI=i;
      }
    }

    ////Cannot calculate time value with max ADC sample at first or last position in window....
    if(maxI==0 || maxI==(tool.size()-1)) {
      throw cms::Exception("InvalidRecoParam") << "HcalSimpleRecAlgo::reconstruct :" 
					       << " Invalid max amplitude position, " 
					       << " max Amplitude: "<< maxI
					       << " first: "<<ifirst
					       << " last: "<<(tool.size()-1)
					       << std::endl;
    }
    
    
    maxA=fabs(maxA);
    int capid=digi[maxI-1].capid();
    float t0 = fabs((tool[maxI-1]-calibs.pedestal(capid))*calibs.gain(capid));
    capid=digi[maxI+1].capid();
    float t2 = fabs((tool[maxI+1]-calibs.pedestal(capid))*calibs.gain(capid));    
    float wpksamp = (maxA + 2.0*t2) / (t0 + maxA + t2);
    float time = (maxI - digi.presamples())*25.0 + timeshift_ns_hbheho(wpksamp);

    if (corr!=0) {
      // Apply phase-based amplitude correction:
      ampl *= corr->getCorrection(fc_ampl);
//      std::cout << fc_ampl << " --> " << corr->getCorrection(fc_ampl) << std::endl;
    }

    
    if (slewCorrect) time-=HcalTimeSlew::delay(std::max(0.0,fc_ampl),slewFlavor);

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
    ta = (tool[i]-calibs.pedestal(capid))*calibs.gain(capid);
    ampl+=ta;
    if(ta>maxA){
      maxA=ta;
      maxI=i;
    }
  }

  ////Cannot calculate time value with max ADC sample at first or last position in window....
  if(maxI==0 || maxI==(tool.size()-1)) {
    throw cms::Exception("InvalidRecoParam") << "HcalSimpleRecAlgo::reconstruct :" 
					 << " Invalid max amplitude position, " 
					 << " max Amplitude: "<< maxI
					 << " first: "<<firstSample_
					 << " last: "<<(tool.size()-1)
					 << std::endl;
  }

  maxA=fabs(maxA);  
  int capid=digi[maxI-1].capid();
  float t0 = fabs((tool[maxI-1]-calibs.pedestal(capid))*calibs.gain(capid));
  capid=digi[maxI+1].capid();
  float t2 = fabs((tool[maxI+1]-calibs.pedestal(capid))*calibs.gain(capid));    
  float wpksamp = (maxA + 2.0*t2) / (t0 + maxA + t2);
  float time = (maxI - digi.presamples())*25.0 + timeshift_ns_hf(wpksamp);
  
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


static const float wpksamp0_hf = 0.500635;
static const float scale_hf    = 0.999301;
static const int   num_bins_hf = 100;

static const float actual_ns_hf[num_bins_hf] = {
 0.00000, // 0.000-0.010
 0.03750, // 0.010-0.020
 0.07250, // 0.020-0.030
 0.10750, // 0.030-0.040
 0.14500, // 0.040-0.050
 0.18000, // 0.050-0.060
 0.21500, // 0.060-0.070
 0.25000, // 0.070-0.080
 0.28500, // 0.080-0.090
 0.32000, // 0.090-0.100
 0.35500, // 0.100-0.110
 0.39000, // 0.110-0.120
 0.42500, // 0.120-0.130
 0.46000, // 0.130-0.140
 0.49500, // 0.140-0.150
 0.53000, // 0.150-0.160
 0.56500, // 0.160-0.170
 0.60000, // 0.170-0.180
 0.63500, // 0.180-0.190
 0.67000, // 0.190-0.200
 0.70750, // 0.200-0.210
 0.74250, // 0.210-0.220
 0.78000, // 0.220-0.230
 0.81500, // 0.230-0.240
 0.85250, // 0.240-0.250
 0.89000, // 0.250-0.260
 0.92750, // 0.260-0.270
 0.96500, // 0.270-0.280
 1.00250, // 0.280-0.290
 1.04250, // 0.290-0.300
 1.08250, // 0.300-0.310
 1.12250, // 0.310-0.320
 1.16250, // 0.320-0.330
 1.20500, // 0.330-0.340
 1.24500, // 0.340-0.350
 1.29000, // 0.350-0.360
 1.33250, // 0.360-0.370
 1.38000, // 0.370-0.380
 1.42500, // 0.380-0.390
 1.47500, // 0.390-0.400
 1.52500, // 0.400-0.410
 1.57750, // 0.410-0.420
 1.63250, // 0.420-0.430
 1.69000, // 0.430-0.440
 1.75250, // 0.440-0.450
 1.82000, // 0.450-0.460
 1.89250, // 0.460-0.470
 1.97500, // 0.470-0.480
 2.07250, // 0.480-0.490
 2.20000, // 0.490-0.500
19.13000, // 0.500-0.510
21.08750, // 0.510-0.520
21.57750, // 0.520-0.530
21.89000, // 0.530-0.540
22.12250, // 0.540-0.550
22.31000, // 0.550-0.560
22.47000, // 0.560-0.570
22.61000, // 0.570-0.580
22.73250, // 0.580-0.590
22.84500, // 0.590-0.600
22.94750, // 0.600-0.610
23.04250, // 0.610-0.620
23.13250, // 0.620-0.630
23.21500, // 0.630-0.640
23.29250, // 0.640-0.650
23.36750, // 0.650-0.660
23.43750, // 0.660-0.670
23.50500, // 0.670-0.680
23.57000, // 0.680-0.690
23.63250, // 0.690-0.700
23.69250, // 0.700-0.710
23.75000, // 0.710-0.720
23.80500, // 0.720-0.730
23.86000, // 0.730-0.740
23.91250, // 0.740-0.750
23.96500, // 0.750-0.760
24.01500, // 0.760-0.770
24.06500, // 0.770-0.780
24.11250, // 0.780-0.790
24.16000, // 0.790-0.800
24.20500, // 0.800-0.810
24.25000, // 0.810-0.820
24.29500, // 0.820-0.830
24.33750, // 0.830-0.840
24.38000, // 0.840-0.850
24.42250, // 0.850-0.860
24.46500, // 0.860-0.870
24.50500, // 0.870-0.880
24.54500, // 0.880-0.890
24.58500, // 0.890-0.900
24.62500, // 0.900-0.910
24.66500, // 0.910-0.920
24.70250, // 0.920-0.930
24.74000, // 0.930-0.940
24.77750, // 0.940-0.950
24.81500, // 0.950-0.960
24.85250, // 0.960-0.970
24.89000, // 0.970-0.980
24.92750, // 0.980-0.990
24.96250, // 0.990-1.000
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

  yval = y1 + (y2-y1)*(flx-(float)index)*(float)num_bins_hf;
  return yval;
}
