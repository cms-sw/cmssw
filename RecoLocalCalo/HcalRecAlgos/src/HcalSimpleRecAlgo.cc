#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSimpleRecAlgo.h"
#include <FWCore/Utilities/interface/Exception.h>

HcalSimpleRecAlgo::HcalSimpleRecAlgo(int firstSample, int samplesToAdd) : firstSample_(firstSample), samplesToAdd_(samplesToAdd) {
}

///Timeshift correction for HPDs based on the position of the peak ADC measurement.
///  Allows for an accurate determination of the relative phase of the pulse shape from
///  the HPD.  Calculated based on a weighted sum of the -1,0,+1 samples relative to the peak
///  as follows:  wpksamp = (0*sample[0] + 1*sample[1] + 2*sample[2]) / (sample[0] + sample[1] + sample[2])
///  where sample[1] is the maximum ADC sample value.
static float timeshift_hbheho(float wpksamp);

///Same as above, but for the HF PMTs.
static float timeshift_hf(float wpksamp);


namespace HcalSimpleRecAlgoImpl {
  template<class Digi, class RecHit>
  inline RecHit reco(const Digi& digi, const HcalCoder& coder, const HcalCalibrations& calibs, int ifirst, int n) {
    CaloSamples tool;
    coder.adc2fC(digi,tool);

    double ampl=0; int maxI = -1; double maxA = -1e10; float ta=0;
    for (int i=ifirst; i<tool.size() && i<n+ifirst; i++) {
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
    float time = (maxI - digi.presamples())*25.0 + timeshift_hbheho(wpksamp);
    
    return RecHit(digi.id(),ampl,time);    
  }
}

HBHERecHit HcalSimpleRecAlgo::reconstruct(const HBHEDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<HBHEDataFrame,HBHERecHit>(digi,coder,calibs,firstSample_,samplesToAdd_);
}
HORecHit HcalSimpleRecAlgo::reconstruct(const HODataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<HODataFrame,HORecHit>(digi,coder,calibs,firstSample_,samplesToAdd_);
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
  float time = (maxI - digi.presamples())*25.0 + timeshift_hf(wpksamp);
  
  return HFRecHit(digi.id(),ampl,time); 
}

// timeshift implementation

static const float wpksamp0_hbheho = 0.778139;
static const float scale_hbheho    = 0.555182;
static const int   num_bins = 50;

static const float actual_ns_hbheho[num_bins] = {
 0.00000, // 0.000-0.020
 0.33250, // 0.020-0.040
 0.65750, // 0.040-0.060
 0.97000, // 0.060-0.080
 1.29000, // 0.080-0.100
 1.60500, // 0.100-0.120
 1.91000, // 0.120-0.140
 2.21750, // 0.140-0.160
 2.53000, // 0.160-0.180
 2.82750, // 0.180-0.200
 3.13000, // 0.200-0.220
 3.43500, // 0.220-0.240
 3.73750, // 0.240-0.260
 4.03750, // 0.260-0.280
 4.34250, // 0.280-0.300
 4.65000, // 0.300-0.320
 4.95750, // 0.320-0.340
 5.27000, // 0.340-0.360
 5.58750, // 0.360-0.380
 5.91250, // 0.380-0.400
 6.24250, // 0.400-0.420
 6.58250, // 0.420-0.440
 6.94500, // 0.440-0.460
 7.31250, // 0.460-0.480
 7.70750, // 0.480-0.500
 8.13500, // 0.500-0.520
 8.57500, // 0.520-0.540
 9.10250, // 0.540-0.560
 9.67000, // 0.560-0.580
10.36250, // 0.580-0.600
11.24500, // 0.600-0.620
12.34500, // 0.620-0.640
13.60750, // 0.640-0.660
14.82250, // 0.660-0.680
15.90750, // 0.680-0.700
16.88250, // 0.700-0.720
17.76750, // 0.720-0.740
18.57500, // 0.740-0.760
19.30500, // 0.760-0.780
19.97750, // 0.780-0.800
20.60750, // 0.800-0.820
21.18250, // 0.820-0.840
21.72250, // 0.840-0.860
22.22500, // 0.860-0.880
22.69750, // 0.880-0.900
23.13500, // 0.900-0.920
23.56000, // 0.920-0.940
23.94250, // 0.940-0.960
24.31750, // 0.960-0.980
24.67000, // 0.980-1.000
};

static float timeshift_hbheho(float wpksamp) {
  int index = (int)(num_bins*(wpksamp - wpksamp0_hbheho)/scale_hbheho);
  
  if      (index <    0)      return actual_ns_hbheho[0];
  else if (index >= num_bins) return actual_ns_hbheho[num_bins-1];
  
  return actual_ns_hbheho[index];
}



static const float wpksamp0_hf = 0.667231;
static const float scale_hf    = 0.666045;
static const int   num_bins_hf = 100;

static const float actual_ns_hf[num_bins_hf] = {
 0.00250, // 0.000-0.010
 0.02750, // 0.010-0.020
 0.05500, // 0.020-0.030
 0.08250, // 0.030-0.040
 0.11000, // 0.040-0.050
 0.13750, // 0.050-0.060
 0.16750, // 0.060-0.070
 0.19500, // 0.070-0.080
 0.22250, // 0.080-0.090
 0.25250, // 0.090-0.100
 0.28000, // 0.100-0.110
 0.31000, // 0.110-0.120
 0.34000, // 0.120-0.130
 0.37000, // 0.130-0.140
 0.40000, // 0.140-0.150
 0.43000, // 0.150-0.160
 0.46000, // 0.160-0.170
 0.49250, // 0.170-0.180
 0.52500, // 0.180-0.190
 0.55500, // 0.190-0.200
 0.58750, // 0.200-0.210
 0.62250, // 0.210-0.220
 0.65500, // 0.220-0.230
 0.69000, // 0.230-0.240
 0.72250, // 0.240-0.250
 0.76000, // 0.250-0.260
 0.79500, // 0.260-0.270
 0.83000, // 0.270-0.280
 0.86750, // 0.280-0.290
 0.90750, // 0.290-0.300
 0.94500, // 0.300-0.310
 0.98500, // 0.310-0.320
 1.02500, // 0.320-0.330
 1.06750, // 0.330-0.340
 1.11000, // 0.340-0.350
 1.15500, // 0.350-0.360
 1.20250, // 0.360-0.370
 1.24750, // 0.370-0.380
 1.29750, // 0.380-0.390
 1.35000, // 0.390-0.400
 1.40250, // 0.400-0.410
 1.46000, // 0.410-0.420
 1.52000, // 0.420-0.430
 1.58250, // 0.430-0.440
 1.65000, // 0.440-0.450
 1.72500, // 0.450-0.460
 1.80750, // 0.460-0.470
 1.90250, // 0.470-0.480
 2.01250, // 0.480-0.490
 2.15500, // 0.490-0.500
19.36000, // 0.500-0.510
21.29000, // 0.510-0.520
21.79250, // 0.520-0.530
22.11000, // 0.530-0.540
22.34750, // 0.540-0.550
22.53750, // 0.550-0.560
22.69750, // 0.560-0.570
22.83750, // 0.570-0.580
22.96250, // 0.580-0.590
23.07250, // 0.590-0.600
23.17500, // 0.600-0.610
23.26750, // 0.610-0.620
23.35500, // 0.620-0.630
23.43500, // 0.630-0.640
23.51000, // 0.640-0.650
23.58250, // 0.650-0.660
23.65000, // 0.660-0.670
23.71500, // 0.670-0.680
23.77500, // 0.680-0.690
23.83500, // 0.690-0.700
23.89000, // 0.700-0.710
23.94250, // 0.710-0.720
23.99500, // 0.720-0.730
24.04500, // 0.730-0.740
24.09250, // 0.740-0.750
24.14000, // 0.750-0.760
24.18500, // 0.760-0.770
24.23000, // 0.770-0.780
24.27000, // 0.780-0.790
24.31250, // 0.790-0.800
24.35250, // 0.800-0.810
24.39250, // 0.810-0.820
24.43000, // 0.820-0.830
24.46750, // 0.830-0.840
24.50250, // 0.840-0.850
24.54000, // 0.850-0.860
24.57500, // 0.860-0.870
24.60750, // 0.870-0.880
24.64250, // 0.880-0.890
24.67500, // 0.890-0.900
24.70750, // 0.900-0.910
24.73750, // 0.910-0.920
24.77000, // 0.920-0.930
24.80000, // 0.930-0.940
24.83000, // 0.940-0.950
24.85750, // 0.950-0.960
24.88750, // 0.960-0.970
24.91500, // 0.970-0.980
24.94500, // 0.980-0.990
24.97250, // 0.990-1.000
};

static float timeshift_hf(float wpksamp) {
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
