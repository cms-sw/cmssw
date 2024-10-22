#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Random/RandFlat.h"

// #include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include "TMath.h"

HcalPulseShapes::HcalPulseShapes() : theDbService(nullptr), theShapes() {
  /*

Reco  MC
-------------------------------------------------------------------------------------------
000                                              not used (reserved)
101   101   hpdShape_                            HPD (original version)
102   102   =101                                 HPD BV 30 volts in HBP iphi54
103   123   hpdShape_v2,hpdShapeMC_v2            HPD (2011. oct version)
104   124   hpdBV30Shape_v2,hpdBV30ShapeMC_v2    HPD bv30 in HBP iph54
105   125   hpdShape_v2,hpdShapeMC_v2            HPD (2011.11.12 version)
201   201   siPMShapeHO_                         SiPMs Zecotec shape   (HO)
202   202   =201,                                SiPMs Hamamatsu shape (HO)
205   203   siPMShapeData2017_,siPMShapeMC2017_  SiPMs from Data, Hamamatsu shape (HE 2017)
207   206   siPMShapeData2018_,siPMShapeMC2018_  SiPMs from Data, Hamamatsu shape (HE 2018)
207   208   siPMShapeData2018_,siPMShapeMCRecoRun3_  SiPMs from Data, 2021 MC phase scan
301   301   hfShape_                             regular HF PMT shape
401   401                                        regular ZDC shape
-------------------------------------------------------------------------------------------

*/

  float ts1, ts2, ts3, thpd, tpre, wd1, wd2, wd3;

  //  HPD Shape  Version 1 (used before CMSSW5, until Oct 2011)
  ts1 = 8.;
  ts2 = 10.;
  ts3 = 29.3;
  thpd = 4.0;
  tpre = 9.0;
  wd1 = 2.0;
  wd2 = 0.7;
  wd3 = 1.0;
  computeHPDShape(ts1, ts2, ts3, thpd, tpre, wd1, wd2, wd3, hpdShape_);
  theShapes[101] = &hpdShape_;
  theShapes[102] = theShapes[101];

  //  HPD Shape  Version 2 for CMSSW 5. Nov 2011  (RECO and MC separately)
  ts1 = 8.;
  ts2 = 10.;
  ts3 = 25.0;
  thpd = 4.0;
  tpre = 9.0;
  wd1 = 2.0;
  wd2 = 0.7;
  wd3 = 1.0;
  computeHPDShape(ts1, ts2, ts3, thpd, tpre, wd1, wd2, wd3, hpdShape_v2);
  theShapes[103] = &hpdShape_v2;

  ts1 = 8.;
  ts2 = 10.;
  ts3 = 29.3;
  thpd = 4.0;
  tpre = 7.0;
  wd1 = 2.0;
  wd2 = 0.7;
  wd3 = 1.0;
  computeHPDShape(ts1, ts2, ts3, thpd, tpre, wd1, wd2, wd3, hpdShapeMC_v2);
  theShapes[123] = &hpdShapeMC_v2;

  //  HPD Shape  Version 3 for CMSSW 5. Nov 2011  (RECO and MC separately)
  ts1 = 8.;
  ts2 = 19.;
  ts3 = 29.3;
  thpd = 4.0;
  tpre = 9.0;
  wd1 = 2.0;
  wd2 = 0.7;
  wd3 = 0.32;
  computeHPDShape(ts1, ts2, ts3, thpd, tpre, wd1, wd2, wd3, hpdShape_v3);
  theShapes[105] = &hpdShape_v3;

  ts1 = 8.;
  ts2 = 10.;
  ts3 = 22.3;
  thpd = 4.0;
  tpre = 7.0;
  wd1 = 2.0;
  wd2 = 0.7;
  wd3 = 1.0;
  computeHPDShape(ts1, ts2, ts3, thpd, tpre, wd1, wd2, wd3, hpdShapeMC_v3);
  theShapes[125] = &hpdShapeMC_v3;

  // HPD with Bias Voltage 30 volts, wider pulse.  (HBPlus iphi54)

  ts1 = 8.;
  ts2 = 12.;
  ts3 = 31.7;
  thpd = 9.0;
  tpre = 9.0;
  wd1 = 2.0;
  wd2 = 0.7;
  wd3 = 1.0;
  computeHPDShape(ts1, ts2, ts3, thpd, tpre, wd1, wd2, wd3, hpdBV30Shape_v2);
  theShapes[104] = &hpdBV30Shape_v2;

  ts1 = 8.;
  ts2 = 12.;
  ts3 = 31.7;
  thpd = 9.0;
  tpre = 9.0;
  wd1 = 2.0;
  wd2 = 0.7;
  wd3 = 1.0;
  computeHPDShape(ts1, ts2, ts3, thpd, tpre, wd1, wd2, wd3, hpdBV30ShapeMC_v2);
  theShapes[124] = &hpdBV30ShapeMC_v2;

  // HF and SiPM

  computeHFShape();
  computeSiPMShapeHO();
  computeSiPMShapeData2017();
  computeSiPMShapeData2018();
  computeSiPMShapeMCRecoRun3();

  theShapes[201] = &siPMShapeHO_;
  theShapes[202] = theShapes[201];
  theShapes[203] = &(computeSiPMShapeHE203());
  theShapes[205] = &siPMShapeData2017_;
  theShapes[206] = &(computeSiPMShapeHE206());
  theShapes[207] = &siPMShapeData2018_;
  theShapes[208] = &siPMShapeMCRecoRun3_;
  theShapes[301] = &hfShape_;
  //theShapes[401] = new CaloCachedShapeIntegrator(&theZDCShape);
}

HcalPulseShapes::HcalPulseShapes(edm::ConsumesCollector iC) : HcalPulseShapes() {
  theDbServiceToken = iC.esConsumes<edm::Transition::BeginRun>();
}

HcalPulseShapes::~HcalPulseShapes() {}

void HcalPulseShapes::beginRun(edm::EventSetup const& es) { theDbService = &es.getData(theDbServiceToken); }

void HcalPulseShapes::beginRun(const HcalDbService* conditions) { theDbService = conditions; }

//void HcalPulseShapes::computeHPDShape()
void HcalPulseShapes::computeHPDShape(
    float ts1, float ts2, float ts3, float thpd, float tpre, float wd1, float wd2, float wd3, Shape& tmphpdShape_) {
  // pulse shape time constants in ns
  /*
  const float ts1  = 8.;          // scintillation time constants : 1,2,3
  const float ts2  = 10.;           
  const float ts3  = 29.3;         
  const float thpd = 4.;          // HPD current collection drift time
  const float tpre = 9.;          // preamp time constant (refit on TB04 data)
  
  const float wd1 = 2.;           // relative weights of decay exponents 
  const float wd2 = 0.7;
  const float wd3 = 1.;
*/
  // pulse shape components over a range of time 0 ns to 255 ns in 1 ns steps
  unsigned int nbin = 256;
  tmphpdShape_.setNBin(nbin);
  std::vector<float> ntmp(nbin, 0.0);  // zeroing output pulse shape
  std::vector<float> nth(nbin, 0.0);   // zeroing HPD drift shape
  std::vector<float> ntp(nbin, 0.0);   // zeroing Binkley preamp shape
  std::vector<float> ntd(nbin, 0.0);   // zeroing Scintillator decay shape

  unsigned int i, j, k;
  float norm;

  // HPD starts at I and rises to 2I in thpd of time
  norm = 0.0;
  for (j = 0; j < thpd && j < nbin; j++) {
    nth[j] = 1.0 + ((float)j) / thpd;
    norm += nth[j];
  }
  // normalize integrated current to 1.0
  for (j = 0; j < thpd && j < nbin; j++) {
    nth[j] /= norm;
  }

  // Binkley shape over 6 time constants
  norm = 0.0;
  for (j = 0; j < 6 * tpre && j < nbin; j++) {
    ntp[j] = ((float)j) * exp(-((float)(j * j)) / (tpre * tpre));
    norm += ntp[j];
  }
  // normalize pulse area to 1.0
  for (j = 0; j < 6 * tpre && j < nbin; j++) {
    ntp[j] /= norm;
  }

  // ignore stochastic variation of photoelectron emission
  // <...>

  // effective tile plus wave-length shifter decay time over 4 time constants
  unsigned int tmax = 6 * (int)ts3;

  norm = 0.0;
  for (j = 0; j < tmax && j < nbin; j++) {
    ntd[j] = wd1 * exp(-((float)j) / ts1) + wd2 * exp(-((float)j) / ts2) + wd3 * exp(-((float)j) / ts3);
    norm += ntd[j];
  }
  // normalize pulse area to 1.0
  for (j = 0; j < tmax && j < nbin; j++) {
    ntd[j] /= norm;
  }

  unsigned int t1, t2, t3, t4;
  for (i = 0; i < tmax && i < nbin; i++) {
    t1 = i;
    //    t2 = t1 + top*rand;
    // ignoring jitter from optical path length
    t2 = t1;
    for (j = 0; j < thpd && j < nbin; j++) {
      t3 = t2 + j;
      for (k = 0; k < 4 * tpre && k < nbin; k++) {  // here "4" is set deliberately,
        t4 = t3 + k;                                // as in test fortran toy MC ...
        if (t4 < nbin) {
          unsigned int ntb = t4;
          ntmp[ntb] += ntd[i] * nth[j] * ntp[k];
        }
      }
    }
  }

  // normalize for 1 GeV pulse height
  norm = 0.;
  for (i = 0; i < nbin; i++) {
    norm += ntmp[i];
  }

  for (i = 0; i < nbin; i++) {
    ntmp[i] /= norm;
  }

  for (i = 0; i < nbin; i++) {
    tmphpdShape_.setShapeBin(i, ntmp[i]);
  }
}

void HcalPulseShapes::computeHFShape() {
  // first create pulse shape over a range of time 0 ns to 255 ns in 1 ns steps
  unsigned int nbin = 256;
  hfShape_.setNBin(nbin);
  std::vector<float> ntmp(nbin, 0.0);  //

  const float k0 = 0.7956;  // shape parameters
  const float p2 = 1.355;
  const float p4 = 2.327;
  const float p1 = 4.3;  // position parameter

  float norm = 0.0;

  for (unsigned int j = 0; j < 25 && j < nbin; ++j) {
    float r0 = j - p1;
    float sigma0 = (r0 < 0) ? p2 : p2 * p4;
    r0 /= sigma0;
    if (r0 < k0)
      ntmp[j] = exp(-0.5 * r0 * r0);
    else
      ntmp[j] = exp(0.5 * k0 * k0 - k0 * r0);
    norm += ntmp[j];
  }
  // normalize pulse area to 1.0
  for (unsigned int j = 0; j < 25 && j < nbin; ++j) {
    ntmp[j] /= norm;
    hfShape_.setShapeBin(j, ntmp[j]);
  }
}
void HcalPulseShapes::computeSiPMShapeMCRecoRun3() {
  //modified shape 206
  //7.2 ns shift in 206
  unsigned int nbin = 250;
  std::array<float, 250> nt{
      {0,           0,           0,           0,           0,           0,           0,           0.000117468,
       0.0031549,   0.0117368,   0.0219974,   0.0305776,   0.0365429,   0.0400524,   0.0415915,   0.0416765,
       0.0408111,   0.0394627,   0.0379353,   0.0363688,   0.0348152,   0.0332891,   0.0317923,   0.0303237,
       0.028883,    0.0274714,   0.0260914,   0.0247462,   0.0234392,   0.0221738,   0.0209531,   0.0197793,
       0.0186544,   0.0175796,   0.0165556,   0.0155823,   0.0146596,   0.0137866,   0.0129623,   0.0121853,
       0.0114539,   0.0107665,   0.0101213,   0.0095162,   0.00894934,  0.00841873,  0.0079224,   0.00745841,
       0.00702487,  0.00661995,  0.00624189,  0.00588898,  0.00555961,  0.00525223,  0.00496539,  0.0046977,
       0.00444786,  0.00421464,  0.00399689,  0.00379353,  0.00360355,  0.00342602,  0.00326004,  0.0031048,
       0.00295954,  0.00282355,  0.00269616,  0.00257676,  0.00246479,  0.00235972,  0.00226106,  0.00216834,
       0.00208117,  0.00199914,  0.00192189,  0.0018491,   0.00178044,  0.00171565,  0.00165445,  0.00159659,
       0.00154186,  0.00149003,  0.00144092,  0.00139435,  0.00135015,  0.00130816,  0.00126825,  0.00123027,
       0.00119412,  0.00115966,  0.0011268,   0.00109544,  0.00106548,  0.00103685,  0.00100946,  0.000983242,
       0.000958125, 0.000934047, 0.000910949, 0.000888775, 0.000867475, 0.000847,    0.000827306, 0.000808352,
       0.000790097, 0.000772506, 0.000755545, 0.000739182, 0.000723387, 0.000708132, 0.00069339,  0.000679138,
       0.000665352, 0.00065201,  0.000639091, 0.000626577, 0.00061445,  0.000602692, 0.000591287, 0.00058022,
       0.000569477, 0.000559044, 0.000548908, 0.000539058, 0.000529481, 0.000520167, 0.000511106, 0.000502288,
       0.000493704, 0.000485344, 0.000477201, 0.000469266, 0.000459912, 0.000448544, 0.000437961, 0.000428079,
       0.000418825, 0.000410133, 0.000401945, 0.00039421,  0.000386883, 0.000379924, 0.000373298, 0.000366973,
       0.000360922, 0.00035512,  0.000349545, 0.000344179, 0.000339003, 0.000334002, 0.000329163, 0.000324475,
       0.000319925, 0.000315504, 0.000311204, 0.000307017, 0.000302935, 0.000298954, 0.000295066, 0.000291267,
       0.000287553, 0.000283919, 0.000280361, 0.000276877, 0.000273462, 0.000270114, 0.000266831, 0.000263609,
       0.000260447, 0.000257343, 0.000254295, 0.0002513,   0.000248358, 0.000245467, 0.000242625, 0.000239831,
       0.000237083, 0.000234381, 0.000231723, 0.000229109, 0.000226536, 0.000224005, 0.000221514, 0.000219062,
       0.000216648, 0.000214272, 0.000211933, 0.00020963,  0.000207362, 0.000205129, 0.000202929, 0.000200763,
       0.000198629, 0.000196526, 0.000194455, 0.000192415, 0.000190405, 0.000188424, 0.000186472, 0.000184548,
       0.000182653, 0.000180784, 0.000178943, 0.000177127, 0.000175338, 0.000173574, 0.000171835, 0.00017012,
       0.000168429, 0.000166762, 0.000165119, 0.000163498, 0.000161899, 0.000160322, 0.000158767, 0.000157233,
       0.000155721, 0.000154228, 0.000152756, 0.000151304, 0.000149871, 0.000148457, 0.000147062, 0.000145686,
       0.000144327, 0.000142987, 0.000141664, 0.000140359, 0.000139071, 0.000137799, 0.000136544, 0.000135305,
       0.000134082, 0.000132874, 0.000131682, 0.000130505, 0.000129344, 0.000128196, 0.000127064, 0.000125945,
       0.00012484,  0.00012375,  0.000122672, 0.000121608, 0.000120558, 0.00011952,  0.000118495, 0.000117482,
       0.000116482, 0.000115493}};

  siPMShapeMCRecoRun3_.setNBin(nbin);

  double norm = 0.;
  for (unsigned int j = 0; j < nbin; ++j) {
    norm += (nt[j] > 0) ? nt[j] : 0.;
  }

  for (unsigned int j = 0; j < nbin; ++j) {
    nt[j] /= norm;
    siPMShapeMCRecoRun3_.setShapeBin(j, nt[j]);
  }
}

void HcalPulseShapes::computeSiPMShapeData2018() {
  //Combination of all phase scan data (May,Jul,Oct2017)
  //runs:  294736-294740, 294929-294950, 298594-298598 and 305744-305758

  unsigned int nbin = 250;

  std::array<float, 250> nt{
      {5.22174e-12, 7.04852e-10, 3.49584e-08, 7.78029e-07, 9.11847e-06, 6.39666e-05, 0.000297587, 0.000996661,
       0.00256618,  0.00535396,  0.00944073,  0.0145521,   0.020145,    0.0255936,   0.0303632,   0.0341078,
       0.0366849,   0.0381183,   0.0385392,   0.0381327,   0.0370956,   0.0356113,   0.0338366,   0.0318978,
       0.029891,    0.0278866,   0.0259336,   0.0240643,   0.0222981,   0.0206453,   0.0191097,   0.0176902,
       0.0163832,   0.0151829,   0.0140826,   0.0130752,   0.0121533,   0.01131,     0.0105382,   0.00983178,
       0.00918467,  0.00859143,  0.00804709,  0.0075471,   0.00708733,  0.00666406,  0.00627393,  0.00591389,
       0.00558122,  0.00527344,  0.00498834,  0.00472392,  0.00447837,  0.00425007,  0.00403754,  0.00383947,
       0.00365465,  0.00348199,  0.00332052,  0.00316934,  0.00302764,  0.0028947,   0.00276983,  0.00265242,
       0.00254193,  0.00243785,  0.00233971,  0.00224709,  0.0021596,   0.00207687,  0.0019986,   0.00192447,
       0.00185421,  0.00178756,  0.0017243,   0.00166419,  0.00160705,  0.00155268,  0.00150093,  0.00145162,
       0.00140461,  0.00135976,  0.00131696,  0.00127607,  0.00123699,  0.00119962,  0.00116386,  0.00112963,
       0.00109683,  0.0010654,   0.00103526,  0.00100634,  0.000978578, 0.000951917, 0.000926299, 0.000901672,
       0.000877987, 0.000855198, 0.00083326,  0.000812133, 0.000791778, 0.000772159, 0.000753242, 0.000734994,
       0.000717384, 0.000700385, 0.000683967, 0.000668107, 0.000652779, 0.00063796,  0.000623629, 0.000609764,
       0.000596346, 0.000583356, 0.000570777, 0.000558592, 0.000546785, 0.00053534,  0.000524243, 0.000513481,
       0.00050304,  0.000492907, 0.000483072, 0.000473523, 0.000464248, 0.000455238, 0.000446483, 0.000437974,
       0.0004297,   0.000421655, 0.00041383,  0.000406216, 0.000398807, 0.000391595, 0.000384574, 0.000377736,
       0.000371076, 0.000364588, 0.000358266, 0.000352104, 0.000346097, 0.00034024,  0.000334528, 0.000328956,
       0.00032352,  0.000318216, 0.000313039, 0.000307986, 0.000303052, 0.000298234, 0.000293528, 0.000288931,
       0.000284439, 0.00028005,  0.000275761, 0.000271567, 0.000267468, 0.000263459, 0.000259538, 0.000255703,
       0.000251951, 0.00024828,  0.000244688, 0.000241172, 0.00023773,  0.000234361, 0.000231061, 0.00022783,
       0.000224666, 0.000221566, 0.000218528, 0.000215553, 0.000212636, 0.000209778, 0.000206977, 0.00020423,
       0.000201537, 0.000198896, 0.000196307, 0.000193767, 0.000191275, 0.000188831, 0.000186432, 0.000184079,
       0.000181769, 0.000179502, 0.000177277, 0.000175092, 0.000172947, 0.000170841, 0.000168772, 0.000166741,
       0.000164745, 0.000162785, 0.000160859, 0.000158967, 0.000157108, 0.00015528,  0.000153484, 0.000151719,
       0.000149984, 0.000148278, 0.000146601, 0.000144951, 0.000143329, 0.000141734, 0.000140165, 0.000138622,
       0.000137104, 0.00013561,  0.000134141, 0.000132695, 0.000131272, 0.000129871, 0.000128493, 0.000127136,
       0.000125801, 0.000124486, 0.000123191, 0.000121917, 0.000120662, 0.000119426, 0.000118209, 0.00011701,
       0.000115829, 0.000114665, 0.000113519, 0.00011239,  0.000111278, 0.000110182, 0.000109102, 0.000108037,
       0.000106988, 0.000105954, 0.000104935, 0.00010393,  0.000102939, 0.000101963, 0.000101,    0.000100051,
       9.91146e-05, 9.81915e-05, 9.7281e-05,  9.63831e-05, 9.54975e-05, 9.46239e-05, 9.37621e-05, 9.2912e-05,
       9.20733e-05, 9.12458e-05}};

  siPMShapeData2018_.setNBin(nbin);

  double norm = 0.;
  for (unsigned int j = 0; j < nbin; ++j) {
    norm += (nt[j] > 0) ? nt[j] : 0.;
  }

  for (unsigned int j = 0; j < nbin; ++j) {
    nt[j] /= norm;
    siPMShapeData2018_.setShapeBin(j, nt[j]);
  }
}

void HcalPulseShapes::computeSiPMShapeData2017() {
  //From Jay Lawhorn: derived from data Edward Laird phase scan may2017
  //https://indico.cern.ch/event/641978/contributions/2604491/attachments/1468666/2271582/17-05-31-hcal-hep17-pulse-shape.pdf
  //Run numbers are 294736-294740 and 294929-294950

  unsigned int nbin = 250;

  std::array<float, 250> nt{
      {3.97958e-29, 1.11634e-22, 9.96106e-18, 6.25334e-14, 5.08863e-11, 8.59141e-09, 4.32285e-07, 8.56617e-06,
       8.28549e-05, 0.000461447, 0.00168052,  0.00441395,  0.00901637,  0.0151806,   0.0220314,   0.028528,
       0.0338471,   0.0375578,   0.0395985,   0.0401567,   0.0395398,   0.0380776,   0.0360669,   0.0337474,
       0.0312984,   0.0288457,   0.0264721,   0.0242276,   0.0221393,   0.0202181,   0.0184647,   0.0168731,
       0.0154335,   0.0141346,   0.0129639,   0.0119094,   0.0109594,   0.0101031,   0.0093305,   0.00863267,
       0.0080015,   0.00742977,  0.00691107,  0.00643969,  0.00601059,  0.00561931,  0.00526188,  0.00493483,
       0.00463505,  0.00435981,  0.00410667,  0.00387348,  0.00365832,  0.00345949,  0.00327547,  0.0031049,
       0.00294656,  0.00279938,  0.00266237,  0.00253467,  0.00241548,  0.0023041,   0.00219989,  0.00210227,
       0.00201072,  0.00192476,  0.00184397,  0.00176795,  0.00169634,  0.00162884,  0.00156512,  0.00150494,
       0.00144803,  0.00139418,  0.00134317,  0.00129481,  0.00124894,  0.00120537,  0.00116398,  0.00112461,
       0.00108715,  0.00105147,  0.00101747,  0.000985042, 0.000954096, 0.000924545, 0.000896308, 0.000869311,
       0.000843482, 0.000818758, 0.000795077, 0.000772383, 0.000750623, 0.000729747, 0.00070971,  0.000690466,
       0.000671977, 0.000654204, 0.00063711,  0.000620663, 0.000604831, 0.000589584, 0.000574894, 0.000560735,
       0.000547081, 0.00053391,  0.0005212,   0.000508929, 0.000497078, 0.000485628, 0.000474561, 0.000463862,
       0.000453514, 0.000443501, 0.000433811, 0.000424429, 0.000415343, 0.00040654,  0.00039801,  0.000389741,
       0.000381722, 0.000373944, 0.000366398, 0.000359074, 0.000351964, 0.00034506,  0.000338353, 0.000331838,
       0.000325505, 0.00031935,  0.000313365, 0.000307544, 0.000301881, 0.000296371, 0.000291009, 0.000285788,
       0.000280705, 0.000275755, 0.000270932, 0.000266233, 0.000261653, 0.00025719,  0.000252837, 0.000248593,
       0.000244454, 0.000240416, 0.000236475, 0.00023263,  0.000228876, 0.000225212, 0.000221633, 0.000218138,
       0.000214724, 0.000211389, 0.00020813,  0.000204945, 0.000201831, 0.000198787, 0.000195811, 0.0001929,
       0.000190053, 0.000187268, 0.000184543, 0.000181876, 0.000179266, 0.000176711, 0.00017421,  0.000171761,
       0.000169363, 0.000167014, 0.000164713, 0.000162459, 0.00016025,  0.000158086, 0.000155964, 0.000153885,
       0.000151847, 0.000149848, 0.000147888, 0.000145966, 0.000144081, 0.000142232, 0.000140418, 0.000138638,
       0.000136891, 0.000135177, 0.000133494, 0.000131843, 0.000130221, 0.00012863,  0.000127066, 0.000125531,
       0.000124023, 0.000122543, 0.000121088, 0.000119658, 0.000118254, 0.000116874, 0.000115518, 0.000114185,
       0.000112875, 0.000111587, 0.000110321, 0.000109076, 0.000107851, 0.000106648, 0.000105464, 0.000104299,
       0.000103154, 0.000102027, 0.000100918, 9.98271e-05, 9.87537e-05, 9.76974e-05, 9.66578e-05, 9.56346e-05,
       9.46274e-05, 9.3636e-05,  9.26599e-05, 9.16989e-05, 9.07526e-05, 8.98208e-05, 8.89032e-05, 8.79995e-05,
       8.71093e-05, 8.62325e-05, 8.53688e-05, 8.45179e-05, 8.36796e-05, 8.28536e-05, 8.20397e-05, 8.12376e-05,
       8.04471e-05, 7.96681e-05, 7.89002e-05, 7.81433e-05, 7.73972e-05, 7.66616e-05, 7.59364e-05, 7.52213e-05,
       7.45163e-05, 7.3821e-05,  7.31354e-05, 7.24592e-05, 7.17923e-05, 7.11345e-05, 7.04856e-05, 6.98455e-05,
       6.9214e-05,  6.8591e-05}};

  siPMShapeData2017_.setNBin(nbin);

  double norm = 0.;
  for (unsigned int j = 0; j < nbin; ++j) {
    norm += (nt[j] > 0) ? nt[j] : 0.;
  }

  for (unsigned int j = 0; j < nbin; ++j) {
    nt[j] /= norm;
    siPMShapeData2017_.setShapeBin(j, nt[j]);
  }
}

void HcalPulseShapes::computeSiPMShapeHO() {
  unsigned int nbin = 128;

  //From Jake Anderson: toy MC convolution of SiPM pulse + WLS fiber shape + SiPM nonlinear response
  std::array<float, 128> nt{
      {2.782980485851731e-6,  4.518134885954626e-5,  2.7689305197392056e-4, 9.18328418900969e-4,
       .002110072599166349,   .003867856860331454,   .006120046224897771,   .008754774090536956,
       0.0116469503358586,    .01467007449455966,    .01770489955229477,    .02064621450689512,
       .02340678093764222,    .02591874610854916,    .02813325527435303,    0.0300189241965647,
       .03155968107671164,    .03275234052577155,    .03360415306318798,    .03413048377960748,
       .03435270899678218,    .03429637464659661,    .03398962975487166,    .03346192884394954,
       .03274298516247742,    .03186195009136525,    .03084679116113031,    0.0297238406141036,
       .02851748748929785,    .02724998816332392,    .02594137274487424,    .02460942736731527,
       .02326973510736116,    .02193576080366117,    0.0206189674254987,    .01932895378564653,
       0.0180736052958666,    .01685925112650875,    0.0156908225633535,    .01457200857138456,
       .01350540559602467,    .01249265947824805,    .01153459805300423,    .01063135355597282,
       .009782474412011936,   .008987026319784546,   0.00824368281357106,   .007550805679909604,
       .006906515742762193,   .006308754629755056,   .005755338185695127,   .005244002229973356,
       .004772441359900532,   .004338341490928299,   .003939406800854143,   0.00357338171220501,
       0.0032380685079891,    .002931341133259233,   .002651155690306086,   .002395558090237333,
       .002162689279320922,   .001950788415487319,   .001758194329648101,   .001583345567913682,
       .001424779275191974,   .001281129147671334,   0.00115112265163774,   .001033577678808199,
       9.273987838127585e-4,  8.315731274976846e-4,  7.451662302008696e-4,  6.673176219006913e-4,
       5.972364609644049e-4,  5.341971801529036e-4,  4.775352065178378e-4,  4.266427928961177e-4,
       3.8096498904225923e-4, 3.3999577417327287e-4, 3.032743659102713e-4,  2.703817158798329e-4,
       2.4093719775272793e-4, 2.145954900503894e-4,  1.9104365317752797e-4, 1.6999839784346724e-4,
       1.5120354022478893e-4, 1.3442763782650755e-4, 1.1946179895521507e-4, 1.0611765796993575e-4,
       9.422550797617687e-5,  8.363258233342666e-5,  7.420147621931836e-5,  6.580869950304933e-5,
       5.834335229919868e-5,  5.17059147771959e-5,   4.5807143072062634e-5, 4.0567063461299446e-5,
       3.591405732740723e-5,  3.178402980354131e-5,  2.811965539165646e-5,  2.4869694240316126e-5,
       2.1988373166730962e-5, 1.9434825899529382e-5, 1.717258740121378e-5,  1.5169137499243157e-5,
       1.339548941011129e-5,  1.1825819079078403e-5, 1.0437131581057595e-5, 9.208961130078894e-6,
       8.12310153137994e-6,   7.163364176588591e-6,  6.315360932244386e-6,  5.566309502463164e-6,
       4.904859063429651e-6,  4.320934164082596e-6,  3.8055950719111903e-6, 3.350912911083174e-6,
       2.9498580949517117e-6, 2.596200697612328e-6,  2.2844215378879293e-6, 2.0096328693141094e-6,
       1.7675076766686654e-6, 1.5542166787225756e-6, 1.366372225473431e-6,  1.200978365778838e-6,
       1.0553864128982371e-6, 9.272554464808518e-7,  8.145171945902259e-7,  7.153448381918271e-7}};

  siPMShapeHO_.setNBin(nbin);

  double norm = 0.;
  for (unsigned int j = 0; j < nbin; ++j) {
    norm += (nt[j] > 0) ? nt[j] : 0.;
  }

  for (unsigned int j = 0; j < nbin; ++j) {
    nt[j] /= norm;
    siPMShapeHO_.setShapeBin(j, nt[j]);
  }
}

const HcalPulseShape& HcalPulseShapes::computeSiPMShapeHE203() {
  //numerical convolution of SiPM pulse + WLS fiber shape
  static const HcalPulseShape siPMShapeMC2017(
      normalize(convolve(nBinsSiPM_, analyticPulseShapeSiPMHE, Y11203), nBinsSiPM_), nBinsSiPM_);
  return siPMShapeMC2017;
}

const HcalPulseShape& HcalPulseShapes::computeSiPMShapeHE206() {
  //numerical convolution of SiPM pulse + WLS fiber shape
  //shift: aligning 206 phase closer to 205 in order to have good reco agreement
  static const HcalPulseShape siPMShapeMC2018(
      normalizeShift(convolve(nBinsSiPM_, analyticPulseShapeSiPMHE, Y11206), nBinsSiPM_, -2), nBinsSiPM_);
  return siPMShapeMC2018;
}

const HcalPulseShapes::Shape& HcalPulseShapes::getShape(int shapeType) const {
  ShapeMap::const_iterator shapeMapItr = theShapes.find(shapeType);
  if (shapeMapItr == theShapes.end()) {
    throw cms::Exception("HcalPulseShapes") << "unknown shapeType";
    return hpdShape_;  // should not return this, but...
  } else {
    return *(shapeMapItr->second);
  }
}

const HcalPulseShapes::Shape& HcalPulseShapes::shape(const HcalDetId& detId) const {
  if (!theDbService) {
    return defaultShape(detId);
  }
  int shapeType = theDbService->getHcalMCParam(detId)->signalShape();

  ShapeMap::const_iterator shapeMapItr = theShapes.find(shapeType);
  if (shapeMapItr == theShapes.end()) {
    return defaultShape(detId);
  } else {
    return *(shapeMapItr->second);
  }
}

const HcalPulseShapes::Shape& HcalPulseShapes::shapeForReco(const HcalDetId& detId) const {
  if (!theDbService) {
    return defaultShape(detId);
  }
  int shapeType = theDbService->getHcalRecoParam(detId.rawId())->pulseShapeID();

  ShapeMap::const_iterator shapeMapItr = theShapes.find(shapeType);
  if (shapeMapItr == theShapes.end()) {
    return defaultShape(detId);
  } else {
    return *(shapeMapItr->second);
  }
}

const HcalPulseShapes::Shape& HcalPulseShapes::defaultShape(const HcalDetId& detId) const {
  edm::LogWarning("HcalPulseShapes") << "Cannot find HCAL MC Params ";
  HcalSubdetector subdet = detId.subdet();
  switch (subdet) {
    case HcalBarrel:
      return hbShape();
    case HcalEndcap:
      return heShape();
    case HcalForward:
      return hfShape();
    case HcalOuter:
      //FIXME doesn't look for SiPMs
      return hoShape(false);
    default:
      throw cms::Exception("HcalPulseShapes") << "unknown detId";
      break;
  }
}

//SiPM helpers

inline double gexp(double t, double A, double c, double t0, double s) {
  static double const root2(sqrt(2));
  return -A * 0.5 * exp(c * t + 0.5 * c * c * s * s - c * s) * (erf(-0.5 * root2 / s * (t - t0 + c * s * s)) - 1);
}

inline double onePulse(double t, double A, double sigma, double theta, double m) {
  return (t < theta) ? 0 : A * TMath::LogNormal(t, sigma, theta, m);
}

double HcalPulseShapes::analyticPulseShapeSiPMHO(double t) {
  // HO SiPM pulse shape fit from Jake Anderson ca. 2013
  double A1(0.08757), c1(-0.5257), t01(2.4013), s1(0.6721);
  double A2(0.007598), c2(-0.1501), t02(6.9412), s2(0.8710);
  return gexp(t, A1, c1, t01, s1) + gexp(t, A2, c2, t02, s2);
}

double HcalPulseShapes::analyticPulseShapeSiPMHE(double t) {
  // taken from fit to laser measurement taken by Iouri M. in Spring 2016.
  double A1(5.204 / 6.94419), sigma1_shape(0.5387), theta1_loc(-0.3976), m1_scale(4.428);
  double A2(1.855 / 6.94419), sigma2_shape(0.8132), theta2_loc(7.025), m2_scale(12.29);
  return onePulse(t, A1, sigma1_shape, theta1_loc, m1_scale) + onePulse(t, A2, sigma2_shape, theta2_loc, m2_scale);
}

double HcalPulseShapes::generatePhotonTime(CLHEP::HepRandomEngine* engine, unsigned int signalShape) {
  if (signalShape == 206)
    return generatePhotonTime206(engine);
  else
    return generatePhotonTime203(engine);
}

double HcalPulseShapes::generatePhotonTime203(CLHEP::HepRandomEngine* engine) {
  double result(0.);
  while (true) {
    result = CLHEP::RandFlat::shoot(engine, HcalPulseShapes::Y11RANGE_);
    if (CLHEP::RandFlat::shoot(engine, HcalPulseShapes::Y11MAX203_) < HcalPulseShapes::Y11203(result))
      return result;
  }
}

double HcalPulseShapes::generatePhotonTime206(CLHEP::HepRandomEngine* engine) {
  double result(0.);
  while (true) {
    result = CLHEP::RandFlat::shoot(engine, HcalPulseShapes::Y11RANGE_);
    if (CLHEP::RandFlat::shoot(engine, HcalPulseShapes::Y11MAX206_) < HcalPulseShapes::Y11206(result))
      return result;
  }
}

//Original scintillator+Y11 fit from Vasken's 2001 measurement
double HcalPulseShapes::Y11203(double t) { return exp(-0.0635 - 0.1518 * t + log(t) * 2.528) / 2485.9; }

//New scintillator+Y11 model from Vasken's 2017 measurement plus a Landau correction term
double HcalPulseShapes::Y11206(double t) {
  //Shifting phase to have better comparison of digi shape with data
  //If necessary, further digi phase adjustment can be done here:
  //SimCalorimetry/HcalSimProducers/python/hcalSimParameters_cfi.py
  //by changing "timePhase"
  double shift = 7.2;

  //Fit From Deconvolved Data
  double A, n, t0, fit;
  A = 0.104204;
  n = 0.44064;
  t0 = 10.0186;
  if (t > shift)
    fit = A * (1 - exp(-(t - shift) / n)) * exp(-(t - shift) / t0);
  else
    fit = 0.0;

  //Correction Term
  double norm, mpv, sigma, corTerm;
  norm = 0.0809882;
  mpv = 0;
  sigma = 20;
  if (t > shift)
    corTerm = norm * TMath::Landau((t - shift), mpv, sigma);
  else
    corTerm = 0.0;

  //Overall Y11
  double frac = 0.11;
  double val = (1 - frac) * fit + frac * corTerm;

  if (val >= 0)
    return val;
  else
    return 0.0;
}
