#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "CondFormats/HcalObjects/interface/HcalMCParam.h"
#include "CondFormats/HcalObjects/interface/HcalMCParams.h"
#include "CondFormats/DataRecord/interface/HcalMCParamsRcd.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParam.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "CLHEP/Random/RandFlat.h"

// #include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include "TMath.h"

HcalPulseShapes::HcalPulseShapes() 
: theMCParams(0),
  theTopology(0),
  theRecoParams(0),
  theShapes()
{
/*

Reco  MC
--------------------------------------------------------------------------------------
000                                                   not used (reserved)
101   101      hpdShape_                              HPD (original version)
102   102      =101                                   HPD BV 30 volts in HBP iphi54
103   123      hpdShape_v2, hpdShapeMC_v2             HPD (2011. oct version)
104   124      hpdBV30Shape_v2, hpdBV30ShapeMC_v2     HPD bv30 in HBP iph54
105   125      hpdShape_v2, hpdShapeMC_v2             HPD (2011.11.12 version)
201   201      siPMShape_                             SiPMs Zecotec shape   (HO)
202   202      =201,                                  SiPMs Hamamatsu shape (HO)
203   203      siPMShape2017_                         SiPMs Hamamatsu shape (HE 2017)
205   205      siPMShapeData2017_                     SiPMs from Data (HE data 2017)
301   301      hfShape_                               regular HF PMT shape
401   401                                             regular ZDC shape
--------------------------------------------------------------------------------------

*/

 
  float ts1, ts2, ts3, thpd, tpre, wd1, wd2, wd3;

  //  HPD Shape  Version 1 (used before CMSSW5, until Oct 2011)
  ts1=8. ; ts2=10. ; ts3=29.3; thpd=4.0; tpre=9.0; wd1=2.0; wd2=0.7; wd3=1.0;  
  computeHPDShape(ts1,ts2,ts3,thpd,tpre,wd1,wd2,wd3, hpdShape_);
  theShapes[101] = &hpdShape_;
  theShapes[102] = theShapes[101];

  //  HPD Shape  Version 2 for CMSSW 5. Nov 2011  (RECO and MC separately)
  ts1=8. ; ts2=10. ; ts3=25.0; thpd=4.0; tpre=9.0; wd1=2.0; wd2=0.7; wd3=1.0;
  computeHPDShape(ts1,ts2,ts3,thpd,tpre,wd1,wd2,wd3, hpdShape_v2);
  theShapes[103] = &hpdShape_v2;

  ts1=8. ; ts2=10. ; ts3=29.3; thpd=4.0; tpre=7.0; wd1=2.0; wd2=0.7; wd3=1.0;
  computeHPDShape(ts1,ts2,ts3,thpd,tpre,wd1,wd2,wd3, hpdShapeMC_v2);
  theShapes[123] = &hpdShapeMC_v2;

  //  HPD Shape  Version 3 for CMSSW 5. Nov 2011  (RECO and MC separately)
  ts1=8. ; ts2=19. ; ts3=29.3; thpd=4.0; tpre=9.0; wd1=2.0; wd2=0.7; wd3=0.32;
  computeHPDShape(ts1,ts2,ts3,thpd,tpre,wd1,wd2,wd3, hpdShape_v3);
  theShapes[105] = &hpdShape_v3;

  ts1=8. ; ts2=10. ; ts3=22.3; thpd=4.0; tpre=7.0; wd1=2.0; wd2=0.7; wd3=1.0;
  computeHPDShape(ts1,ts2,ts3,thpd,tpre,wd1,wd2,wd3, hpdShapeMC_v3);
  theShapes[125] = &hpdShapeMC_v3;

  // HPD with Bias Voltage 30 volts, wider pulse.  (HBPlus iphi54)

  ts1=8. ; ts2=12. ; ts3=31.7; thpd=9.0; tpre=9.0; wd1=2.0; wd2=0.7; wd3=1.0;
  computeHPDShape(ts1,ts2,ts3,thpd,tpre,wd1,wd2,wd3, hpdBV30Shape_v2);
  theShapes[104] = &hpdBV30Shape_v2;

  ts1=8. ; ts2=12. ; ts3=31.7; thpd=9.0; tpre=9.0; wd1=2.0; wd2=0.7; wd3=1.0;
  computeHPDShape(ts1,ts2,ts3,thpd,tpre,wd1,wd2,wd3, hpdBV30ShapeMC_v2);
  theShapes[124] = &hpdBV30ShapeMC_v2;

  // HF and SiPM

  computeHFShape();
  computeSiPMShape();
  computeSiPMShape2017();
  computeSiPMShapeData2017();

  theShapes[201] = &siPMShape_;
  theShapes[202] = theShapes[201];
  theShapes[203] = &siPMShape2017_;
  theShapes[205] = &siPMShapeData2017_;
  theShapes[301] = &hfShape_;
  //theShapes[401] = new CaloCachedShapeIntegrator(&theZDCShape);

}


HcalPulseShapes::~HcalPulseShapes() {
  if (theMCParams) delete theMCParams;
  if (theRecoParams) delete theRecoParams;
  if (theTopology) delete theTopology;
}


void HcalPulseShapes::beginRun(edm::EventSetup const & es)
{
  edm::ESHandle<HcalMCParams> p;
  es.get<HcalMCParamsRcd>().get(p);
  theMCParams = new HcalMCParams(*p.product());

  edm::ESHandle<HcalTopology> htopo;
  es.get<HcalRecNumberingRecord>().get(htopo);
  theTopology=new HcalTopology(*htopo);
  theMCParams->setTopo(theTopology);

  edm::ESHandle<HcalRecoParams> q;
  es.get<HcalRecoParamsRcd>().get(q);
  theRecoParams = new HcalRecoParams(*q.product());
  theRecoParams->setTopo(theTopology);
}


void HcalPulseShapes::endRun()
{
  if (theMCParams) delete theMCParams;
  if (theRecoParams) delete theRecoParams;
  if (theTopology) delete theTopology;


  theMCParams = 0;
  theRecoParams = 0;
  theTopology = 0;
}


//void HcalPulseShapes::computeHPDShape()
void HcalPulseShapes::computeHPDShape(float ts1, float ts2, float ts3, float thpd, float tpre,
                                float wd1, float wd2, float wd3, Shape &tmphpdShape_)
{

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
  std::vector<float> ntmp(nbin,0.0);  // zeroing output pulse shape
  std::vector<float> nth(nbin,0.0);   // zeroing HPD drift shape
  std::vector<float> ntp(nbin,0.0);   // zeroing Binkley preamp shape
  std::vector<float> ntd(nbin,0.0);   // zeroing Scintillator decay shape

  unsigned int i,j,k;
  float norm;

  // HPD starts at I and rises to 2I in thpd of time
  norm=0.0;
  for(j=0;j<thpd && j<nbin;j++){
    nth[j] = 1.0 + ((float)j)/thpd;
    norm += nth[j];
  }
  // normalize integrated current to 1.0
  for(j=0;j<thpd && j<nbin;j++){
    nth[j] /= norm;
  }
  
  // Binkley shape over 6 time constants
  norm=0.0;
  for(j=0;j<6*tpre && j<nbin;j++){
    ntp[j] = ((float)j)*exp(-((float)(j*j))/(tpre*tpre));
    norm += ntp[j];
  }
  // normalize pulse area to 1.0
  for(j=0;j<6*tpre && j<nbin;j++){
    ntp[j] /= norm;
  }

// ignore stochastic variation of photoelectron emission
// <...>

// effective tile plus wave-length shifter decay time over 4 time constants
  unsigned int tmax = 6 * (int)ts3;
 
  norm=0.0;
  for(j=0;j<tmax && j<nbin;j++){
    ntd[j] = wd1 * exp(-((float)j)/ts1) + 
      wd2 * exp(-((float)j)/ts2) + 
      wd3 * exp(-((float)j)/ts3) ; 
    norm += ntd[j];
  }
  // normalize pulse area to 1.0
  for(j=0;j<tmax && j<nbin;j++){
    ntd[j] /= norm;
  }
  
  unsigned int t1,t2,t3,t4;
  for(i=0;i<tmax && i<nbin;i++){
    t1 = i;
    //    t2 = t1 + top*rand;
    // ignoring jitter from optical path length
    t2 = t1;
    for(j=0;j<thpd && j<nbin;j++){
      t3 = t2 + j;
      for(k=0;k<4*tpre && k<nbin;k++){       // here "4" is set deliberately,
	t4 = t3 + k;                         // as in test fortran toy MC ...
	if(t4<nbin){                         
	  unsigned int ntb=t4;                        
	  ntmp[ntb] += ntd[i]*nth[j]*ntp[k];
	}
      }
    }
  }
  
  // normalize for 1 GeV pulse height
  norm = 0.;
  for(i=0;i<nbin;i++){
    norm += ntmp[i];
  }

  for(i=0; i<nbin; i++){
    ntmp[i] /= norm;
  }

  for(i=0; i<nbin; i++){
    tmphpdShape_.setShapeBin(i,ntmp[i]);
  }
}

void HcalPulseShapes::computeHFShape() {
  // first create pulse shape over a range of time 0 ns to 255 ns in 1 ns steps
  unsigned int nbin = 256;
  hfShape_.setNBin(nbin);
  std::vector<float> ntmp(nbin,0.0);  // 

  const float k0=0.7956; // shape parameters
  const float p2=1.355;
  const float p4=2.327;
  const float p1=4.3;    // position parameter

  float norm = 0.0;

  for(unsigned int j = 0; j < 25 && j < nbin; ++j){

    float r0 = j-p1;
    float sigma0 = (r0<0) ? p2 : p2*p4;
    r0 /= sigma0;
    if(r0 < k0) ntmp[j] = exp(-0.5*r0*r0);
    else ntmp[j] = exp(0.5*k0*k0-k0*r0);
    norm += ntmp[j];
  }
  // normalize pulse area to 1.0
  for(unsigned int j = 0; j < 25 && j < nbin; ++j){
    ntmp[j] /= norm;
    hfShape_.setShapeBin(j,ntmp[j]);
  }
}


void HcalPulseShapes::computeSiPMShapeData2017()
{
  //From Jay Lawhorn: derived from data Edward Laird phase scan may2017
  //https://indico.cern.ch/event/641978/contributions/2604491/attachments/1468666/2271582/17-05-31-hcal-hep17-pulse-shape.pdf
  //Run numbers are 294736-294740 and 294929-294950

  unsigned int nbin = 250;

  std::vector<float> nt = {
    3.97958e-29,
    1.11634e-22,
    9.96106e-18,
    6.25334e-14,
    5.08863e-11,
    8.59141e-09,
    4.32285e-07,
    8.56617e-06,
    8.28549e-05,
    0.000461447,
    0.00168052,
    0.00441395,
    0.00901637,
    0.0151806,
    0.0220314,
    0.028528,
    0.0338471,
    0.0375578,
    0.0395985,
    0.0401567,
    0.0395398,
    0.0380776,
    0.0360669,
    0.0337474,
    0.0312984,
    0.0288457,
    0.0264721,
    0.0242276,
    0.0221393,
    0.0202181,
    0.0184647,
    0.0168731,
    0.0154335,
    0.0141346,
    0.0129639,
    0.0119094,
    0.0109594,
    0.0101031,
    0.0093305,
    0.00863267,
    0.0080015,
    0.00742977,
    0.00691107,
    0.00643969,
    0.00601059,
    0.00561931,
    0.00526188,
    0.00493483,
    0.00463505,
    0.00435981,
    0.00410667,
    0.00387348,
    0.00365832,
    0.00345949,
    0.00327547,
    0.0031049,
    0.00294656,
    0.00279938,
    0.00266237,
    0.00253467,
    0.00241548,
    0.0023041,
    0.00219989,
    0.00210227,
    0.00201072,
    0.00192476,
    0.00184397,
    0.00176795,
    0.00169634,
    0.00162884,
    0.00156512,
    0.00150494,
    0.00144803,
    0.00139418,
    0.00134317,
    0.00129481,
    0.00124894,
    0.00120537,
    0.00116398,
    0.00112461,
    0.00108715,
    0.00105147,
    0.00101747,
    0.000985042,
    0.000954096,
    0.000924545,
    0.000896308,
    0.000869311,
    0.000843482,
    0.000818758,
    0.000795077,
    0.000772383,
    0.000750623,
    0.000729747,
    0.00070971,
    0.000690466,
    0.000671977,
    0.000654204,
    0.00063711,
    0.000620663,
    0.000604831,
    0.000589584,
    0.000574894,
    0.000560735,
    0.000547081,
    0.00053391,
    0.0005212,
    0.000508929,
    0.000497078,
    0.000485628,
    0.000474561,
    0.000463862,
    0.000453514,
    0.000443501,
    0.000433811,
    0.000424429,
    0.000415343,
    0.00040654,
    0.00039801,
    0.000389741,
    0.000381722,
    0.000373944,
    0.000366398,
    0.000359074,
    0.000351964,
    0.00034506,
    0.000338353,
    0.000331838,
    0.000325505,
    0.00031935,
    0.000313365,
    0.000307544,
    0.000301881,
    0.000296371,
    0.000291009,
    0.000285788,
    0.000280705,
    0.000275755,
    0.000270932,
    0.000266233,
    0.000261653,
    0.00025719,
    0.000252837,
    0.000248593,
    0.000244454,
    0.000240416,
    0.000236475,
    0.00023263,
    0.000228876,
    0.000225212,
    0.000221633,
    0.000218138,
    0.000214724,
    0.000211389,
    0.00020813,
    0.000204945,
    0.000201831,
    0.000198787,
    0.000195811,
    0.0001929,
    0.000190053,
    0.000187268,
    0.000184543,
    0.000181876,
    0.000179266,
    0.000176711,
    0.00017421,
    0.000171761,
    0.000169363,
    0.000167014,
    0.000164713,
    0.000162459,
    0.00016025,
    0.000158086,
    0.000155964,
    0.000153885,
    0.000151847,
    0.000149848,
    0.000147888,
    0.000145966,
    0.000144081,
    0.000142232,
    0.000140418,
    0.000138638,
    0.000136891,
    0.000135177,
    0.000133494,
    0.000131843,
    0.000130221,
    0.00012863,
    0.000127066,
    0.000125531,
    0.000124023,
    0.000122543,
    0.000121088,
    0.000119658,
    0.000118254,
    0.000116874,
    0.000115518,
    0.000114185,
    0.000112875,
    0.000111587,
    0.000110321,
    0.000109076,
    0.000107851,
    0.000106648,
    0.000105464,
    0.000104299,
    0.000103154,
    0.000102027,
    0.000100918,
    9.98271e-05,
    9.87537e-05,
    9.76974e-05,
    9.66578e-05,
    9.56346e-05,
    9.46274e-05,
    9.3636e-05,
    9.26599e-05,
    9.16989e-05,
    9.07526e-05,
    8.98208e-05,
    8.89032e-05,
    8.79995e-05,
    8.71093e-05,
    8.62325e-05,
    8.53688e-05,
    8.45179e-05,
    8.36796e-05,
    8.28536e-05,
    8.20397e-05,
    8.12376e-05,
    8.04471e-05,
    7.96681e-05,
    7.89002e-05,
    7.81433e-05,
    7.73972e-05,
    7.66616e-05,
    7.59364e-05,
    7.52213e-05,
    7.45163e-05,
    7.3821e-05,
    7.31354e-05,
    7.24592e-05,
    7.17923e-05,
    7.11345e-05,
    7.04856e-05,
    6.98455e-05,
    6.9214e-05,
    6.8591e-05
  };


  siPMShapeData2017_.setNBin(nbin);

  double norm = 0.;
  for (unsigned int j = 0; j < nbin; ++j) {
    norm += (nt[j]>0) ? nt[j] : 0.;
  }

  for (unsigned int j = 0; j < nbin; ++j) {
    nt[j] /= norm;
    siPMShapeData2017_.setShapeBin(j,nt[j]);
  }
}

void HcalPulseShapes::computeSiPMShape()
{

  unsigned int nbin = 128; 

//From Jake Anderson: toy MC convolution of SiPM pulse + WLS fiber shape + SiPM nonlinear response
  std::vector<float> nt = {
    2.782980485851731e-6,
    4.518134885954626e-5,
    2.7689305197392056e-4,
    9.18328418900969e-4,
    .002110072599166349,
    .003867856860331454,
    .006120046224897771,
    .008754774090536956,
    0.0116469503358586,
    .01467007449455966,
    .01770489955229477,
    .02064621450689512,
    .02340678093764222,
    .02591874610854916,
    .02813325527435303,
    0.0300189241965647,
    .03155968107671164,
    .03275234052577155,
    .03360415306318798,
    .03413048377960748,
    .03435270899678218,
    .03429637464659661,
    .03398962975487166,
    .03346192884394954,
    .03274298516247742,
    .03186195009136525,
    .03084679116113031,
    0.0297238406141036,
    .02851748748929785,
    .02724998816332392,
    .02594137274487424,
    .02460942736731527,
    .02326973510736116,
    .02193576080366117,
    0.0206189674254987,
    .01932895378564653,
    0.0180736052958666,
    .01685925112650875,
    0.0156908225633535,
    .01457200857138456,
    .01350540559602467,
    .01249265947824805,
    .01153459805300423,
    .01063135355597282,
    .009782474412011936,
    .008987026319784546,
    0.00824368281357106,
    .007550805679909604,
    .006906515742762193,
    .006308754629755056,
    .005755338185695127,
    .005244002229973356,
    .004772441359900532,
    .004338341490928299,
    .003939406800854143,
    0.00357338171220501,
    0.0032380685079891,
    .002931341133259233,
    .002651155690306086,
    .002395558090237333,
    .002162689279320922,
    .001950788415487319,
    .001758194329648101,
    .001583345567913682,
    .001424779275191974,
    .001281129147671334,
    0.00115112265163774,
    .001033577678808199,
    9.273987838127585e-4,
    8.315731274976846e-4,
    7.451662302008696e-4,
    6.673176219006913e-4,
    5.972364609644049e-4,
    5.341971801529036e-4,
    4.775352065178378e-4,
    4.266427928961177e-4,
    3.8096498904225923e-4,
    3.3999577417327287e-4,
    3.032743659102713e-4,
    2.703817158798329e-4,
    2.4093719775272793e-4,
    2.145954900503894e-4,
    1.9104365317752797e-4,
    1.6999839784346724e-4,
    1.5120354022478893e-4,
    1.3442763782650755e-4,
    1.1946179895521507e-4,
    1.0611765796993575e-4,
    9.422550797617687e-5,
    8.363258233342666e-5,
    7.420147621931836e-5,
    6.580869950304933e-5,
    5.834335229919868e-5,
    5.17059147771959e-5,
    4.5807143072062634e-5,
    4.0567063461299446e-5,
    3.591405732740723e-5,
    3.178402980354131e-5,
    2.811965539165646e-5,
    2.4869694240316126e-5,
    2.1988373166730962e-5,
    1.9434825899529382e-5,
    1.717258740121378e-5,
    1.5169137499243157e-5,
    1.339548941011129e-5,
    1.1825819079078403e-5,
    1.0437131581057595e-5,
    9.208961130078894e-6,
    8.12310153137994e-6,
    7.163364176588591e-6,
    6.315360932244386e-6,
    5.566309502463164e-6,
    4.904859063429651e-6,
    4.320934164082596e-6,
    3.8055950719111903e-6,
    3.350912911083174e-6,
    2.9498580949517117e-6,
    2.596200697612328e-6,
    2.2844215378879293e-6,
    2.0096328693141094e-6,
    1.7675076766686654e-6,
    1.5542166787225756e-6,
    1.366372225473431e-6,
    1.200978365778838e-6,
    1.0553864128982371e-6,
    9.272554464808518e-7,
    8.145171945902259e-7,
    7.153448381918271e-7
  };

  siPMShape_.setNBin(nbin);

  double norm = 0.;
  for (unsigned int j = 0; j < nbin; ++j) {
    norm += (nt[j]>0) ? nt[j] : 0.;
  }

  for (unsigned int j = 0; j < nbin; ++j) {
    nt[j] /= norm;
    siPMShape_.setShapeBin(j,nt[j]);
  }
}

void HcalPulseShapes::computeSiPMShape2017()
{
  //numerical convolution of SiPM pulse + WLS fiber shape
  std::vector<double> nt = convolve(nBinsSiPM_,analyticPulseShapeSiPMHE,Y11TimePDF);

  siPMShape2017_.setNBin(nBinsSiPM_);

  //skip first bin, always 0
  double norm = 0.;
  for (unsigned int j = 1; j <= nBinsSiPM_; ++j) {
    norm += (nt[j]>0) ? nt[j] : 0.;
  }

  for (unsigned int j = 1; j <= nBinsSiPM_; ++j) {
    nt[j] /= norm;
    siPMShape2017_.setShapeBin(j,nt[j]);
  }
}

const HcalPulseShapes::Shape &
HcalPulseShapes::getShape(int shapeType) const
{
  ShapeMap::const_iterator shapeMapItr = theShapes.find(shapeType);
  if(shapeMapItr == theShapes.end()) {
   throw cms::Exception("HcalPulseShapes") << "unknown shapeType";
   return  hpdShape_;   // should not return this, but...
  } else {
    return *(shapeMapItr->second);
  }
}


const HcalPulseShapes::Shape &
HcalPulseShapes::shape(const HcalDetId & detId) const
{
  if(!theMCParams) {
    return defaultShape(detId);
  }
  int shapeType = theMCParams->getValues(detId)->signalShape();

  ShapeMap::const_iterator shapeMapItr = theShapes.find(shapeType);
  if(shapeMapItr == theShapes.end()) {
    return defaultShape(detId);
  } else {
    return *(shapeMapItr->second);
  }
}

const HcalPulseShapes::Shape &
HcalPulseShapes::shapeForReco(const HcalDetId & detId) const
{
  if(!theRecoParams) {
    return defaultShape(detId);
  }
  int shapeType = theRecoParams->getValues(detId.rawId())->pulseShapeID();

  ShapeMap::const_iterator shapeMapItr = theShapes.find(shapeType);
  if(shapeMapItr == theShapes.end()) {
    return defaultShape(detId);
  } else {
    return *(shapeMapItr->second);
  }
}


const HcalPulseShapes::Shape &
HcalPulseShapes::defaultShape(const HcalDetId & detId) const
{
  edm::LogWarning("HcalPulseShapes") << "Cannot find HCAL MC Params ";
  HcalSubdetector subdet = detId.subdet();
  switch(subdet) {
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
  return -A*0.5*exp(c*t+0.5*c*c*s*s-c*s)*(erf(-0.5*root2/s*(t-t0+c*s*s))-1);
}

inline double onePulse(double t, double A, double sigma, double theta, double m) {
  return (t<theta) ? 0 : A*TMath::LogNormal(t,sigma,theta,m);
}

double HcalPulseShapes::analyticPulseShapeSiPMHO(double t) {
  // HO SiPM pulse shape fit from Jake Anderson ca. 2013
  double A1(0.08757), c1(-0.5257), t01(2.4013), s1(0.6721);
  double A2(0.007598), c2(-0.1501), t02(6.9412), s2(0.8710);
  return gexp(t,A1,c1,t01,s1) + gexp(t,A2,c2,t02,s2);
}

double HcalPulseShapes::analyticPulseShapeSiPMHE(double t) {
  // taken from fit to laser measurement taken by Iouri M. in Spring 2016.
  double A1(5.204/6.94419), sigma1_shape(0.5387), theta1_loc(-0.3976), m1_scale(4.428);
  double A2(1.855/6.94419), sigma2_shape(0.8132), theta2_loc(7.025),   m2_scale(12.29);
  return
    onePulse(t,A1,sigma1_shape,theta1_loc,m1_scale) +
    onePulse(t,A2,sigma2_shape,theta2_loc,m2_scale);
}

double HcalPulseShapes::generatePhotonTime(CLHEP::HepRandomEngine* engine) {
  double result(0.);
  while (true) {
    result = CLHEP::RandFlat::shoot(engine, HcalPulseShapes::Y11RANGE_);
    if (CLHEP::RandFlat::shoot(engine, HcalPulseShapes::Y11MAX_) < HcalPulseShapes::Y11TimePDF(result))
      return result;
  }
}

double HcalPulseShapes::Y11TimePDF(double t) {
  return exp(-0.0635-0.1518*t)*pow(t, 2.528)/2485.9;
}
