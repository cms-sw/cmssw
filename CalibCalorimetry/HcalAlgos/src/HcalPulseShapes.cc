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

  theShapes[201] = &siPMShape_;
  theShapes[202] = theShapes[201];
  theShapes[203] = &siPMShape2017_;
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

//From Kevin Pedro: numerical convolution of SiPM pulse + WLS fiber shape
  std::vector<float> nt = {
    2.03644e-08,
    1.31271e-05,
    0.000104047,
    0.000396473,
    0.00101485,
    0.00203522,
    0.00347024,
    0.00527766,
    0.00737819,
    0.00967527,
    0.0120718,
    0.0144801,
    0.0168259,
    0.0190497,
    0.021106,
    0.0229621,
    0.0245962,
    0.025996,
    0.027157,
    0.0280809,
    0.0287747,
    0.0292488,
    0.0295167,
    0.0295937,
    0.0294966,
    0.0292426,
    0.0288491,
    0.0283336,
    0.0277129,
    0.0270028,
    0.0262187,
    0.0253747,
    0.0244838,
    0.0235579,
    0.0226078,
    0.0216431,
    0.0206725,
    0.0197035,
    0.0187426,
    0.0177953,
    0.0168665,
    0.0159601,
    0.0150793,
    0.0142267,
    0.0134044,
    0.0126138,
    0.0118559,
    0.0111314,
    0.0104404,
    0.00978306,
    0.00915898,
    0.00856766,
    0.00800843,
    0.00748045,
    0.00698277,
    0.00651434,
    0.00607406,
    0.00566077,
    0.00527327,
    0.00491036,
    0.00457083,
    0.00425349,
    0.00395714,
    0.00368063,
    0.00342282,
    0.00318263,
    0.00295898,
    0.00275087,
    0.00255732,
    0.00237739,
    0.00221021,
    0.00205493,
    0.00191076,
    0.00177693,
    0.00165275,
    0.00153754,
    0.00143068,
    0.00133157,
    0.00123966,
    0.00115444,
    0.00107542,
    0.00100215,
    0.000934208,
    0.000871209,
    0.000812786,
    0.000758599,
    0.000708335,
    0.000661701,
    0.000618427,
    0.000578262,
    0.000540972,
    0.000506344,
    0.000474177,
    0.000444286,
    0.000416502,
    0.000390667,
    0.000366635,
    0.000344272,
    0.000323452,
    0.000304062,
    0.000285995,
    0.000269153,
    0.000253446,
    0.00023879,
    0.000225109,
    0.00021233,
    0.000200389,
    0.000189225,
    0.000178781,
    0.000169007,
    0.000159854,
    0.000151278,
    0.000143239,
    0.000135699,
    0.000128622,
    0.000121978,
    0.000115735,
    0.000109867,
    0.000104348,
    9.91543e-05,
    9.42644e-05,
    8.96579e-05,
    8.5316e-05,
    8.12213e-05,
    7.73578e-05,
    7.37106e-05,
    7.02656e-05,
    6.701e-05,
    6.3932e-05,
    6.10202e-05,
    5.82644e-05,
    5.5655e-05,
    5.3183e-05,
    5.084e-05,
    4.86183e-05,
    4.65107e-05,
    4.45103e-05,
    4.26109e-05,
    4.08066e-05,
    3.90918e-05,
    3.74615e-05,
    3.59108e-05,
    3.44353e-05,
    3.30307e-05,
    3.16932e-05,
    3.04189e-05,
    2.92045e-05,
    2.80467e-05,
    2.69424e-05,
    2.58889e-05,
    2.48834e-05,
    2.39233e-05,
    2.30064e-05,
    2.21304e-05,
    2.12932e-05,
    2.04928e-05,
    1.97273e-05,
    1.89951e-05,
    1.82944e-05,
    1.76236e-05,
    1.69814e-05,
    1.63663e-05,
    1.5777e-05,
    1.52123e-05,
    1.46709e-05,
    1.41518e-05,
    1.3654e-05,
    1.31764e-05,
    1.27181e-05,
    1.22782e-05,
    1.18558e-05,
    1.14502e-05,
    1.10606e-05,
    1.06863e-05,
    1.03266e-05,
    9.98081e-06,
    9.64837e-06,
    9.32866e-06,
    9.02114e-06,
    8.72529e-06,
    8.44058e-06,
    8.16655e-06,
    7.90275e-06,
    7.64873e-06,
    7.40408e-06,
    7.16842e-06,
    6.94136e-06,
    6.72255e-06,
    6.51166e-06,
    6.30834e-06,
    6.11231e-06,
    5.92325e-06,
    5.74089e-06,
    5.56497e-06,
    5.39521e-06,
    5.23138e-06,
    5.07325e-06,
    4.92058e-06,
    4.77317e-06,
    4.63081e-06,
    4.49331e-06,
    4.36047e-06,
    4.23212e-06,
    4.10808e-06,
    3.9882e-06,
    3.87231e-06,
    3.76027e-06,
    3.65193e-06,
    3.54716e-06,
    3.44581e-06,
    3.34777e-06,
    3.25291e-06,
    3.16111e-06,
    3.07227e-06,
    2.98628e-06,
    2.90302e-06,
    2.82242e-06,
    2.74436e-06,
    2.66877e-06,
    2.59555e-06,
    2.52461e-06,
    2.45588e-06,
    2.38928e-06,
    2.32474e-06,
    2.26219e-06,
    2.20154e-06,
    2.14275e-06,
    2.08575e-06,
    2.03047e-06,
    1.97685e-06,
    1.92485e-06,
    1.8744e-06,
    1.82546e-06,
    1.77796e-06,
    1.73187e-06,
    1.68714e-06,
    1.64373e-06,
    1.60158e-06,
    1.56066e-06,
    1.52092e-06,
    1.48234e-06,
    1.44487e-06,
    1.40847e-06,
    1.37311e-06,
    1.33876e-06,
    1.30539e-06,
    1.27296e-06,
    1.24144e-06,
    1.21081e-06,
    1.18104e-06
  };

  siPMShape2017_.setNBin(nt.size());

  double norm = 0.;
  for (unsigned int j = 0; j < nt.size(); ++j) {
    norm += (nt[j]>0) ? nt[j] : 0.;
  }

  for (unsigned int j = 0; j < nt.size(); ++j) {
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