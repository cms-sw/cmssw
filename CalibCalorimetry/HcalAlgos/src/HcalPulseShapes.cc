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
#include "Geometry/Records/interface/IdealGeometryRecord.h"

// #include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include <cmath>

#include <iostream>
#include <fstream>

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

  theShapes[201] = &siPMShape_;
  theShapes[202] = theShapes[201];
  theShapes[301] = &hfShape_;
  //theShapes[401] = new CaloCachedShapeIntegrator(&theZDCShape);

  /*
  // backward-compatibility with old scheme
  theShapes[0] = theShapes[101];
  //FIXME "special" HB
  theShapes[1] = theShapes[101];
  theShapes[2] = theShapes[201];
  theShapes[3] = theShapes[301];
  //theShapes[4] = theShapes[401];
  */
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
  es.get<IdealGeometryRecord>().get(htopo);
  theTopology=new HcalTopology(*htopo);
  theMCParams->setTopo(theTopology);

  edm::ESHandle<HcalRecoParams> q;
  es.get<HcalRecoParamsRcd>().get(q);
  theRecoParams = new HcalRecoParams(*q.product());
  theRecoParams->setTopo(theTopology);

//      std::cout<<" skdump in HcalPulseShapes::beginRun   dupm MCParams "<<std::endl;
//      std::ofstream skfile("skdumpMCParamsNewFormat.txt");
//      HcalDbASCIIIO::dumpObject(skfile, (*theMCParams) );
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

  /*
  std::cout << "o HcalPulseShapes::computeHPDShape  " 
            << " ts1, ts2, ts3, thpd, tpre, w1, w2, w3 =" 
	    <<  ts1 << ", " << ts2 << ", " << ts3 << ", " 
	    << thpd << ", " << tpre << ", " << wd1 << ", " <<  wd2 
            << ", "  << wd3 << std::endl;
  */

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
  // pulse shape componnts over a range of time 0 ns to 255 ns in 1 ns steps
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

  //cout << " Convoluted SHAPE ==============  " << endl;
  for(i=0; i<nbin; i++){
    ntmp[i] /= norm;
    //  cout << " shape " << i << " = " << ntmp[i] << endl;   
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
  unsigned int nbin = 512;
  siPMShape_.setNBin(nbin);
  std::vector<float> nt(nbin,0.0);  //

  double norm = 0.;
  for (unsigned int j = 0; j < nbin; ++j) {
    if (j <= 31.)
      nt[j] = 2.15*j;
    else if ((j > 31) && (j <= 96))
      nt[j] = 102.1 - 1.12*j;
    else
      nt[j] = 0.0076*j - 6.4;
    norm += (nt[j]>0) ? nt[j] : 0.;
  }

  for (unsigned int j = 0; j < nbin; ++j) {
    nt[j] /= norm;
    siPMShape_.setShapeBin(j,nt[j]);
  }
}


const HcalPulseShapes::Shape &
HcalPulseShapes::getShape(int shapeType) const
{

  //  std::cout << "- HcalPulseShapes::Shape for type "<< shapeType 
  //            << std::endl;

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
  /*
	  int sub     = detId.subdet();
	  int depth   = detId.depth();
	  int inteta  = detId.ieta();
	  int intphi  = detId.iphi();
	  
	  std::cout << " HcalPulseShapes::shape cell:" 
		    << " sub, ieta, iphi, depth = " 
		    << sub << "  " << inteta << "  " << intphi 
		    << "  " << depth  << " => ShapeId "<<  shapeType 
		    << std::endl;
  */
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
  /*
	  int sub     = detId.subdet();
	  int depth   = detId.depth();
	  int inteta  = detId.ieta();
	  int intphi  = detId.iphi();
	  
	  std::cout << ">> HcalPulseShapes::shapeForReco cell:" 
		    << " sub, ieta, iphi, depth = " 
		    << sub << "  " << inteta << "  " << intphi 
		    << "  " << depth  << " => ShapeId "<<  shapeType 
		    << std::endl;
  */

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

