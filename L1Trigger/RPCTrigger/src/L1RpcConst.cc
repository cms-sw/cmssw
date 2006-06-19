#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#include <cmath>
#include <iostream> 
#include "L1Trigger/RPCTrigger/src/L1RpcConst.h"
using namespace std;



int L1RpcConst::iptFromPt(const double pt) {
  if(pt == 0.)return 0;
  if(pt<pts[0]) {
    edm::LogError("RPCTrigger")<<"** L1RpcConst ** iptFromPt called with illegal pt="<<pt;
    return 0;
  }
 int ipt=L1RpcConst::IPT_MAX;
 while (pt < pts[ipt] ) { ipt--; };
 return ipt;

}


double L1RpcConst::ptFromIpt(const int ipt) {
  if ( ipt<0 || ipt>L1RpcConst::IPT_MAX ) {
    edm::LogError("RPCTrigger") <<"**L1RpcConst::ptFromIpt** problem with ipt: "<<ipt;
    return 0.;
  }
  else return pts[ipt];
}


double L1RpcConst::etaFromTowerNum(const int atower){

  int iabsitow = (atower >= 0)? atower: -atower;
  if (0==iabsitow) return 0.;
  if( iabsitow>L1RpcConst::ITOW_MAX) {
    edm::LogError("RPCTrigger") << "**L1RpcConst::etaFromTowerNum** iabsitow>ITOW_MAX for tower:"
         << atower ;
    return 0.;
  }
  double eta = (etas[iabsitow]+etas[iabsitow+1])/2.;
  return (atower>= 0) ? eta : -eta;
}


int L1RpcConst::towerNumFromEta(const double eta){
  int tower=0;
  double abseta = (eta >=0.) ? eta:-eta;
  while (tower<=ITOW_MAX){
      if(etas[tower] <= abseta && abseta< etas[tower+1])break;
      tower++;
  }
  if(tower > ITOW_MAX)
    tower = ITOW_MAX;
  return (eta>=0) ? tower:-tower; 
}

double L1RpcConst::phiFromSegmentNum(const int iseg) {
  double phi = OFFSET + 2.*M_PI*( iseg )/ (double) L1RpcConst::NSEG;
  return (phi <2.*M_PI) ? phi: phi-2.*M_PI;
}

double L1RpcConst::phiFromLogSegSec(const int logSegment, const int logSector) {
  int iseg = logSegment*12 + logSector;
  double phi = OFFSET + 2.*M_PI*( iseg )/ (double) L1RpcConst::NSEG;
  return (phi <2.*M_PI) ? phi: phi-2.*M_PI;
}

int L1RpcConst::segmentNumFromPhi(const double phi) {
  double iphi;
  if(phi-OFFSET < 0) {
    iphi = 2*M_PI + phi;
  }    
  else {
    iphi = phi-OFFSET;    
  }    
  int iseg = (int)(iphi * L1RpcConst::NSEG/(2. * M_PI));
  return iseg;
}

/*
int L1RpcConst::checkBarrel(const int atower) {
  int iabsitow = (atower >= 0)? atower: -atower;
  if(iabsitow <= L1RpcConst::ITOW_MAX_LOWPT) {
    return 1;
  } else if (iabsitow <= L1RpcConst::ITOW_MAX) {
    return 0;
  }
  return -1;
} */

double L1RpcConst::VxMuRate(int ptCode) {  
  double pt_ev = L1RpcConst::ptFromIpt(ptCode);
  if (pt_ev == 0)
    return 0.0;
  const double lum = 2.0e33; //defoult is 1.0e34;
  const double dabseta = 1.0;
  const double dpt = 1.0;
  const double afactor = 1.0e-34*lum*dabseta*dpt;
  const double a  = 2*1.3084E6;
  const double mu=-0.725;
  const double sigma=0.4333;
  const double s2=2*sigma*sigma;
  
  double ptlog10;
  ptlog10 = log10(pt_ev);
  double ex = (ptlog10-mu)*(ptlog10-mu)/s2;
  double rate = (a * exp(-ex) * afactor); 

  //edm::LogError("RPCTrigger")<<ptCode<<" "<<rate;//<<<<<<<<<<<<<<<<<<<<<<<<
  return rate;
}

//muon rate for pt from ptCode to ptCode + 1
//i.e for ptCode bin
double L1RpcConst::VxIntegMuRate(int ptCode, double etaFrom, double etaTo) {  
  //calkowanie metoda trapezow - nie do konca dobre
  double rate = 0.5 * (VxMuRate(ptCode) + VxMuRate(ptCode+1) )* 
               (L1RpcConst::ptFromIpt(ptCode + 1) - L1RpcConst::ptFromIpt(ptCode) );

  rate = rate * (etaTo - etaFrom);

  //edm::LogError("RPCTrigger")<<ptCode<<" "<<rate;//<<<<<<<<<<<<<<<<<<<<<<<<
  return rate;
}

//muon rate for pt from ptCode to ptCode + 1 in a given tower - only one!!! (mutliply by 2 to take oalso negative!!!)
//i.e for ptCode bin
double L1RpcConst::VxIntegMuRate(int ptCode, int tower) {  
  //calkowanie metoda trapezow - nie do konca dobre
  double rate = VxIntegMuRate(ptCode, L1RpcConst::etas[abs(tower)], L1RpcConst::etas[abs(tower)+1]);

  //edm::LogError("RPCTrigger")<<ptCode<<" "<<rate;//<<<<<<<<<<<<<<<<<<<<<<<<
  return rate;
}


/*
const int L1RpcConst::IPT_THRESHOLD [2][L1RpcConst::ITOW_MAX+1]={
//0   1   2   3   4     5   6   7   8   9    10  11  12  13  14  15  16  Tower
{17, 17, 17, 17, 17,   16, 16, 15, 17, 14,   12, 11, 12, 17, 16, 15, 15}, //LOW
{12, 12, 12, 12, 12,   11, 8,  11, 12, 9,    9,  8,  7,  11, 11, 11, 11} //VLOW
};
*/



const double L1RpcConst::pts[L1RpcConst::IPT_MAX+1]={
                        0.0,  0.01,    //<<<<<<<<<<<<<<<<<dla ptCode = 1 bylo 0, ale to powoduje problemy w VxMuRate
                        1.5,  2.0, 2.5,  3.0,  3.5,  4.0,  4.5, 
                        5.,   6.,   7.,   8.,  
                        10.,  12., 14.,  16.,  18.,  
                        20.,  25.,  30., 35.,  40.,  45., 
                        50.,  60.,  70., 80.,  90.,  100., 120., 140. };

// etas contain approximate lower egges of eta towers
// 0:ITOW_MAX  +additionaly upper edge  of last tower
const double L1RpcConst::etas[L1RpcConst::ITOW_MAX+2]=
                                         {0.00, 0.07, 0.27, 0.44, 0.58, 0.72,
                                          0.83, 0.93, 1.04, 1.14, 1.24, 1.36,
                                          1.48, 1.61, 1.73, 1.85, 1.97, 2.10};





