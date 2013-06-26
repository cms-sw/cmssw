#include <cmath>
#include <iostream> 
#include <sstream> 
#include <cstdio>
#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
#include "L1Trigger/RPCTrigger/interface/RPCException.h"

#define m_pi 3.14159265358979

int RPCConst::iptFromPt(const double pt) {
  if(pt == 0.)return 0;
  if(pt<m_pts[0]) {
    //edm::LogError("RPCTrigger")<<"** RPCConst ** iptFromPt called with illegal pt="<<pt;
    std::string msg = "[RPCConst::iptFromPt] called with illegal pt=";
    std::ostringstream ostr;
    ostr<<pt;
    msg += ostr.str();
    throw RPCException(msg.c_str());
    return 0;
  }
 int ipt=RPCConst::IPT_MAX;
 while (pt < m_pts[ipt]) { ipt--; };
 return ipt;

}


double RPCConst::ptFromIpt(const int ipt) {
  
  if ( ipt<0 || ipt>RPCConst::IPT_MAX ) {
    //edm::LogError("RPCTrigger") <<"**RPCConst::ptFromIpt** problem with ipt: "<<ipt;
    std::string msg = "[RPCConst::ptFromIpt] problem with ipt: ";
    std::ostringstream ostr;
    ostr<<ipt;
    msg += ostr.str();
    throw RPCException(msg.c_str());
    return 0.;
  }
  else return m_pts[ipt];
}


double RPCConst::etaFromTowerNum(const int atower){

  int iabsitow = (atower >= 0)? atower: -atower;
  if (0==iabsitow) return 0.;
  if( iabsitow>RPCConst::ITOW_MAX) {
    //edm::LogError("RPCTrigger") << "**RPCConst::etaFromTowerNum** iabsitow>ITOW_MAX for m_tower:"
    //     << atower ;
    std::string msg = "[RPCConst::etaFromTowerNum] iabsitow>ITOW_MAX for m_tower:";
    std::ostringstream ostr;
    ostr<<atower;
    msg += ostr.str();
    throw RPCException(msg.c_str());
    return 0.;
  }
  double eta = (m_etas[iabsitow]+m_etas[iabsitow+1])/2.;
  return (atower>= 0) ? eta : -eta;
}


int RPCConst::towerNumFromEta(const double eta){
  int m_tower=0;
  double abseta = (eta >=0.) ? eta:-eta;
  while (m_tower<=ITOW_MAX){
      if(m_etas[m_tower] <= abseta && abseta< m_etas[m_tower+1])break;
      m_tower++;
  }
  if(m_tower > ITOW_MAX)
    m_tower = ITOW_MAX;
  return (eta>=0) ? m_tower:-m_tower; 
}

double RPCConst::phiFromSegmentNum(const int iseg) {
  double phi = OFFSET + 2.*m_pi*(iseg)/ (double) RPCConst::NSEG;
  return (phi <2.*m_pi) ? phi: phi-2.*m_pi;
}

double RPCConst::phiFromLogSegSec(const int logSegment, const int logSector) {
  int iseg = logSegment*12 + logSector;
  double phi = OFFSET + 2.*m_pi*(iseg)/ (double) RPCConst::NSEG;
  return (phi <2.*m_pi) ? phi: phi-2.*m_pi;
}

int RPCConst::segmentNumFromPhi(const double phi) {
  double iphi;
  if(phi-OFFSET < 0) {
    iphi = 2*m_pi + phi;
  }    
  else {
    iphi = phi-OFFSET;    
  }    
  int iseg = (int)(iphi * RPCConst::NSEG/(2.*m_pi));
  return iseg;
}

/*
int RPCConst::checkBarrel(const int atower) {
  int iabsitow = (atower >= 0)? atower: -atower;
  if(iabsitow <= RPCConst::ITOW_MAX_LOWPT) {
    return 1;
  } else if (iabsitow <= RPCConst::ITOW_MAX) {
    return 0;
  }
  return -1;
} */

double RPCConst::vxMuRate(int ptCode) {  
  double pt_ev = RPCConst::ptFromIpt(ptCode);
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
double RPCConst::vxIntegMuRate(int ptCode, double etaFrom, double etaTo) {  
  //calkowanie metoda trapezow - nie do konca dobre
  double rate = 0.5 * (vxMuRate(ptCode) + vxMuRate(ptCode+1))*
               (RPCConst::ptFromIpt(ptCode + 1) - RPCConst::ptFromIpt(ptCode));

  rate = rate * (etaTo - etaFrom);

  //edm::LogError("RPCTrigger")<<ptCode<<" "<<rate;//<<<<<<<<<<<<<<<<<<<<<<<<
  return rate;
}

//muon rate for pt from ptCode to ptCode + 1 in a given m_tower - only one!!! (mutliply by 2 to take oalso negative!!!)
//i.e for ptCode bin
double RPCConst::vxIntegMuRate(int ptCode, int m_tower) {  
  //calkowanie metoda trapezow - nie do konca dobre
  double rate = vxIntegMuRate(ptCode, RPCConst::m_etas[abs(m_tower)], RPCConst::m_etas[abs(m_tower)+1]);

  //edm::LogError("RPCTrigger")<<ptCode<<" "<<rate;//<<<<<<<<<<<<<<<<<<<<<<<<
  return rate;
}


/*
const int RPCConst::IPT_THRESHOLD [2][RPCConst::ITOW_MAX+1]={
//0   1   2   3   4     5   6   7   8   9    10  11  12  13  14  15  16  m_Tower
{17, 17, 17, 17, 17,   16, 16, 15, 17, 14,   12, 11, 12, 17, 16, 15, 15}, //LOW
{12, 12, 12, 12, 12,   11, 8,  11, 12, 9,    9,  8,  7,  11, 11, 11, 11} //VLOW
};
*/



const double RPCConst::m_pts[RPCConst::IPT_MAX+1]={
                        0.0,  0.01,    //<<<<<<<<<<<<<<<<<dla ptCode = 1 bylo 0, ale to powoduje problemy w vxMuRate
                        1.5,  2.0, 2.5,  3.0,  3.5,  4.0,  4.5, 
                        5.,   6.,   7.,   8.,  
                        10.,  12., 14.,  16.,  18.,  
                        20.,  25.,  30., 35.,  40.,  45., 
                        50.,  60.,  70., 80.,  90.,  100., 120., 140.};

// m_etas contain approximate lower egges of eta towers
// 0:ITOW_MAX  +additionaly upper edge  of last m_tower
const double RPCConst::m_etas[RPCConst::ITOW_MAX+2]=
                                         {0.00, 0.07, 0.27, 0.44, 0.58, 0.72,
                                          0.83, 0.93, 1.04, 1.14, 1.24, 1.36,
                                          1.48, 1.61, 1.73, 1.85, 1.97, 2.10};

// imported constants

    const std::string RPCConst::m_LOGPLANE_STR[RPCConst::m_LOGPLANES_COUNT] = {
      "m_LOGPLANE1", "m_LOGPLANE2", "m_LOGPLANE3", "m_LOGPLANE4", "m_LOGPLANE5", "m_LOGPLANE6"
    }; 
    
    const unsigned int RPCConst::m_LOGPLANE_SIZE[m_TOWER_COUNT][m_LOGPLANES_COUNT] = {
    //LOGPLANE  1,  2,  3   4   5   6
              {72, 56,  8, 40, 40, 24}, //TOWER 0
              {72, 56,  8, 40, 40, 24}, //TOWER 1
              {72, 56,  8, 40, 40, 24}, //TOWER 2
              {72, 56,  8, 40, 40, 24}, //TOWER 3
              {72, 56,  8, 40, 40, 24}, //TOWER 4
              {72, 56, 40,  8, 40, 24}, //TOWER 5
              {56, 72, 40,  8, 24,  0}, //TOWER 6
              {72, 56, 40,  8, 24,  0}, //TOWER 7
              {72, 24, 40,  8,  0,  0}, //TOWER 8
              {72,  8, 40,  0,  0,  0}, //TOWER 9
              {72,  8, 40, 24,  0,  0}, //TOWER 10
              {72,  8, 40, 24,  0,  0}, //TOWER 11
              {72,  8, 40, 24,  0,  0}, //TOWER 12
              {72,  8, 40, 24,  0,  0}, //TOWER 13
              {72,  8, 40, 24,  0,  0}, //TOWER 14
              {72,  8, 40, 24,  0,  0}, //TOWER 15
              {72,  8, 40, 24,  0,  0}  //TOWER 16
    };  



    const int RPCConst::m_VLPT_PLANES_COUNT[m_TOWER_COUNT] = {
      4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3,  3,  3,  3,  3,  3,  3
    };

    const int RPCConst::m_USED_PLANES_COUNT[m_TOWER_COUNT] = {
    //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
      6, 6, 6, 6, 6, 6, 5, 5, 4, 3, 4,  4,  4,  4,  4,  4,  4
    };

    const int RPCConst::m_REF_PLANE[m_TOWER_COUNT] = {
    //     0,         1,         2,         3,         4,
      m_LOGPLANE3, m_LOGPLANE3, m_LOGPLANE3, m_LOGPLANE3, m_LOGPLANE3,
    //     5,         6,         7,         8,
      m_LOGPLANE4,  m_LOGPLANE4, m_LOGPLANE4, m_LOGPLANE4,
    //     9,         10,       11,        12,        13,        14,         15,        16,
      m_LOGPLANE2, m_LOGPLANE2, m_LOGPLANE2, m_LOGPLANE2, m_LOGPLANE2,  m_LOGPLANE2, m_LOGPLANE2, m_LOGPLANE2
    };
    
    
/*    
    const int m_PT_CODE_MAX = 31; //!< Pt_code range = 0-m_PT_CODE_MAX
    
    const int m_LOGPLANE1 = 0; //!< The Logic Planes are named starting from '1', but in varoius loop indeks are from '0', that's why always use these consts 
    const int m_LOGPLANE2 = 1;
    const int m_LOGPLANE3 = 2;
    const int m_LOGPLANE4 = 3;
    const int m_LOGPLANE5 = 4;
    const int m_LOGPLANE6 = 5;
    
    const int m_FIRST_PLANE = m_LOGPLANE1; //!< Use ase a first index in loops.
    const int m_LAST_PLANE  = m_LOGPLANE6; //!< Use ase a last index in loops.
*/
    
//------- imported fucntions

int RPCConst::stringToInt(std::string str) {
  for(unsigned int i = 0; i < str.size(); i++)
    if(str[i] < '0' || str[i] > '9')
      throw RPCException("Error in stringToInt(): the string cannot be converted to a number");
      //edm::LogError("RPCTrigger")<< "Error in stringToInt(): the string cannot be converted to a number";
  return atoi(str.c_str());
}

//inline
std::string RPCConst::intToString(int number) {
  std::string str;
  /* Some problems. AK
  std::ostringstream ostr;
  ostr<<number;
  str = ostr.str();
  edm::LogError("RPCTrigger")<<"std::string intToString(int number)";
  edm::LogError("RPCTrigger")<<str;
  */
  char tmp[20];
  sprintf(tmp,"%d",number);
  str.append(tmp);
  return str;
}

 bool RPCConst::l1RpcConeCrdnts::operator <(const l1RpcConeCrdnts& cone) const{
  if(m_Tower != cone.m_Tower)
    return (m_Tower < cone.m_Tower);
  if(m_LogSector != cone.m_LogSector)
    return (m_LogSector < cone.m_LogSector);
  if(m_LogSegment != cone.m_LogSegment)
    return (m_LogSegment < cone.m_LogSegment);

  return false;
}

 bool RPCConst::l1RpcConeCrdnts::operator ==(const l1RpcConeCrdnts& cone) const{
  if(m_Tower != cone.m_Tower)
    return false;
  if(m_LogSector != cone.m_LogSector)
    return false;
  if(m_LogSegment != cone.m_LogSegment)
    return false;

  return true;
}



