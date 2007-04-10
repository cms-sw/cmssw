/**
 * Author: P.Paganini, Ursula Berthon
 * Created: 20 March 2007
 * $Id: EcalTPParameters.cc,v 1.1 2006/11/16 18:18:24 uberthon Exp $
 **/
#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

EcalTPParameters::EcalTPParameters() {
  //FIXME should be those used in TPG.txt.... could they be put in TPG.txt?
   ttfLowEB_=0;
   ttfHighEB_= 0;
   ttfLowEE_= 0;
   ttfHighEE_=0;
}

EcalTPParameters::~EcalTPParameters() {

}

void EcalTPParameters::changeThresholds(double ttfLowEB, double ttfHighEB, double ttfLowEE, double ttfHighEE) {
  if (  ttfLowEB_==ttfLowEB &&  ttfHighEB_== ttfHighEB &&  ttfLowEE_== ttfLowEE &&  ttfHighEE_==ttfHighEE) return;
  ttfLowEB_=ttfLowEB ;
  ttfHighEB_= ttfHighEB;
  ttfLowEE_= ttfLowEE;
  ttfHighEE_=ttfHighEE;
  update();
}

std::vector<unsigned int> EcalTPParameters::getTowerParameters(int SM, int towerInSM, bool print) const
{
  // SM = 1->36 , towerInSM = 1->68
  //  int index = 68*SM + towerInSM ;
  std::map<int, std::vector<unsigned int> >::const_iterator it ;
  it = towerParam_.find(getIndex(SM,towerInSM));
  if (it == towerParam_.end()) {
     throw cms::Exception("EcalTPG")<<"===> EcalTPParameters::getTowerParameters("<<std::dec<<SM<<", "<<towerInSM<<")";
  }
  std::vector<unsigned int> param = it->second ;
  if (print) {
    std::cout<<"===> EcalTPParameters::getTowerParameters("<<std::dec<<SM<<", "<<towerInSM<<")"<<std::endl ;
    for (int i=0 ; i<1024 ; i++) std::cout<<"LUT["<<std::dec<<i<<"] = "<<std::hex<<param[i]<<std::endl ;
    std::cout<<"Fine Grain:  el="<<param[1024]<<", eh="<<param[1025]
	     <<", tl="<<param[1026]<<",  th="<<param[1027]
	     <<", lut_fg="<<param[1028]<<std::endl ;
  }
  return param ;
}

std::vector<unsigned int> EcalTPParameters::getStripParameters(int SM, int towerInSM, int stripInTower, bool print) const
{
  // SM = 1->36 , towerInSM = 1->68, stripInTower = 1->5
  //  int index = (68*SM + towerInSM)*5 + stripInTower ;
  std::map<int, std::vector<unsigned int> >::const_iterator it ;
  it = stripParam_.find(getIndex(SM,towerInSM,stripInTower));
  if (it == stripParam_.end()) {
    throw cms::Exception("EcalTPG")<<"===> EcalTPParameters::getStripParameters("<<std::dec<<SM<<", "<<towerInSM<<", "<<stripInTower<<")";
  }
  std::vector<unsigned int> param = it->second ;
  if (print) {
    std::cout<<"===> EcalTPParameters::getStripParameters("<<std::dec<<SM<<", "<<towerInSM<<", "<<stripInTower<<")"<<std::endl ;
    std::cout<<"sliding window = "<<std::hex<<param[0]<<std::endl ;
    for (int i=0 ; i<5 ; i++) std::cout<<"Weight["<<std::dec<<i<<"] ="<<std::hex<<param[i+1]<<std::endl ;
  }
  return param ;
}

std::vector<unsigned int> EcalTPParameters::getXtalParameters(int SM, int towerInSM, int stripInTower, int xtalInStrip, bool print) const 
{
  // SM = 1->36 , towerInSM = 1->68, stripInTower = 1->5, xtalInStrip = 1->5
  //  int index = ((68*SM + towerInSM)*5 + stripInTower)*5 +  xtalInStrip ;
  std::map<int, std::vector<unsigned int> >::const_iterator it ;
  it = xtalParam_.find(getIndex(SM,towerInSM,stripInTower,xtalInStrip));
  if (it == xtalParam_.end()) {
    throw cms::Exception("EcalTPG")<<"===> EcalTPParameters::getXtalParameters("<<std::dec<<SM<<", "
				   <<towerInSM<<", "<<stripInTower<<", "<<xtalInStrip<<")";
  }
  std::vector<unsigned int> param = it->second ;
  if (print) {
    std::cout<<"===> EcalTPParameters::getXtalParameters("<<std::dec<<SM<<", "
	     <<towerInSM<<", "<<stripInTower<<", "<<xtalInStrip<<")"<<std::endl ;
    std::cout<<"Gain12, ped = "<<std::hex<<param[0]<<", mult = "<<param[1]<<", shift = "<<param[2]<<std::endl ;
    std::cout<<"Gain6,  ped = "<<std::hex<<param[3]<<", mult = "<<param[4]<<", shift = "<<param[5]<<std::endl ;
    std::cout<<"Gain1,  ped = "<<std::hex<<param[6]<<", mult = "<<param[7]<<", shift = "<<param[8]<<std::endl ;
  }
  return param ;
}

  int EcalTPParameters::getIndex(int SM, int towerInSM, int stripInTower, int xtalInStrip) const  { 
    // SM = 1->36 , towerInSM = 1->68, stripInTower = 1->5, xtalInStrip = 1->5
    const int NrTowersInSM=68;                  //FIXME
    const int NrStripsInT=5;
    const int NrXtalsInS=5;

    int i=NrTowersInSM + towerInSM ;
    if (stripInTower>0) i=i*NrStripsInT+stripInTower;
    if (xtalInStrip>0) i=i*NrXtalsInS+xtalInStrip;
    return i;
  }

