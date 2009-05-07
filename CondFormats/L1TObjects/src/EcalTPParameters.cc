/**
 * Author: P.Paganini, Ursula Berthon
 * Created: 20 March 2007
 * $Id: EcalTPParameters.cc,v 1.6 2007/06/14 17:14:33 uberthon Exp $
 **/
#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

int EcalTPParameters::nrMinTccEB_=0;
int EcalTPParameters::nrMaxTccEB_=0;
int EcalTPParameters::nbMaxXtals_=0;
int EcalTPParameters::nbMaxStrips_=0;
int EcalTPParameters::nbMaxTowers_=0;

EcalTPParameters::EcalTPParameters()   {
  //FIXME should be those used in TPG.txt.... could they be put in TPG.txt?
  ttfLowEB_=0;
  ttfHighEB_= 0;
  ttfLowEE_= 0;
  ttfHighEE_=0;
}

EcalTPParameters::~EcalTPParameters() {

}
void EcalTPParameters::setConstants(const int nbMaxTowers, const int nbMaxStrips, const int nbMaxXtals, const int nrMinTccEB, const int nrMaxTccEB) {
  nbMaxTowers_=nbMaxTowers;
  nbMaxStrips_=nbMaxStrips;
  nbMaxXtals_=nbMaxXtals;
  nrMinTccEB_=nrMinTccEB;
  nrMaxTccEB_=nrMaxTccEB;
}

void EcalTPParameters::changeThresholds(double ttfLowEB, double ttfHighEB, double ttfLowEE, double ttfHighEE) {
  if (  ttfLowEB_==ttfLowEB &&  ttfHighEB_== ttfHighEB &&  ttfLowEE_== ttfLowEE &&  ttfHighEE_==ttfHighEE) return;
  ttfLowEB_=ttfLowEB ;
  ttfHighEB_= ttfHighEB;
  ttfLowEE_= ttfLowEE;
  ttfHighEE_=ttfHighEE;
  update();
}

std::vector<unsigned int> const * EcalTPParameters::getTowerParameters(int TCC, int towerInTCC, bool print) const
{
  std::map<int, std::vector<unsigned int> >::const_iterator it ;
  it = towerParam_.find(getIndex(TCC,towerInTCC));
  if (it == towerParam_.end()) {
    throw cms::Exception("EcalTPG")<<"===> EcalTPParameters::getTowerParameters("<<std::dec<<TCC<<", "<<towerInTCC<<")";
  }
  if (print) {
    std::vector<unsigned int> param = it->second ;
    if (TCC>=nrMinTccEB_   && TCC<= nrMaxTccEB_ ) {
      std::cout<<"===> EcalTPParameters::getTowerParameters("<<std::dec<<TCC<<", "<<towerInTCC<<")"<<std::endl ;
      //    for (int i=0 ; i<1024 ; i++) std::cout<<"LUT["<<std::dec<<i<<"] = "<<std::hex<<param[i]<<std::endl ;
      // barrel only
      std::cout<<"Fine Grain:  el="<<param[1024]<<", eh="<<param[1025]  //FIXME 
	       <<", tl="<<param[1026]<<",  th="<<param[1027]
	       <<", lut_fg="<<param[1028]<<std::endl ;
    } else
      std::cout<<"Fine Grain:  tower_lut_fg="<<std::hex<<param[1024]<<std::endl ;
  }
  return &(it->second);
}

std::vector<unsigned int> const * EcalTPParameters::getStripParameters(int TCC, int towerInTCC, int stripInTower, bool print) const
{
  std::map<int, std::vector<unsigned int> >::const_iterator it ;
  it = stripParam_.find(getIndex(TCC,towerInTCC,stripInTower));
  if (it == stripParam_.end()) {
    throw cms::Exception("EcalTPG")<<"===> EcalTPParameters::getStripParameters("<<std::dec<<TCC<<", "<<towerInTCC<<", "<<stripInTower<<")";
  }
  if (print) {
    std::vector<unsigned int> param = it->second ;
    param = it->second ;
    std::cout<<"===> EcalTPParameters::getStripParameters("<<std::dec<<TCC<<", "<<towerInTCC<<", "<<stripInTower<<")"<<std::endl ;
    std::cout<<"sliding window = "<<std::hex<<param[0]<<std::endl ;
    for (int i=0 ; i<5 ; i++) std::cout<<"Weight["<<std::dec<<i<<"] ="<<std::hex<<param[i+1]<<std::endl ;
  }
  return &(it->second);
}

std::vector<unsigned int> const *  EcalTPParameters::getXtalParameters(int TCC, int towerInTCC, int stripInTower, int xtalInStrip, bool print) const 
{

  std::map<int, std::vector<unsigned int> >::const_iterator it ;
  it = xtalParam_.find(getIndex(TCC,towerInTCC,stripInTower,xtalInStrip));
  if (it == xtalParam_.end()) {
    throw cms::Exception("EcalTPG")<<"===> EcalTPParameters::couldnt find entry for getXtalParameters("<<std::dec<<TCC<<", "
				   <<towerInTCC<<", "<<stripInTower<<", "<<xtalInStrip<<")";
  }
  if (print) {
    std::vector<unsigned int> param = it->second ;
    std::cout<<"===> EcalTPParameters::getXtalParameters("<<std::dec<<TCC<<", "
	     <<towerInTCC<<", "<<stripInTower<<", "<<xtalInStrip<<")"<<std::endl ;
    std::cout<<"Gain12, ped = "<<std::hex<<param[0]<<", mult = "<<param[1]<<", shift = "<<param[2]<<std::endl ;
    std::cout<<"Gain6,  ped = "<<std::hex<<param[3]<<", mult = "<<param[4]<<", shift = "<<param[5]<<std::endl ;
    std::cout<<"Gain1,  ped = "<<std::hex<<param[6]<<", mult = "<<param[7]<<", shift = "<<param[8]<<std::endl ;
  }
  return &(it->second);
}

int EcalTPParameters::getIndex(int TCC, int towerInTCC, int stripInTower, int xtalInStrip) const  { 

  int i=nbMaxTowers_*TCC + towerInTCC ;
  if (stripInTower>0) i=i*nbMaxStrips_+stripInTower;
  if (xtalInStrip>0) i=i*nbMaxXtals_+xtalInStrip;
  return i;
}

double EcalTPParameters::getTPGinGeVEB(unsigned int TCC, unsigned int towerInTCC, unsigned int compressedEt) const {
  //FIXME!! 1024 should be a database constant....
    
  double lsb_tcp = EtSatEB_/1024 ;//FIXME!!
  std::vector<unsigned int> const *lut;
  lut=this->getTowerParameters(TCC,towerInTCC,lut) ;
  if (lut->size() <1024) { //FIXME!!
    // FIXME should throw an exception!
    return 0. ;
  }

  unsigned int lin_TPG = 1024 ;
  for (unsigned int i=0 ; i<1024 ; i++) {
    if ((*lut)[i] == compressedEt) {
      lin_TPG = i ;
      break ;
    }
  }
  if (lin_TPG >= 1024) {
    // FIXME should throw an exception!
    return 0. ;
  } 

  return lin_TPG*lsb_tcp ;
}

double EcalTPParameters::getTPGinGeVEE(unsigned int TCC, unsigned int towerInTCC, unsigned int compressedEt) const {
//FIXME!! 1024 should be a database constant....
    
  double lsb_tcp = EtSatEE_/1024 ;//FIXME!!
  std::vector<unsigned int> *lut;
  this->getTowerParameters(TCC,towerInTCC,lut) ;
   if (lut->size() <1024) { //FIXME!!
    // FIXME should throw an exception!
    return 0. ;
  }

  unsigned int lin_TPG = 1024 ;
  for (unsigned int i=0 ; i<1024 ; i++) {
   if ((*lut)[i] == compressedEt) {
     lin_TPG = i ;
     break ;
   }
  }
  if (lin_TPG >= 1024) {
    // FIXME should throw an exception!
    return 0. ;
  } 

  return lin_TPG*lsb_tcp ;
}

void EcalTPParameters::update() {
  //FIXME: endcap?

  // 1st ttf thresholds:
  double lsb_tcp_EB = EtSatEB_/1024 ;
  unsigned int ttfLowEB_ADC = static_cast<unsigned int>(ttfLowEB_/lsb_tcp_EB) ;
  unsigned int ttfHighEB_ADC =  static_cast<unsigned int>(ttfHighEB_/lsb_tcp_EB );

   for (int tcc=nrMinTccEB_ ; tcc<=nrMaxTccEB_ ; tcc++) {
    for (int tower=1 ; tower<=nbMaxTowers_ ; tower++) {
       std::vector<unsigned int> * lut;
       lut= const_cast<std::vector<unsigned int> *> (getTowerParameters(tcc, tower));
       for (unsigned int i=0 ; i<1024 ; i++) {
	int ttf = 0 ;
	if (i>=ttfHighEB_ADC) ttf = 3 ; 
	if (i>=ttfLowEB_ADC && i<ttfHighEB_ADC) ttf = 1 ;
	ttf = ttf<<8 ; 
	(*lut)[i] = ((*lut)[i] & 0xff) + ttf ;
      }
      setTowerParameters(tcc, tower, (*lut)) ;
    } 
  }
}
