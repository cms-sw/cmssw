/**
 * Author: Sridhara Dasu
 * Created: 04 July 2007
 * $Id: L1RCTParameters.cc,v 1.20 2008/10/09 10:42:26 jleonard Exp $
 **/

#include <iostream>
#include <fstream>
#include <cmath>

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

L1RCTParameters::L1RCTParameters(double eGammaLSB,
				 double jetMETLSB,
				 double eMinForFGCut,
				 double eMaxForFGCut,
				 double hOeCut,
				 double eMinForHoECut,
				 double eMaxForHoECut,
				 double hMinForHoECut,
				 double eActivityCut,
				 double hActivityCut,
				 unsigned eicIsolationThreshold,
				 unsigned jscQuietThresholdBarrel,
				 unsigned jscQuietThresholdEndcap,
				 bool noiseVetoHB,
				 bool noiseVetoHEplus,
				 bool noiseVetoHEminus,
				 bool useLindsey,
				 const std::vector<double>& eGammaECalScaleFactors,
				 const std::vector<double>& eGammaHCalScaleFactors,
				 const std::vector<double>& jetMETECalScaleFactors,
				 const std::vector<double>& jetMETHCalScaleFactors,
				 const std::vector<double>& ecal_calib_Lindsey,
				 const std::vector<double>& hcal_calib_Lindsey,
				 const std::vector<double>& hcal_high_calib_Lindsey,
				 const std::vector<double>& cross_terms_Lindsey,
				 const std::vector<double>& lowHoverE_smear,
				 const std::vector<double>& highHoverE_smear
				 ) :
  eGammaLSB_(eGammaLSB),
  jetMETLSB_(jetMETLSB),
  eMinForFGCut_(eMinForFGCut),
  eMaxForFGCut_(eMaxForFGCut),
  hOeCut_(hOeCut),
  eMinForHoECut_(eMinForHoECut),
  eMaxForHoECut_(eMaxForHoECut),
  hMinForHoECut_(hMinForHoECut),
  eActivityCut_(eActivityCut),
  hActivityCut_(hActivityCut),
  eicIsolationThreshold_(eicIsolationThreshold),
  jscQuietThresholdBarrel_(jscQuietThresholdBarrel),
  jscQuietThresholdEndcap_(jscQuietThresholdEndcap),
  noiseVetoHB_(noiseVetoHB),
  noiseVetoHEplus_(noiseVetoHEplus),
  noiseVetoHEminus_(noiseVetoHEminus),
  useCorrectionsLindsey(useLindsey),
  eGammaECalScaleFactors_(eGammaECalScaleFactors),
  eGammaHCalScaleFactors_(eGammaHCalScaleFactors),
  jetMETECalScaleFactors_(jetMETECalScaleFactors),
  jetMETHCalScaleFactors_(jetMETHCalScaleFactors),
  HoverE_smear_low_Lindsey_(lowHoverE_smear),
  HoverE_smear_high_Lindsey_(highHoverE_smear)
{
  ecal_calib_Lindsey_.resize(28);
  hcal_calib_Lindsey_.resize(28);
  hcal_high_calib_Lindsey_.resize(28);
  cross_terms_Lindsey_.resize(28);

  for(unsigned i = 0; i < ecal_calib_Lindsey.size(); ++i)
    ecal_calib_Lindsey_[i/3].push_back(ecal_calib_Lindsey[i]);
  for(unsigned i = 0; i < hcal_calib_Lindsey.size(); ++i)
    hcal_calib_Lindsey_[i/3].push_back(hcal_calib_Lindsey[i]);
  for(unsigned i = 0; i < hcal_high_calib_Lindsey.size(); ++i)
    hcal_high_calib_Lindsey_[i/3].push_back(hcal_high_calib_Lindsey[i]);
  for(unsigned i = 0; i < cross_terms_Lindsey.size(); ++i)
    cross_terms_Lindsey_[i/6].push_back(cross_terms_Lindsey[i]);
}

// maps rct iphi, ieta of tower to crate
unsigned short L1RCTParameters::calcCrate(unsigned short rct_iphi, short ieta) const
{
  unsigned short crate = rct_iphi/8;
  if(abs(ieta) > 28) crate = rct_iphi / 2;
  if (ieta > 0){
    crate = crate + 9;
  }
  return crate;
}

//map digi rct iphi, ieta to card
unsigned short L1RCTParameters::calcCard(unsigned short rct_iphi, 
					 unsigned short absIeta) const
{
  unsigned short card = 999;
  // Note absIeta counts from 1-32 (not 0-31)
  if (absIeta <= 24){
    card =  ((absIeta-1)/8)*2 + (rct_iphi%8)/4;
  }
  // 25 <= absIeta <= 28 (card 6)
  else if ((absIeta >= 25) && (absIeta <= 28)){
    card = 6;
  }
  else{}
  return card;
}

//map digi rct iphi, ieta to tower
unsigned short L1RCTParameters::calcTower(unsigned short rct_iphi, 
					  unsigned short absIeta) const
{
  unsigned short tower = 999;
  unsigned short iphi = rct_iphi;
  unsigned short regionPhi = (iphi % 8)/4;

  // Note absIeta counts from 1-32 (not 0-31)
  if (absIeta <= 24){
    // assume iphi between 0 and 71; makes towers from 0-31, mod. 7Nov07
    tower = ((absIeta-1)%8)*4 + (iphi%4);  // REMOVED +1
  }
  // 25 <= absIeta <= 28 (card 6)
  else if ((absIeta >= 25) && (absIeta <= 28)){
    if (regionPhi == 0){
      // towers from 0-31, modified 7Nov07 Jessica Leonard
      tower = (absIeta-25)*4 + (iphi%4);  // REMOVED +1
    }
    else {
      tower = 28 + iphi % 4 + (25 - absIeta) * 4;  // mod. 7Nov07 JLL
    }
  }
  // absIeta >= 29 (HF regions)
  else if ((absIeta >= 29) && (absIeta <= 32)){
    // SPECIAL DEFINITION OF REGIONPHI FOR HF SINCE HF IPHI IS 0-17 
    // Sept. 19 J. Leonard
    regionPhi = iphi % 2;
    // HF MAPPING, just regions now, don't need to worry about towers
    // just calling it "tower" for convenience
    tower = (regionPhi) * 4 + absIeta - 29;
  }
  return tower;
}

// iCrate 0-17, iCard 0-6, NEW iTower 0-31
short L1RCTParameters::calcIEta(unsigned short iCrate, unsigned short iCard, 
				unsigned short iTower) const
{
  unsigned short absIEta = calcIAbsEta(iCrate, iCard, iTower);
  short iEta;
  if(iCrate < 9) iEta = -absIEta;
  else iEta = absIEta;
  return iEta;
}

// iCrate 0-17, iCard 0-6, NEW iTower 0-31
unsigned short L1RCTParameters::calcIPhi(unsigned short iCrate, 
					 unsigned short iCard, 
					 unsigned short iTower) const
{
  short iPhi;
  if(iCard < 6)
    iPhi = (iCrate % 9) * 8 + (iCard % 2) * 4 + (iTower % 4); // rm -1 7Nov07
  else if(iCard == 6){
    // region 0
    if(iTower < 16)  // 17->16
      iPhi = (iCrate % 9) * 8 + (iTower % 4);  // rm -1
    // region 1
    else
      iPhi = (iCrate % 9) * 8 + ((iTower - 16) % 4) + 4; // 17 -> 16
  }
  // HF regions
  else
    iPhi = (iCrate % 9) * 2 + iTower / 4;
  return iPhi;
}

// iCrate 0-17, iCard 0-6, NEW iTower 0-31
unsigned short L1RCTParameters::calcIAbsEta(unsigned short iCrate, unsigned short iCard, 
					    unsigned short iTower) const
{
  unsigned short absIEta;
  if(iCard < 6) 
    absIEta = (iCard / 2) * 8 + (iTower / 4) + 1;  // rm -1 JLL 7Nov07
  else if(iCard == 6) {
    // card 6, region 0
    if(iTower < 16) // 17->16
      absIEta = 25 + iTower / 4;  // rm -1
    // card 6, region 1
    else
      absIEta = 28 - ((iTower - 16) / 4);  // 17->16
  }
  // HF regions
  else
    absIEta = 29 + iTower % 4;
  return absIEta;
}

float L1RCTParameters::JetMETTPGSum(const float& ecal, const float& hcal, const unsigned& iAbsEta) const
{
  float ecal_c = ecal*jetMETECalScaleFactors_.at(iAbsEta-1);
  float hcal_c = hcal*jetMETHCalScaleFactors_.at(iAbsEta-1);
  float result = ecal_c + hcal_c;

  if(useCorrectionsLindsey)
    {
      // In my corrections I expect eGamma corrected ecal readings.
      ecal_c *= eGammaECalScaleFactors_.at(iAbsEta-1)/jetMETECalScaleFactors_.at(iAbsEta-1);
      result = correctedTPGSum_Lindsey(ecal_c,hcal,iAbsEta-1); // I apply my own HCAL correction.
    }

  return result;
}

float L1RCTParameters::EGammaTPGSum(const float& ecal, const float& hcal, const unsigned& iAbsEta) const
{
  float ecal_c = ecal*eGammaECalScaleFactors_.at(iAbsEta-1);
  float hcal_c = hcal*eGammaHCalScaleFactors_.at(iAbsEta-1);
  float result = ecal_c + hcal_c;

  if(useCorrectionsLindsey)
    {
      result = correctedTPGSum_Lindsey(ecal_c,hcal,iAbsEta-1); // I apply my own HCAL correction.
    }

  return result;
}

// index = iAbsEta - 1... make sure you call the function like so: "correctedTPGSum_Lindsey(ecal,hcal, iAbsEta - 1)"
float L1RCTParameters::correctedTPGSum_Lindsey(const float& ecal, const float& hcal, const unsigned& index) const
{
  if(index >= 28 && ecal > 120 && hcal > 120) return (ecal + hcal); // return plain sum if outside of calibration range or index is too high

  // let's make sure we're asking for items that are there.
  if(ecal_calib_Lindsey_.at(index).size() != 3 || hcal_calib_Lindsey_.at(index).size() != 3 ||
     hcal_high_calib_Lindsey_.at(index).size() != 3 || cross_terms_Lindsey_.at(index).size() != 6 ||
     HoverE_smear_high_Lindsey_.size() <= index || HoverE_smear_low_Lindsey_.size() <= index)
    return (ecal+hcal);

  double e = ecal, h = hcal;
  double ec = 0.0, hc = 0.0, c = 0.0;
  
  ec = (ecal_calib_Lindsey_.at(index).at(0)*std::pow(e,3.) +
	ecal_calib_Lindsey_.at(index).at(1)*std::pow(e,2.) +
	ecal_calib_Lindsey_.at(index).at(2)*e);
  
  if(e + h < 23)
    {
      hc = (hcal_calib_Lindsey_.at(index).at(0)*std::pow(h,3.) + 
	    hcal_calib_Lindsey_.at(index).at(1)*std::pow(h,2.) + 
	    hcal_calib_Lindsey_.at(index).at(2)*h);
      
      c = (cross_terms_Lindsey_.at(index).at(0)*std::pow(e, 2.)*h +
	   cross_terms_Lindsey_.at(index).at(1)*std::pow(h, 2.)*e +
	   cross_terms_Lindsey_.at(index).at(2)*e*h +
	   cross_terms_Lindsey_.at(index).at(3)*std::pow(e, 3.)*h +
	   cross_terms_Lindsey_.at(index).at(4)*std::pow(h, 3.)*e +
	   cross_terms_Lindsey_.at(index).at(5)*std::pow(h, 2.)*std::pow(e, 2.));
    }
  else
    {
      hc = (hcal_high_calib_Lindsey_.at(index).at(0)*std::pow(h,3.) + 
	    hcal_high_calib_Lindsey_.at(index).at(1)*std::pow(h,2.) + 
	    hcal_high_calib_Lindsey_.at(index).at(2)*h);
    }
  
  if(e == 0.0) e += 0.000000001;
  
  if(h/(e+h) >= 0.05)
    {
      ec *= HoverE_smear_high_Lindsey_.at(index);
      hc *= HoverE_smear_high_Lindsey_.at(index);
      c *= HoverE_smear_high_Lindsey_.at(index);
    }
  else
    {
      ec *= HoverE_smear_low_Lindsey_.at(index);
    }
  return ec + hc + c;
}
 
