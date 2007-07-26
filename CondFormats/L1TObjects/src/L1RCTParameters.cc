/**
 * Author: Sridhara Dasu
 * Created: 04 July 2007
 * $Id: L1RCTParameters.cc,v 1.4 2007/07/18 08:03:16 dasu Exp $
 **/

#include <iostream>
#include <fstream>

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

L1RCTParameters::L1RCTParameters(double eGammaLSB,
				 double jetMETLSB,
				 double eMinForFGCut,
				 double eMaxForFGCut,
				 double hOeCut,
				 double eMinForHoECut,
				 double eMaxForHoECut,
				 double eActivityCut,
				 double hActivityCut,
				 std::vector<double> eGammaECalScaleFactors,
				 std::vector<double> eGammaHCalScaleFactors,
				 std::vector<double> jetMETECalScaleFactors,
				 std::vector<double> jetMETHCalScaleFactors
				 ) :
  eGammaLSB_(eGammaLSB),
  jetMETLSB_(jetMETLSB),
  eMinForFGCut_(eMinForFGCut),
  eMaxForFGCut_(eMaxForFGCut),
  hOeCut_(hOeCut),
  eMinForHoECut_(eMinForHoECut),
  eMaxForHoECut_(eMaxForHoECut),
  eActivityCut_(eActivityCut),
  hActivityCut_(hActivityCut),
  eGammaECalScaleFactors_(eGammaECalScaleFactors),
  eGammaHCalScaleFactors_(eGammaHCalScaleFactors),
  jetMETECalScaleFactors_(jetMETECalScaleFactors),
  jetMETHCalScaleFactors_(jetMETHCalScaleFactors),
  transcoder_(0),
  l1CaloEtScale_(0)
{
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
    // assume iphi between 0 and 71; makes towers from 1-32
    tower = ((absIeta-1)%8)*4 + (iphi%4) + 1;
  }
  // 25 <= absIeta <= 28 (card 6)
  else if ((absIeta >= 25) && (absIeta <= 28)){
    if (regionPhi == 0){
      // towers from 1-32, modified Aug. 1 Jessica Leonard
      tower = (absIeta-25)*4 + (iphi%4) + 1;
    }
    else {
      tower = 29 + iphi % 4 + (25 - absIeta) * 4;
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

short L1RCTParameters::calcIEta(unsigned short iCrate, unsigned short iCard, 
				unsigned short iTower) const
{
  unsigned short absIEta;
  if(iCard < 6) 
    absIEta = (iCard / 2) * 8 + ((iTower - 1) / 4) + 1;
  else if(iCard == 6) {
    if(iTower < 17)
      absIEta = 25 + (iTower - 1) / 4;
    else
      absIEta = 28 - ((iTower - 17) / 4);
  }
  else
    absIEta = 29 + iTower % 4;
  short iEta;
  if(iCrate < 9) iEta = -absIEta;
  else iEta = absIEta;
  return iEta;
}

unsigned short L1RCTParameters::calcIPhi(unsigned short iCrate, 
					 unsigned short iCard, 
					 unsigned short iTower) const
{
  short iPhi;
  if(iCard < 6)
    iPhi = (iCrate % 9) * 8 + (iCard % 2) * 4 + ((iTower - 1) % 4);
  else if(iCard == 6){
    if(iTower < 17)
      iPhi = (iCrate % 9) * 8 + ((iTower - 1) % 4);
    else
      iPhi = (iCrate % 9) * 8 + ((iTower - 17) % 4) + 4;
  }
  else
    iPhi = (iCrate % 9) * 2 + iTower / 4;
  return iPhi;
}

// converts compressed ecal energy to linear (real) scale
float L1RCTParameters::convertEcal(unsigned short ecal, int iAbsEta) const
{
  return ((float) ecal) * eGammaLSB_;
}

// converts compressed hcal energy to linear (real) scale
float L1RCTParameters::convertHcal(unsigned short hcal, int iAbsEta) const
{
  static bool first = true;
  if(transcoder_ != 0)
    {
      if(first)
	std::cout << "L1RCTParameters: Using transcoder" << std::endl;
      return (transcoder_->hcaletValue(iAbsEta, hcal));
    }
  else
    {
      if(first)
	std::cout << "L1RCTParameters: Not using transcoder" << std::endl;
      return ((float) hcal) * jetMETLSB_;
    }
  first = false;
}

// integerize given an LSB and set maximum value of 2^precision-1
unsigned long L1RCTParameters::convertToInteger(float et, 
						float lsb, 
						int precision) const
{
  unsigned long etBits = (unsigned long)(et/lsb);
  unsigned long maxValue = (1 << precision) - 1;
  if(etBits > maxValue)
    return maxValue;
  else
    return etBits;
}

unsigned int L1RCTParameters::eGammaETCode(float ecal, float hcal, int iAbsEta) const
{
  float etLinear = 
    eGammaECalScaleFactors_[iAbsEta] * ecal +
    eGammaHCalScaleFactors_[iAbsEta] * hcal;
  return convertToInteger(etLinear, eGammaLSB_, 7);
}

unsigned int L1RCTParameters::jetMETETCode(float ecal, float hcal, int iAbsEta) const
{
  float etLinear = 
    jetMETECalScaleFactors_[iAbsEta] * ecal +
    jetMETHCalScaleFactors_[iAbsEta] * hcal;
  return convertToInteger(etLinear, jetMETLSB_, 9);
}

unsigned int L1RCTParameters::lookup(unsigned short ecalInput,
				     unsigned short hcalInput,
				     unsigned short fgbit,
				     unsigned short crtNo,
				     unsigned short crdNo,
				     unsigned short twrNo) const
{
  if(ecalInput > 0xFF) 
    throw cms::Exception("Invalid Data") 
      << "ECAL compressedET should be less than 0xFF, is " << ecalInput;
  if(hcalInput > 0xFF) 
    throw cms::Exception("Invalid Data") 
      << "HCAL compressedET should be less than 0xFF, is " << hcalInput;
  if(fgbit > 1) 
    throw cms::Exception("Invalid Data") 
      << "ECAL finegrain should be a single bit, is " << fgbit;
  int iEta = calcIEta(crtNo, crdNo, twrNo);
  int iAbsEta = abs(iEta);
  if(iAbsEta < 1 || iAbsEta > 28) 
    throw cms::Exception("Invalid Data") 
      << "1 <= |IEta| <= 28, is " << iAbsEta;
  float ecal = convertEcal(ecalInput, iAbsEta);
  float hcal = convertHcal(hcalInput, iAbsEta);
  unsigned long etIn7Bits = eGammaETCode(ecal, hcal, iAbsEta);
  unsigned long etIn9Bits = jetMETETCode(ecal, hcal, iAbsEta);
  unsigned long shiftEtIn9Bits = etIn9Bits<<8;
  unsigned long shiftHE_FGBit = hOeFGVetoBit(ecal, hcal, fgbit)<<7;
  unsigned long shiftActivityBit = activityBit(ecal, hcal)<<17;
  unsigned long output=etIn7Bits+shiftHE_FGBit+shiftEtIn9Bits+shiftActivityBit;
  return output;
}

unsigned int L1RCTParameters::lookup(unsigned short hfInput,
				     unsigned short crtNo,
				     unsigned short crdNo,
				     unsigned short twrNo
				     ) const
{
  if(hfInput > 0xFF) 
    throw cms::Exception("Invalid Data") 
      << "HF compressedET should be less than 0xFF, is " << hfInput;
  int iEta = calcIEta(crtNo, crdNo, twrNo);
  int iAbsEta = abs(iEta);
  if(iAbsEta < 29 || iAbsEta > 32) 
    throw cms::Exception("Invalid Data") 
      << "29 <= |iEta| <= 32, is " << iAbsEta;
  float et = convertHcal(hfInput, iAbsEta);
  return convertToInteger(et, jetMETLSB(), 8);
}

bool L1RCTParameters::hOeFGVetoBit(float ecal, float hcal, bool fgbit) const
{
  bool veto = false;
  if(ecal > eMinForFGCut_ && ecal < eMaxForFGCut_)
    {
      if(fgbit) veto = true;
    }
  if(ecal > eMinForHoECut_ && ecal < eMaxForHoECut_)
    {
      if((hcal / ecal) > hOeCut_) veto = true;
    }
  return veto;
}

bool L1RCTParameters::activityBit(float ecal, float hcal) const
{
  return ((ecal > eActivityCut_) || (hcal > hActivityCut_));
}
