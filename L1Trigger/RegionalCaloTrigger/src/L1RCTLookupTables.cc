#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
#include <string>

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

// public variable initialization - these are typical values -- they are reset to values stored in the LUTFile

float L1RCTLookupTables::eActivityCut_ = 3.0;
float L1RCTLookupTables::hActivityCut_ = 3.0;
float L1RCTLookupTables::hOeCut_ = 0.05;
float L1RCTLookupTables::eGammaLSB_ = 0.5;
float L1RCTLookupTables::jetMETLSB_ = 0.5;

// constructor

L1RCTLookupTables::L1RCTLookupTables(const std::string& filename)
{
  loadLUTConstants(filename);
  useTranscoder_ = false;
}

L1RCTLookupTables::L1RCTLookupTables(const std::string& filename, edm::ESHandle<CaloTPGTranscoder> transcoder)
{
  loadLUTConstants(filename);
  transcoder_ = transcoder;
  useTranscoder_ = true;
}

// lookup method for HF
unsigned short L1RCTLookupTables::lookup(unsigned short hfenergy,
					 unsigned short crtNo,
					 unsigned short crdNo, 
					 unsigned short twrNo){
  if(hfenergy > 0xFF) throw cms::Exception("Invalid Data") << "HF compressedET should be less than 0xFF, is " << hfenergy;
  int iEta = L1RCT::calcIEta(crtNo, crdNo, twrNo);
  int iAbsEta = abs(iEta);
  if(iAbsEta < 29 || iAbsEta > 32) throw cms::Exception("Invalid Data") << "29 <= |iEta| <= 32, is " << iAbsEta;
  float et;
  if(useTranscoder_) et = transcoder_->hcaletValue(iAbsEta, hfenergy);
  else et = hfenergy;           // This is so debugging can happen without the transcoder
  return convertToInteger(et, jetMETLSB(), 8);
}

// lookup method for barrel (ecal and hcal)
unsigned long L1RCTLookupTables::lookup(unsigned short ecal,unsigned short hcal,
					unsigned short fgbit,
					unsigned short crtNo, 
					unsigned short crdNo, 
					unsigned short twrNo){
  if(ecal > 0xFF) throw cms::Exception("Invalid Data") << "ECAL compressedET should be less than 0xFF, is " << ecal;
  if(hcal > 0xFF) throw cms::Exception("Invalid Data") << "HCAL compressedET should be less than 0xFF, is " << hcal;
  if(fgbit > 1) throw cms::Exception("Invalid Data") << "ECAL finegrain should be a single bit, is " << fgbit;
  int iEta = L1RCT::calcIEta(crtNo, crdNo, twrNo);
  int iAbsEta = abs(iEta);
  if(iAbsEta < 1 || iAbsEta > 28) throw cms::Exception("Invalid Data") << "1 <= |IEta| <= 28, is " << iAbsEta;
  float ecalLinear = convertEcal(ecal);
  float hcalLinear;
  float hcalELinear;
  if(useTranscoder_) hcalLinear = transcoder_->hcaletValue(iAbsEta, hcal);
  else hcalLinear = hcal;
  
  float etLinear = ecalLinear + hcalLinear;
  unsigned long HE_FGBit = (calcHEBit(ecalLinear,hcalLinear) || fgbit);
  unsigned long etIn7Bits = convertToInteger(ecalLinear, eGammaLSB_, 7);
  unsigned long etIn9Bits = convertToInteger(etLinear, jetMETLSB_, 9);
  unsigned long activityBit = calcActivityBit(ecalLinear, hcalLinear);
  unsigned long shiftEtIn9Bits = etIn9Bits<<8;
  unsigned long shiftHE_FGBit = HE_FGBit<<7;
  unsigned long shiftActivityBit = activityBit<<17;
  unsigned long output=etIn7Bits+shiftHE_FGBit+shiftEtIn9Bits+shiftActivityBit;
  return output;
}

// converts compressed ecal energy to linear (real) scale
float L1RCTLookupTables::convertEcal(unsigned short ecal){
  return ((float) ecal) * eGammaLSB_;
}

// calculates activity bit for each tower - assume that noise is well suppressed
unsigned short L1RCTLookupTables::calcActivityBit(float ecal, float hcal){
  return ((ecal > eActivityCut_) || (hcal > hActivityCut_));
}

// calculates h-over-e veto bit (true if hcal/ecal energy > 5%)
unsigned short L1RCTLookupTables::calcHEBit(float ecal, float hcal){
  if((ecal > eActivityCut_) && (hcal/ecal)>hOeCut_)
    return true;
  else
    return false;
}

// integerize given an LSB and set maximum value of 2^precision
unsigned long L1RCTLookupTables::convertToInteger(float et, float lsb, int precision){
  unsigned long etBits = (unsigned long)(et/lsb);
  unsigned long maxValue = (unsigned long) pow(2,precision)-1;
  if(etBits > maxValue)
    return maxValue;
  else
    return etBits;
}

void L1RCTLookupTables::loadLUTConstants(const std::string& filename)
{
  std::ifstream userfile;
  userfile.open(filename.c_str());
  if( userfile )
    {
      char junk[1024];
      userfile >> junk >> eActivityCut_;
      std::cout << "L1RCTLookupTables: Using eActivityCut = " 
		<< eActivityCut_ << std::endl;
      userfile >> junk >> hActivityCut_;
      std::cout << "L1RCTLookupTables: Using hActivityCut = " 
		<< hActivityCut_ << std::endl;
      userfile >> junk >> hOeCut_;
      std::cout << "L1RCTLookupTables: Using hOeCut = " 
		<< hOeCut_ << std::endl;
      userfile >> junk >> eGammaLSB_;
      std::cout << "L1RCTLookupTables: Using eGammaLSB = " 
		<< eGammaLSB_ << std::endl;
      userfile >> junk >> jetMETLSB_;
      std::cout << "L1RCTLookupTables: Using jetMETLSB = " 
		<< jetMETLSB_ << std::endl;
      userfile.close();
    }
  else 
    {
      throw cms::Exception("Invalid Data") << "Unable to open " << filename;
    }
}
