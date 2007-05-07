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
float L1RCTLookupTables::eMaxForFGCut_ = 50.;
float L1RCTLookupTables::hOeCut_ = 0.05;
float L1RCTLookupTables::eGammaLSB_ = 0.5;
float L1RCTLookupTables::jetMETLSB_ = 0.5;
float L1RCTLookupTables::eGammaSCF_[32] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
float L1RCTLookupTables::hcalSCF_[32] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

bool patternTest_ = false;
bool ignoreFG_ = false;

// constructor

L1RCTLookupTables::L1RCTLookupTables(const std::string& filename)
{
  loadLUTConstants(filename);
  useTranscoder_ = false;
}

L1RCTLookupTables::L1RCTLookupTables(const std::string& filename, const std::string& filename2, bool patternTest)
{
  loadLUTConstants(filename);
  loadHcalLut(filename2);
  patternTest_ = patternTest;
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
  //  float et;
  //  if(useTranscoder_) et = transcoder_->hcaletValue(iAbsEta, hfenergy);
  //  else et = hfenergy;           // This is so debugging can happen without the transcoder
  float et = convertHcal(hfenergy, iAbsEta);
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
  float ecalLinear = convertEcal(ecal, iAbsEta);
  float hcalLinear = convertHcal(hcal, iAbsEta);
  
  float etLinear = ecalLinear + hcalLinear;
  unsigned long HE_FGBit;
  if(patternTest_)
    {
      if(ignoreFG_)
	{
	  HE_FGBit = calcHEBit(ecalLinear,hcalLinear,false);
	  //cout << "L1RCT: fine grain bit ignored!" << endl;
	}
      else
	{
	  HE_FGBit = calcHEBit(ecalLinear,hcalLinear,fgbit);
	}
    }
  else
    {
      HE_FGBit = calcHEBit(ecalLinear,hcalLinear, fgbit);
      if(ecal == 0xFF) HE_FGBit = 0; // For saturated towers ignore H/E & FG veto
    }
  unsigned long etIn7Bits = convertToInteger(etLinear, eGammaLSB_, 7); // changed from ecalLinear
  unsigned long etIn9Bits = convertToInteger(etLinear, jetMETLSB_, 9);
  unsigned long activityBit = calcActivityBit(ecalLinear, hcalLinear);
  unsigned long shiftEtIn9Bits = etIn9Bits<<8;
  unsigned long shiftHE_FGBit = HE_FGBit<<7;
  unsigned long shiftActivityBit = activityBit<<17;
  unsigned long output=etIn7Bits+shiftHE_FGBit+shiftEtIn9Bits+shiftActivityBit;
  return output;
}

// converts compressed ecal energy to linear (real) scale
float L1RCTLookupTables::convertEcal(unsigned short ecal, int iAbsEta){
  return ((float) ecal) * eGammaLSB_ * eGammaSCF_[iAbsEta];
}

// converts compressed hcal energy to linear (real) scale
float L1RCTLookupTables::convertHcal(unsigned short hcal, int iAbsEta){
  if(useTranscoder_) 
    {
      return (transcoder_->hcaletValue(iAbsEta, hcal));
    }
  else 
    {
      return ((float) hcal) * jetMETLSB_ * hcalSCF_[iAbsEta];
    }
}

// calculates activity bit for each tower - assume that noise is well suppressed
unsigned short L1RCTLookupTables::calcActivityBit(float ecal, float hcal){
  return ((ecal > eActivityCut_) || (hcal > hActivityCut_));
}

// Calculates h-over-e veto bit (true if hcal/ecal energy > hOeCut)
// Uses finegrain veto only if the energy is within eActivityCut and eMaxForFGCut
unsigned short L1RCTLookupTables::calcHEBit(float ecal, float hcal, bool fgbit){
  bool veto = false;
  if(ecal > eMaxForFGCut_)
    {
      if((hcal/ecal) > hOeCut_) veto = true;
    }
  else if(ecal > eActivityCut_)
    {
      if((hcal/ecal) > hOeCut_) veto = true;
      if(fgbit) veto = true;
    }
  return veto;
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
      int answer;
      userfile >> junk >> eActivityCut_;
      std::cout << "L1RCTLookupTables: Using eActivityCut = " 
		<< eActivityCut_ << std::endl;
      userfile >> junk >> hActivityCut_;
      std::cout << "L1RCTLookupTables: Using hActivityCut = " 
		<< hActivityCut_ << std::endl;
      userfile >> junk >> eMaxForFGCut_;
      std::cout << "L1RCTLookupTables: Using eMaxForFGCut = " 
		<< eMaxForFGCut_ << std::endl;
      userfile >> junk >> hOeCut_;
      std::cout << "L1RCTLookupTables: Using hOeCut = " 
		<< hOeCut_ << std::endl;
      userfile >> junk >> eGammaLSB_;
      std::cout << "L1RCTLookupTables: Using eGammaLSB = " 
		<< eGammaLSB_ << std::endl;
      userfile >> junk >> jetMETLSB_;
      std::cout << "L1RCTLookupTables: Using jetMETLSB = " 
		<< jetMETLSB_ << std::endl;
      userfile >> junk >> answer;
      //      userfile.getline(junk,199);
      //      std::cout << "junk is " << junk << endl;
      if(answer == 1)
	{
	  ignoreFG_ = true;
	}
      else if(answer == 0)
	{
	  ignoreFG_ = false;
	}
      else
	{
	  std::cout << "L1RCTLookupTables: ignoreFineGrain not true or false!" << std::endl; 
	  //std::cout << "variable 'answer' is " << answer << std::endl;
	}
      std:: cout << "L1RCTLookupTables: ignoreFineGrain is "
		 << ignoreFG_ << std::endl;
      userfile >> junk;
      for(int i = 0; i < 26; i++) 
	{
	  userfile >> eGammaSCF_[i];
	  //std::cout << "L1RCTLookupTables: eGammaSCF_[" << i << "] is " << eGammaSCF_[i] << endl;
	}
      for(int i = 26; i < 32; i++) eGammaSCF_[i] = eGammaSCF_[i-1];
      userfile.close();
    }
  else 
    {
      throw cms::Exception("Invalid Data") << "Unable to open " << filename;
    }
}

void L1RCTLookupTables::loadHcalLut(const std::string& filename)
{
  std::ifstream userfile;
  userfile.open(filename.c_str());
  if ( userfile )
    {
      char junk[1024];
      userfile >> junk;
      for (int i = 0; i < 26; i++) 
	{
	  userfile >> hcalSCF_[i];
	  //std::cout << "L1RCTLookupTables: hcalSCF_[" << i << "] is " << hcalSCF_[i] << std::endl;
	}
      for (int i = 26; i < 32; i++) 
	{
	  hcalSCF_[i] = hcalSCF_[i-1];
	  //std::cout << "L1RCTLookupTables: hcalSCF_[" << i << "] is " << hcalSCF_[i] << std::endl;
	}
      userfile.close();
    }
  else
    {
      throw cms::Exception("Invalid Data") << "Unable to open " << filename;
    }
}
