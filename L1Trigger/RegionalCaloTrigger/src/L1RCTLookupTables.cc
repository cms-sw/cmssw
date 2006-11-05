#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <fstream>
#include <string>
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
using std::cout;
using std::endl;

// constructor

L1RCTLookupTables::L1RCTLookupTables(const std::string& filename)
{
  loadHcalConstants(filename);
}

// lookup method for HF
unsigned short L1RCTLookupTables::lookup(unsigned short hfenergy,
					 unsigned short crtNo,
					 unsigned short crdNo, 
					 unsigned short twrNo){
  if(hfenergy > 0xFF) throw cms::Exception("Invalid Data") << "HF compressedET should be less than 0xFF, is " << hfenergy;
  short iEta = L1RCT::calcIEta(crtNo, crdNo, twrNo);
  unsigned short iAbsEta = abs(iEta);
  if(iAbsEta < 29 || iAbsEta > 32) throw cms::Exception("Invalid Data") << "29 <= |iEta| <= 32, is " << iAbsEta;
  float energy = hcalConversionConstants_.at(iAbsEta).at(hfenergy);
  return convertToInteger(energy, jetMETLSB(), 8);
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
  short iEta = L1RCT::calcIEta(crtNo, crdNo, twrNo);
  unsigned short iAbsEta = abs(iEta);
  if(iAbsEta < 1 || iAbsEta > 28) throw cms::Exception("Invalid Data") << "1 <= |IEta| <= 28, is " << iAbsEta;
  float ecalLinear = convertEcal(ecal);
  float hcalLinear = hcalConversionConstants_.at(iAbsEta).at(hcal);
  float etLinear = ecalLinear + hcalLinear;
  unsigned long HE_FGBit = (calcHEBit(ecalLinear,hcalLinear) || fgbit);
  unsigned long etIn7Bits = convertToInteger(etLinear, eGammaLSB(), 7);
  unsigned long etIn9Bits = convertToInteger(etLinear, jetMETLSB(), 9);
  unsigned long activityBit = calcActivityBit(ecal,hcal);
  unsigned long shiftEtIn9Bits = etIn9Bits<<8;
  unsigned long shiftHE_FGBit = HE_FGBit<<7;
  unsigned long shiftActivityBit = activityBit<<17;
  unsigned long output=etIn7Bits+shiftHE_FGBit+shiftEtIn9Bits+shiftActivityBit;
  return output;
}

// converts compressed ecal energy to linear (real) scale
float L1RCTLookupTables::convertEcal(unsigned short ecal){
  return ((float) ecal) * eGammaLSB();
}

// calculates activity bit for each tower - assume that noise is well suppressed
unsigned short L1RCTLookupTables::calcActivityBit(unsigned short ecal, unsigned short hcal){
  return ((ecal > 2) || (hcal > 2));
}

// calculates h-over-e veto bit (true if hcal/ecal energy > 5%)
unsigned short L1RCTLookupTables::calcHEBit(float ecal, float hcal){
  return ((hcal/ecal)>0.05);
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

void L1RCTLookupTables::loadHcalConstants(const std::string& filename)
{
  std::ifstream userfile;
  userfile.open(filename.c_str());
  if( userfile )
    {
      char junk[256];
      userfile >> junk;
      userfile >> junk;
      userfile >> junk;
      userfile >> junk;
      userfile >> junk;
      hcalConversionConstants_.resize(N_ET_CONSTS);
      for(int iAbsEta = 1; iAbsEta <= N_TOWERS; iAbsEta++) {
	hcalConversionConstants_.at(iAbsEta).resize(N_ET_CONSTS);
      }
      for(int hcalETAddress = 0; hcalETAddress < N_ET_CONSTS; hcalETAddress++) {
	for(int iAbsEta = 1; iAbsEta <= N_TOWERS; iAbsEta++) {
	  float value;
	  userfile >> value;
	  hcalConversionConstants_.at(iAbsEta-1).at(hcalETAddress) = value;
	}
      }
      userfile.close();
    }
  else 
    {
      throw cms::Exception("Invalid Data") << "Unable to open " << filename;
    }
}
