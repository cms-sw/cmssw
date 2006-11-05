#ifndef L1RCTLookupTables_h
#define L1RCTLookupTables_h

#include <math.h>
#include <iostream>
#include <string>
#include <vector>

class L1RCTLookupTables {
 
 public:

  // constructor

  L1RCTLookupTables(const std::string& filename);

  // function needs to output an unsigned long

  unsigned long lookup(unsigned short ecal, 
		       unsigned short hcal,
		       unsigned short fgbit, 
		       unsigned short crtNo, 
		       unsigned short crdNo, 
		       unsigned short twrNo
		       );
  unsigned short lookup(unsigned short hfenergy,
			unsigned short crtNo, 
			unsigned short crdNo, 
			unsigned short twrNo
			);

  // LSBs in which RCT operates -- these should be tunable, but are left hardcoded temporarily

  static float eGammaLSB() {return 0.5;}
  static float jetMETLSB() {return 0.5;}

 private:

  L1RCTLookupTables();  // Do not implement so one cannot instantiate without input file

  float convertEcal(unsigned short ecal);
  unsigned short calcActivityBit(unsigned short ecal,unsigned short hcal);
  unsigned short calcHEBit(float ecal,float hcal);
  unsigned long convertToInteger(float et, float lsb, int precision);

  // We implement LUTs in code -- real lookup tables are large 2^17 (addresses) x 18 phi bins x 18 bits
  // So we only read in conversion constants in float given by TPG developers, and implement LUTs on-the-fly
  // This same code can be used to write out real LUTs and load in the hardware
  // The read in conversion constants should come from the database, but for the moment we use a flat file
  void loadHcalConstants(const std::string& filename);
  static const int N_TOWERS = 32;     // Number of |eta| towers - reflect to -eta and rotate for all phi
  static const int N_ET_CONSTS = 256; // Corresponding to 8-bits of ET for ECAL and HCAL
  std::vector<std::vector<float> > hcalConversionConstants_; // Lookup floating point ET corresponding to NL hcal compressed ET

};
#endif
