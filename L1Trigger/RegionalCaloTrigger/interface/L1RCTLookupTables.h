#ifndef L1RCTLookupTables_h
#define L1RCTLookupTables_h

#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

class L1RCTLookupTables {
 
 public:

  // constructor

  L1RCTLookupTables(const std::string& filename);
  L1RCTLookupTables(const std::string& filename, const std::string& filename2, bool patternTest);
  L1RCTLookupTables(const std::string& filename, edm::ESHandle<CaloTPGTranscoder> transcoder);

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

  // Parameters of RCT LUTs and LSBs which are read from LUT file

  static float eActivityCut() {return eActivityCut_;}
  static float hActivityCut() {return hActivityCut_;}
  static float hOeCut() {return hOeCut_;}
  static float eGammaLSB() {return eGammaLSB_;}
  static float jetMETLSB() {return jetMETLSB_;}

  //  void setPatternTest(bool patternTestBit) {patternTest_ = patternTestBit;}
  //  void setIgnoreFineGrain(bool ignoreFineGrainBit) {ignoreFG_ = ignoreFineGrainBit;}

 private:

  L1RCTLookupTables();  // Do not implement so one cannot instantiate without input file

  float convertEcal(unsigned short ecal, int iAbsEta);
  float convertHcal(unsigned short hcal, int iAbsEta);
  bool calcActivityBit(float ecal, float hcal);
  bool calcHEBit(float ecal,float hcal, bool fgbit);
  unsigned long convertToInteger(float et, float lsb, int precision);

  bool useTranscoder_;
  edm::ESHandle<CaloTPGTranscoder> transcoder_;

  bool patternTest_;
  bool ignoreFG_;

  // We implement LUTs in code -- real lookup tables are large 2^17 (addresses) x 18 phi bins x 18 bits
  // So we only read in conversion constants in float given by TPG developers, and implement LUTs on-the-fly
  // This same code can be used to write out real LUTs and load in the hardware
  // The read in conversion constants should come from the database, but for the moment we use a flat file
  void loadLUTConstants(const std::string& filename);
  void loadHcalLut(const std::string& filename);
  static float eActivityCut_;
  static float hActivityCut_;
  static float eMaxForFGCut_;
  static float hOeCut_;
  static float eGammaLSB_;
  static float jetMETLSB_;
  static float eGammaSCF_[32];
  static float hcalSCF_[32];
};
#endif
