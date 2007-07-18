#ifndef L1TObjects_L1RCTParameters_h
#define L1TObjects_L1RCTParameters_h
// -*- C++ -*-
//
// Package:     L1TObjects
// Class  :     L1RCTParameters
// 
/**\class L1RCTParameters L1RCTParameters.h CondFormats/L1TObjects/interface/L1RCTParameters.h

 Description: Class to contain parameters which define RCT Lookup Tables

 Usage:
    <usage>

*/
//
// Author:      Sridhara Dasu
// Created:     Thu Jun 07 06:35 PDT 2007
// $Id: 
//

#include <boost/cstdint.hpp>
#include <vector>
#include <ostream>

#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

class L1RCTParameters {

 public:

  // constructor
  L1RCTParameters(double eGammaLSB,
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
		  );
  // this can only be set after construction -- constructor inits to zero
  // to indicate that transcoder cannot be used -- if this function is
  // called to set transcoder, lookup after that call will use it.
  void setTranscoder(edm::ESHandle<CaloTPGTranscoder> transcoder)
    {
      transcoder_ = transcoder;
    }
  
  // destructor -- no virtual methods in this class
  ~L1RCTParameters() {;}
  
  // accessors
  
  double eGammaLSB() {return eGammaLSB_;}
  double jetMETLSB() {return jetMETLSB_;}
  double hOeCut() {return hOeCut_;}
  double eMinForHoECut() {return eMinForHoECut_;}
  double eMaxForHoECut() {return eMaxForHoECut_;}
  double eActivityCut() {return eActivityCut_;}
  double hActivityCut() {return hActivityCut_;}
  double eMaxForFGCut() {return eMaxForFGCut_;}

  unsigned int lookup(unsigned short ecalInput,
		      unsigned short hcalInput,
		      unsigned short fgbit,
		      unsigned short crtNo,
		      unsigned short crdNo,
		      unsigned short twrNo
		      );

  unsigned int lookup(unsigned short hfInput, 
		      unsigned short crtNo,
		      unsigned short crdNo,
		      unsigned short twrNo
		      );

  unsigned int eGammaETCode(float ecal, float hcal, int iAbsEta);
  unsigned int jetMETETCode(float ecal, float hcal, int iAbsEta);
  bool hOeFGVetoBit(float ecal, float hcal, bool fgbit);
  bool activityBit(float ecal, float hcal);
  
  // Helper methods to convert from trigger tower (iphi, ieta) 
  // to RCT (crate, card, tower)
  
  unsigned short calcCrate(unsigned short rct_iphi, short ieta);
  unsigned short calcCard(unsigned short rct_iphi, unsigned short absIeta);
  unsigned short calcTower(unsigned short rct_iphi, unsigned short absIeta);
  short calcIEta(unsigned short iCrate, unsigned short iCard, unsigned short iTower); // negative eta is used
  unsigned short calcIPhi(unsigned short iCrate, unsigned short iCard, unsigned short iTower);

  void print(std::ostream& s) const {return;}

 private:

  // default constructor is not implemented

  L1RCTParameters();

  // helper functions

  float convertEcal(unsigned short ecal, int iAbsEta);
  float convertHcal(unsigned short hcal, int iAbsEta);
  unsigned short calcActivityBit(float ecal, float hcal);
  unsigned short calcHEBit(float ecal,float hcal, bool fgbit);
  unsigned long convertToInteger(float et, float lsb, int precision);

  // LSB of the eGamma object corresponds to this ET (in GeV)

  double eGammaLSB_;

  // LSB of the jetMET object corresponds to this ET (in GeV)

  double jetMETLSB_;

  // Minimum ET of the eGamma object below which FG cut is ignored (in GeV)

  double eMinForFGCut_;

  // Maximum ET of the eGamma object above which FG cut is ignored (in GeV)

  double eMaxForFGCut_;

  // H/E ratio cut

  double hOeCut_;

  // Minimum ET of the ecal (in GeV) below which H/E always passes

  double eMinForHoECut_;

  // Maximum ET of the ecal (in GeV) above which H/E always passes

  double eMaxForHoECut_;

  // If the ET of the ECAL trigger tower is above this value
  // the tower is deemed active (in GeV)  --  these are used
  // for tau pattern logic

  double eActivityCut_;

  // If the ET of the HCAL trigger tower is above this value
  // the tower is deemed active (in GeV) -- these are used
  // for tau pattern logic
  
  double hActivityCut_;

  // eGamma object ET is computed using the trigger tower ET defined as
  // ecal * eGammaECalScaleFactors[iEta] + hcal * eGammaHCalScaleFactors[iEta]
  // The result is then digitized using the eGamma LSB

  std::vector<double> eGammaECalScaleFactors_;
  std::vector<double> eGammaHCalScaleFactors_;

  // jetMET object ET is computed using the trigger tower ET defined as
  // ecal * jetMETECalScaleFactors[iEta] + hcal * jetMETHCalScaleFactors[iEta]
  // The result is then digitized using the jetMET LSB

  std::vector<double> jetMETECalScaleFactors_;
  std::vector<double> jetMETHCalScaleFactors_;

  edm::ESHandle<CaloTPGTranscoder> transcoder_;

};

#endif
