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

  // destructor -- no virtual methods in this class
  ~L1RCTParameters() {;}
  
  // accessors
  
  double eGammaLSB() const {return eGammaLSB_;}
  double jetMETLSB() const {return jetMETLSB_;}
  double eMinForFGCut() const {return eMinForFGCut_;}
  double eMaxForFGCut() const {return eMaxForFGCut_;}
  double hOeCut() const {return hOeCut_;}
  double eMinForHoECut() const {return eMinForHoECut_;}
  double eMaxForHoECut() const {return eMaxForHoECut_;}
  double eActivityCut() const {return eActivityCut_;}
  double hActivityCut() const {return hActivityCut_;}
  std::vector<double> eGammaECalScaleFactors() const {return eGammaECalScaleFactors_;}
  std::vector<double> eGammaHCalScaleFactors() const {return eGammaHCalScaleFactors_;}
  std::vector<double> jetMETECalScaleFactors() const {return jetMETECalScaleFactors_;}
  std::vector<double> jetMETHCalScaleFactors() const {return jetMETHCalScaleFactors_;}

  // Helper methods to convert from trigger tower (iphi, ieta) 
  // to RCT (crate, card, tower)
  
  unsigned short calcCrate(unsigned short rct_iphi, short ieta) const;
  unsigned short calcCard(unsigned short rct_iphi, unsigned short absIeta) const;
  unsigned short calcTower(unsigned short rct_iphi, unsigned short absIeta) const;
  short calcIEta(unsigned short iCrate, unsigned short iCard, unsigned short iTower) const; // negative eta is used
  unsigned short calcIPhi(unsigned short iCrate, unsigned short iCard, unsigned short iTower) const;
  unsigned short calcIAbsEta(unsigned short iCrate, unsigned short iCard, unsigned short iTower) const;

  void print(std::ostream& s) const {return;}

 private:

  // default constructor is not implemented

  L1RCTParameters();

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

};

#endif
