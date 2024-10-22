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

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <ostream>

class L1RCTParameters {
public:
  // constructor
  L1RCTParameters() {}

  L1RCTParameters(double eGammaLSB,
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
                  const std::vector<double>& ecal_calib,
                  const std::vector<double>& hcal_calib,
                  const std::vector<double>& hcal_high_calib,
                  const std::vector<double>& cross_terms,
                  const std::vector<double>& lowHoverE_smear,
                  const std::vector<double>& highHoverE_smear);

  // destructor -- no virtual methods in this class
  ~L1RCTParameters() { ; }

  // accessors

  double eGammaLSB() const { return eGammaLSB_; }
  double jetMETLSB() const { return jetMETLSB_; }
  double eMinForFGCut() const { return eMinForFGCut_; }
  double eMaxForFGCut() const { return eMaxForFGCut_; }
  double hOeCut() const { return hOeCut_; }
  double eMinForHoECut() const { return eMinForHoECut_; }
  double eMaxForHoECut() const { return eMaxForHoECut_; }
  double hMinForHoECut() const { return hMinForHoECut_; }
  double eActivityCut() const { return eActivityCut_; }
  double hActivityCut() const { return hActivityCut_; }
  unsigned eicIsolationThreshold() const { return eicIsolationThreshold_; }
  unsigned jscQuietThresholdBarrel() const { return jscQuietThresholdBarrel_; }
  unsigned jscQuietThresholdEndcap() const { return jscQuietThresholdEndcap_; }
  bool noiseVetoHB() const { return noiseVetoHB_; }
  bool noiseVetoHEplus() const { return noiseVetoHEplus_; }
  bool noiseVetoHEminus() const { return noiseVetoHEminus_; }
  const std::vector<double>& eGammaECalScaleFactors() const { return eGammaECalScaleFactors_; }
  const std::vector<double>& eGammaHCalScaleFactors() const { return eGammaHCalScaleFactors_; }
  const std::vector<double>& jetMETECalScaleFactors() const { return jetMETECalScaleFactors_; }
  const std::vector<double>& jetMETHCalScaleFactors() const { return jetMETHCalScaleFactors_; }

  // Helper methods to convert from trigger tower (iphi, ieta)
  // to RCT (crate, card, tower)

  unsigned short calcCrate(unsigned short rct_iphi, short ieta) const;
  unsigned short calcCard(unsigned short rct_iphi, unsigned short absIeta) const;
  unsigned short calcTower(unsigned short rct_iphi, unsigned short absIeta) const;
  short calcIEta(unsigned short iCrate, unsigned short iCard, unsigned short iTower) const;  // negative eta is used
  unsigned short calcIPhi(unsigned short iCrate, unsigned short iCard, unsigned short iTower) const;
  unsigned short calcIAbsEta(unsigned short iCrate, unsigned short iCard, unsigned short iTower) const;

  // Sum ecal and hcal TPGs using JetMET / EGamma Correactions and Lindsey's Calibration if flag is set
  float JetMETTPGSum(const float& ecal, const float& hcal, const unsigned& iAbsEta) const;
  float EGammaTPGSum(const float& ecal, const float& hcal, const unsigned& iAbsEta) const;

  void print(std::ostream& s) const;

private:
  // default constructor is not implemented

  //L1RCTParameters();

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

  // Minimum ET of the hcal (in GeV) above which H/E always fails (veto true)

  double hMinForHoECut_;

  // If the ET of the ECAL trigger tower is above this value
  // the tower is deemed active (in GeV)  --  these are used
  // for tau pattern logic

  double eActivityCut_;

  // If the ET of the HCAL trigger tower is above this value
  // the tower is deemed active (in GeV) -- these are used
  // for tau pattern logic

  double hActivityCut_;

  // This parameter is used for the five-tower-corner isolation
  // algorithm in the electron isolation card.  If one corner
  // set of five neighbor towers falls below this threshold,
  // the electron candidate is isolated.

  unsigned eicIsolationThreshold_;

  // 9-bit threshold below which quiet bit is set for a barrel region in JSC
  // (i.e. receiver cards 0-3)

  unsigned jscQuietThresholdBarrel_;

  // 9-bit threshold below which quiet bit is set for an endcap region in JSC
  // (i.e. receiver cards 4-6)

  unsigned jscQuietThresholdEndcap_;

  // Ignores HCAL barrel energy if no ECAL energy in corresponding
  // tower -- to reduce HCAL noise.  Endcaps enabled separately
  // to allow for lack of one/both ECAL endcaps.

  bool noiseVetoHB_;

  // Ignores HCAL energy in plus endcap if no ECAL energy in
  // corresponding tower.

  bool noiseVetoHEplus_;

  // Ignores HCAL energy in minus endcap if no ECAL energy in
  // corresponding tower.

  bool noiseVetoHEminus_;

  // Use Cubic Fitting Corrections ?
  bool useCorrections_;

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

  // Applies Lindsey's calibration to HCAL and ECAL (ECAL must corrected by eGamma scale factors)
  // Provides corrected Et sum.
  float correctedTPGSum(const float& ecal, const float& hcal, const unsigned& index) const;

  // Lindsey's Calibration Coefficients
  // Basically a higher order approximation of the energy response of the calorimeters.
  // Powers in ecal and hcal Et are defined below.
  std::vector<std::vector<double> > ecal_calib_;       // [0] = ecal^3, [1] = ecal^2, [2] = ecal
  std::vector<std::vector<double> > hcal_calib_;       // [0] = hcal^3, [1] = hcal^2, [2] = hcal
  std::vector<std::vector<double> > hcal_high_calib_;  // same as above but used to capture Et dependence for large Et
  std::vector<std::vector<double> > cross_terms_;      // [0] = ecal^2*hcal, [1] = hcal^2*ecal, [2] = ecal*hcal
                                                       // [3] = ecal^3*hcal, [1] = hcal^3*ecal, [2] = ecal^2*hcal^2
  // These two sets of correction factors help to center the corrected
  // Et distributions for different values of H/E.
  std::vector<double> HoverE_smear_low_;
  std::vector<double> HoverE_smear_high_;

  COND_SERIALIZABLE;
};

#endif
