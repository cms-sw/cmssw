/**
 * Author: Sridhara Dasu
 * Created: 04 July 2007
 * $Id: L1RCTParameters.cc,v 1.25 2010/05/26 22:52:54 ghete Exp $
 **/

#include <iostream>
#include <fstream>
#include <cmath>

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include <iomanip>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace std;

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
				 bool useCorrections,
				 const std::vector<double>& eGammaECalScaleFactors,
				 const std::vector<double>& eGammaHCalScaleFactors,
				 const std::vector<double>& jetMETECalScaleFactors,
				 const std::vector<double>& jetMETHCalScaleFactors,
				 const std::vector<double>& ecal_calib,
				 const std::vector<double>& hcal_calib,
				 const std::vector<double>& hcal_high_calib,
				 const std::vector<double>& cross_terms,
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
  useCorrections_(useCorrections),
  eGammaECalScaleFactors_(eGammaECalScaleFactors),
  eGammaHCalScaleFactors_(eGammaHCalScaleFactors),
  jetMETECalScaleFactors_(jetMETECalScaleFactors),
  jetMETHCalScaleFactors_(jetMETHCalScaleFactors),
  HoverE_smear_low_(lowHoverE_smear),
  HoverE_smear_high_(highHoverE_smear)
{
  ecal_calib_.resize(28);
  hcal_calib_.resize(28);
  hcal_high_calib_.resize(28);
  cross_terms_.resize(28);

  for(unsigned i = 0; i < ecal_calib.size(); ++i)
    ecal_calib_[i/3].push_back(ecal_calib[i]);
  for(unsigned i = 0; i < hcal_calib.size(); ++i)
    hcal_calib_[i/3].push_back(hcal_calib[i]);
  for(unsigned i = 0; i < hcal_high_calib.size(); ++i)
    hcal_high_calib_[i/3].push_back(hcal_high_calib[i]);
  for(unsigned i = 0; i < cross_terms.size(); ++i)
    cross_terms_[i/6].push_back(cross_terms[i]);
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

  if(useCorrections_)
    {
      if(jetMETHCalScaleFactors_.at(iAbsEta-1) != 0)
	hcal_c = hcal;

      if(jetMETECalScaleFactors_.at(iAbsEta-1) != 0)
	ecal_c = ecal*eGammaECalScaleFactors_.at(iAbsEta-1); // Use eGamma Corrections

      result = correctedTPGSum(ecal_c,hcal_c,iAbsEta-1);
    }

  return result;
}

float L1RCTParameters::EGammaTPGSum(const float& ecal, const float& hcal, const unsigned& iAbsEta) const
{
  float ecal_c = ecal*eGammaECalScaleFactors_.at(iAbsEta-1);
  float hcal_c = hcal*eGammaHCalScaleFactors_.at(iAbsEta-1);
  float result = ecal_c + hcal_c;

  if(useCorrections_)
    {
      if(eGammaHCalScaleFactors_.at(iAbsEta-1) != 0)
	hcal_c = hcal;
      
      result = correctedTPGSum(ecal_c,hcal_c,iAbsEta-1);
    }

  return result;
}

// index = iAbsEta - 1... make sure you call the function like so: "correctedTPGSum(ecal,hcal, iAbsEta - 1)"
float L1RCTParameters::correctedTPGSum(const float& ecal, const float& hcal, const unsigned& index) const
{
  if(index >= 28 && ecal > 120 && hcal > 120) return (ecal + hcal); // return plain sum if outside of calibration range or index is too high

  // let's make sure we're asking for items that are there.
  if(ecal_calib_.at(index).size() != 3 || hcal_calib_.at(index).size() != 3 ||
     hcal_high_calib_.at(index).size() != 3 || cross_terms_.at(index).size() != 6 ||
     HoverE_smear_high_.size() <= index || HoverE_smear_low_.size() <= index)
    return (ecal+hcal);

  double e = ecal, h = hcal;
  double ec = 0.0, hc = 0.0, c = 0.0;
  
  ec = (ecal_calib_.at(index).at(0)*std::pow(e,3.) +
	ecal_calib_.at(index).at(1)*std::pow(e,2.) +
	ecal_calib_.at(index).at(2)*e);
  
  if(e + h < 23)
    {
      hc = (hcal_calib_.at(index).at(0)*std::pow(h,3.) + 
	    hcal_calib_.at(index).at(1)*std::pow(h,2.) + 
	    hcal_calib_.at(index).at(2)*h);
      
      c = (cross_terms_.at(index).at(0)*std::pow(e, 2.)*h +
	   cross_terms_.at(index).at(1)*std::pow(h, 2.)*e +
	   cross_terms_.at(index).at(2)*e*h +
	   cross_terms_.at(index).at(3)*std::pow(e, 3.)*h +
	   cross_terms_.at(index).at(4)*std::pow(h, 3.)*e +
	   cross_terms_.at(index).at(5)*std::pow(h, 2.)*std::pow(e, 2.));
    }
  else
    {
      hc = (hcal_high_calib_.at(index).at(0)*std::pow(h,3.) + 
	    hcal_high_calib_.at(index).at(1)*std::pow(h,2.) + 
	    hcal_high_calib_.at(index).at(2)*h);
    }
    
  if(h/(e+h) >= 0.05)
    {
      ec *= HoverE_smear_high_.at(index);
      hc *= HoverE_smear_high_.at(index);
      c *= HoverE_smear_high_.at(index);
    }
  else
    {
      ec *= HoverE_smear_low_.at(index);
    }
  return ec + hc + c;
}
 
void 
L1RCTParameters::print(std::ostream& s)  const {

    s << "\nPrinting record L1RCTParametersRcd" <<endl;
    s << "\n\"Parameter description\" \n  \"Parameter name\"  \"Value\" "
            << endl;
    s << "\ne/gamma least significant bit energy transmitted from receiver cards to EIC cards. \n  "
            << "eGammaLSB = " << eGammaLSB_ << endl ;
    s << "\nLSB of region Et scale from RCT to GCT (GeV) \n  "
            << "jetMETLSB = " << jetMETLSB_ << endl ;
    s << "\nminimum ECAL Et for which fine-grain veto is applied (GeV) \n  "
            << " eMinForFGCut = " << eMinForFGCut_ << endl ;
    s << "\nmaximum ECAL Et for which fine-grain veto is applied (GeV) \n  "
            << "eMaxForFGCut = " << eMaxForFGCut_ << endl ;
    s << "\nmaximum value of (HCAL Et / ECAL Et) \n  "
            << "hOeCut = " << hOeCut_ << endl ;
    s << "\nminimum ECAL Et for which H/E veto is applied (GeV) \n  "
            << "eMinForHoECut = " << eMinForHoECut_ << endl ;
    s << "\nmaximum ECAL Et for which H/E veto is applied (GeV) \n  "
            << "eMaxForHoECut = " << eMaxForHoECut_ << endl ;
    s << "\nminimum HCAL Et for which H/E veto is applied (GeV)  \n  "
            << "hMinForHoECut = " << hMinForHoECut_ << endl ;
    s << "\nECAL Et threshold above which tau activity bit is set (GeV)  \n  "
            << "eActivityCut = " << eActivityCut_ << endl ;
    s << "\nHCAL Et threshold above which tau activity bit is set (GeV)  \n  "
            << "hActivityCut = " << hActivityCut_ << endl ;
    s << "\nNeighboring trigger tower energy minimum threshold that marks candidate as non-isolated. (LSB bits) \n  "
            << "eicIsolationThreshold = " << eicIsolationThreshold_ << endl ;
    s << "\nIf jetMet energy in RCT Barrel Region is below this value, a quiet bit is set. (LSB bits)\n  "
            << "jscQuietThreshBarrel = " << jscQuietThresholdBarrel_ << endl ;
    s << "\nIf jetMet energy in RCT Endcap Region is below this value, a quiet bit is set. (LSB bits) \n  "
            << "jscQuietThreshEndcap = " << jscQuietThresholdEndcap_ << endl ;
    s << "\nWhen set to TRUE, HCAL energy is ignored if no ECAL energy is present in corresponding trigger tower for RCT Barrel \n  "
            << "noiseVetoHB = " << noiseVetoHB_ << endl ;
    s << "\nWhen set to TRUE, HCAL energy is ignored if no ECAL energy is present in corresponding trigger tower for RCT Encap+ \n  "
            << "noiseVetoHEplus = " << noiseVetoHEplus_ << endl ;
    s << "\nWhen set to TRUE, HCAL energy is ignored if no ECAL energy is present in corresponding trigger tower for RCT Endcap- \n  "
            << "noiseVetoHEminus = " << noiseVetoHEminus_ << endl ;

    s << "\n\neta-dependent multiplicative factors for ECAL Et before summation \n  "
            << "eGammaECal Scale Factors " << endl;
    s << "ieta  ScaleFactor" <<endl;
    for(int i = 0 ; i<28; i++)
        s << setw(4) << i+1 << "  " << eGammaECalScaleFactors_.at(i) <<endl;

    s << "\n\neta-dependent multiplicative factors for HCAL Et before summation \n  "
            <<"eGammaHCal Scale Factors "<<endl;
    s << "ieta  ScaleFactor" <<endl;
    for(int i = 0 ; i<28; i++)
        s << setw(4) << i+1 << "  " << eGammaHCalScaleFactors_.at(i) <<endl;
     
    s << "\n\neta-dependent multiplicative factors for ECAL Et before summation \n  "
            <<"jetMETECal Scale Factors "<<endl;
    s << "ieta  ScaleFactor" <<endl;
    for(int i = 0 ; i<28; i++)
        s<< setw(4) << i+1 << "  " << jetMETECalScaleFactors_.at(i) <<endl;
     
    s << "\n\neta-dependent multiplicative factors for HCAL Et before summation \n"
            <<"jetMETHCal Scale Factors "<<endl;
    s << "ieta  ScaleFactor" <<endl;
    for(int i = 0 ; i<28; i++)
        s << setw(4) <<i+1 << "  " << jetMETHCalScaleFactors_.at(i) <<endl;


    if(useCorrections_) {
        s<< "\n\nUSING calibration variables " <<endl;


        s << "\n\nH over E smear low Correction Factors "<<endl;
        s << "ieta  Correction Factor" <<endl;
        for(int i = 0 ; i<28; i++)
            s << setw(4) <<i+1 << "  " << HoverE_smear_low_.at(i) <<endl;


        s << "\n\nH over E smear high Correction Factors "<<endl;
        s <<"ieta  Correction Factor" <<endl;
        for(int i = 0 ; i<28; i++)
            s << setw(4) <<i+1 << "  " << HoverE_smear_high_.at(i) <<endl;

        s << "\n\necal calibrations "<<endl;
        s << "ieta  CorrFactor1  CorrFactor2  CorrFactor3" <<endl;
        int end =  ecal_calib_[0].size();
        for(int i = 0 ; i<28; i++) {
            s << setw(4) << i;
            for(int j = 0; j< end ; j++)
                s << setw(11) << setprecision(8) << ecal_calib_[i][j] ;
	 
            s << endl;

        }

        s <<"\n\nhcal calibrations "<<endl;
        s <<"ieta  CorrFactor1  CorrFactor2  CorrFactor3" <<endl;
        end =  hcal_calib_[0].size();
        for(int i = 0 ; i<28; i++) {
            s << setw(4) << i;
            for(int j = 0; j< end ; j++)
                s << setw(11) << setprecision(8) << hcal_calib_[i][j] ;
	 
            s << endl;

        }
        s <<"\n\nhcal_high calibrations "<<endl;
        s <<"ieta  CorrFactor1  CorrFactor2  CorrFactor3" <<endl;
        end =  hcal_high_calib_[0].size();
        for(int i = 0 ; i<28; i++) {
            s << setw(4) << i;
            for(int j = 0; j< end ; j++)
                s << setw(11) << setprecision(8) << hcal_high_calib_[i][j] ;
	 
            s << endl;

        }
        end = cross_terms_[0].size();
        s <<"\n\ncross terms calibrations "<<endl;
        s <<"ieta  CorrFactor1  CorrFactor2  CorrFactor3  CorrFactor4  CorrFactor5  CorrFactor6" <<endl;
        for(int i = 0 ; i<28; i++) {
            s << setw(4) << i;
            for(int j = 0; j< end ; j++)
                s << setw(11) << setprecision(8) << cross_terms_[i][j] ;
	 
            s << endl;

        }
 
    } else
        s<< "\n\nNOT USING calibration variables " <<endl;

    s << "\n\n" <<endl;

}
