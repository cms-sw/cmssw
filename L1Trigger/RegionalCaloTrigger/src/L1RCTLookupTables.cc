#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/L1TObjects/interface/L1RCTNoisyChannelMask.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

unsigned int L1RCTLookupTables::lookup(unsigned short ecalInput,
				       unsigned short hcalInput,
				       unsigned short fgbit,
				       unsigned short crtNo,
				       unsigned short crdNo,
				       unsigned short twrNo) const
{
  if(rctParameters_ == 0)
    throw cms::Exception("L1RCTParameters Invalid")
      << "L1RCTParameters should be set every event" << rctParameters_;
  if(channelMask_ == 0)
    throw cms::Exception("L1RCTChannelMask Invalid")
      << "L1RCTChannelMask should be set every event" << channelMask_;
  if(noisyChannelMask_ == 0)
    throw cms::Exception("L1RCTNoisyChannelMask Invalid")
      << "L1RCTNoisyChannelMask should be set every event" << noisyChannelMask_;
  if(ecalInput > 0xFF) 
    throw cms::Exception("Invalid Data") 
      << "ECAL compressedET should be less than 0xFF, is " << ecalInput;
  if(hcalInput > 0xFF) 
    throw cms::Exception("Invalid Data") 
      << "HCAL compressedET should be less than 0xFF, is " << hcalInput;
  if(fgbit > 1) 
    throw cms::Exception("Invalid Data") 
      << "ECAL finegrain should be a single bit, is " << fgbit;
  short iEta = (short) rctParameters_->calcIEta(crtNo, crdNo, twrNo);
  unsigned short iAbsEta = (unsigned short) abs(iEta);
  short sign = iEta/iAbsEta;
  unsigned short iPhi = rctParameters_->calcIPhi(crtNo, crdNo, twrNo);
  unsigned short phiSide = (iPhi/4)%2;
  if(iAbsEta < 1 || iAbsEta > 28) 
    throw cms::Exception("Invalid Data") 
      << "1 <= |IEta| <= 28, is " << iAbsEta;


  //Pre Input bits
  unsigned short ecalAfterMask=0;
  unsigned short hcalAfterMask=0;


  // using channel mask to mask off ecal channels
  //Mike: Introducing the hot channel mask 
  //If the Et is above the threshold then mask it as well


  
  float ecalBeforeMask = convertEcal(ecalInput, iAbsEta, sign);


  bool resetECAL = (channelMask_->ecalMask[crtNo][phiSide][iAbsEta-1]) || //channel mask
    (noisyChannelMask_->ecalMask[crtNo][phiSide][iAbsEta-1] &&
     ecalBeforeMask<noisyChannelMask_->ecalThreshold)||//hot mask
      (rctParameters_->eGammaECalScaleFactors()[iAbsEta-1] == 0.&&
       rctParameters_->jetMETECalScaleFactors()[iAbsEta-1] == 0.);       
    


  if (resetECAL)    {
      ecalAfterMask=0;
    }
  else    {
      ecalAfterMask=ecalInput;
    }

  float ecal =  convertEcal(ecalAfterMask, iAbsEta, sign);


  // masking off hcal for channels in channel mask
  float hcalBeforeMask = convertHcal(hcalInput, iAbsEta, sign);

  bool resetHCAL = channelMask_->hcalMask[crtNo][phiSide][iAbsEta-1]||
    (noisyChannelMask_->hcalMask[crtNo][phiSide][iAbsEta-1] &&
     hcalBeforeMask<noisyChannelMask_->hcalThreshold)||//hot mask
      (rctParameters_->eGammaHCalScaleFactors()[iAbsEta-1] == 0.&&
       rctParameters_->jetMETHCalScaleFactors()[iAbsEta-1] == 0.);

  if (resetHCAL)    {
      hcalAfterMask=0;
    }
  else    {
      hcalAfterMask=hcalInput;
    }

  float hcal = convertHcal(hcalAfterMask, iAbsEta, sign);

  unsigned long etIn7Bits;
  unsigned long etIn9Bits;

  if((ecalAfterMask == 0 && hcalAfterMask > 0) &&
     ((rctParameters_->noiseVetoHB() && iAbsEta > 0 && iAbsEta < 18)
      || (rctParameters_->noiseVetoHEplus() && iAbsEta>17 && crtNo>8)
      || (rctParameters_->noiseVetoHEminus() && iAbsEta>17 && crtNo<9)))
    {
      etIn7Bits = 0;
      etIn9Bits = 0;
    }
  else
    {
      etIn7Bits = eGammaETCode(ecal, hcal, iAbsEta);
      etIn9Bits = jetMETETCode(ecal, hcal, iAbsEta);
    }
  // Saturated input towers cause tower ET pegging at the highest value
  if((ecalAfterMask == 0xFF && 
      rctParameters_->eGammaECalScaleFactors()[iAbsEta-1] != 0. ) 
     || (hcalAfterMask == 0xFF &&
	 rctParameters_->eGammaHCalScaleFactors()[iAbsEta-1] != 0. )
     )
    {
      etIn7Bits = 0x7F; // egamma path
    }
  if((ecalAfterMask == 0xFF &&
      rctParameters_->jetMETECalScaleFactors()[iAbsEta-1] != 0. )
     || (hcalAfterMask == 0xFF &&
	 rctParameters_->jetMETHCalScaleFactors()[iAbsEta-1] != 0. ))
    {
      etIn9Bits = 0x1FF; // sums path
    }

  unsigned long shiftEtIn9Bits = etIn9Bits<<8;
  unsigned long shiftHE_FGBit = hOeFGVetoBit(ecal, hcal, fgbit)<<7;
  unsigned long shiftActivityBit = 0;
  if ( rctParameters_->jetMETECalScaleFactors()[iAbsEta-1] == 0.
       && rctParameters_->jetMETHCalScaleFactors()[iAbsEta-1] == 0. )
    {
      // do nothing, it's already zero
    }
  else if (rctParameters_->jetMETECalScaleFactors()[iAbsEta-1] == 0. )
    {
      shiftActivityBit = activityBit(0., hcal,fgbit)<<17;
    }
  else if (rctParameters_->jetMETHCalScaleFactors()[iAbsEta-1] == 0. )
    {
      shiftActivityBit = activityBit(ecal, 0.,fgbit)<<17;
    }
  else
    {
      shiftActivityBit = activityBit(ecal, hcal,fgbit)<<17;
    }
  unsigned long output=etIn7Bits+shiftHE_FGBit+shiftEtIn9Bits+shiftActivityBit;
  return output;
}

unsigned int L1RCTLookupTables::lookup(unsigned short hfInput,
				       unsigned short crtNo,
				       unsigned short crdNo,
				       unsigned short twrNo
				       ) const
{
  if(rctParameters_ == 0)
    throw cms::Exception("L1RCTParameters Invalid")
      << "L1RCTParameters should be set every event" << rctParameters_;
  if(channelMask_ == 0)
    throw cms::Exception("L1RCTChannelMask Invalid")
      << "L1RCTChannelMask should be set every event" << channelMask_;
  if(hfInput > 0xFF) 
    throw cms::Exception("Invalid Data") 
      << "HF compressedET should be less than 0xFF, is " << hfInput;
  short iEta = rctParameters_->calcIEta(crtNo, crdNo, twrNo);
  unsigned short iAbsEta = abs(iEta);
  short sign = (iEta/iAbsEta);
  unsigned short phiSide = twrNo/4;
  if(iAbsEta < 29 || iAbsEta > 32) 
    throw cms::Exception("Invalid Data") 
      << "29 <= |iEta| <= 32, is " << iAbsEta;

  float et = convertHcal(hfInput, iAbsEta, sign);;



  if (channelMask_->hfMask[crtNo][phiSide][iAbsEta-29]||
      (noisyChannelMask_->hfMask[crtNo][phiSide][iAbsEta-29]&&
       et<noisyChannelMask_->hfThreshold))
    {
      et = 0;
    }

  unsigned int result = convertToInteger(et, rctParameters_->jetMETLSB(), 8);
  return result;
}

bool L1RCTLookupTables::hOeFGVetoBit(float ecal, float hcal, bool fgbit) const
{
  if(rctParameters_ == 0)
    throw cms::Exception("L1RCTParameters Invalid")
      << "L1RCTParameters should be set every event" << rctParameters_;
  bool veto = false;
  if(ecal > rctParameters_->eMinForFGCut() && 
     ecal < rctParameters_->eMaxForFGCut())
    {
      if(fgbit) veto = true;
    }
  if(ecal >= rctParameters_->eMinForHoECut() && 
     ecal < rctParameters_->eMaxForHoECut())
    {
      if((hcal / ecal) > rctParameters_->hOeCut()) veto = true;
    }
  //  else
  if (ecal < rctParameters_->eMinForHoECut())
    {
      if(hcal >= rctParameters_->hMinForHoECut()) veto = true;  // Changed from eMinForHoECut() - JLL 2008-Feb-13
    }
  return veto;
}

bool L1RCTLookupTables::activityBit(float ecal, float hcal,bool fgbit) const
{
  // Redefined for upgrade as EM activity only
  if(rctParameters_ == 0)
    throw cms::Exception("L1RCTParameters Invalid")
      << "L1RCTParameters should be set every event" << rctParameters_;
  bool aBit = false;
  if(rctParameters_->eMinForHoECut() < rctParameters_->eMaxForHoECut()) {
    // For RCT operations HoE cut and tauVeto are used
    aBit = ((ecal > rctParameters_->eActivityCut()) || 
	    (hcal > rctParameters_->hActivityCut()));
  }
  else {
    // We redefine tauVeto() for upgrade as EM activity only  -- 
    // both EG and Tau make it through the EIC and JSC to CTP cards
    // In the CTP card we want to rechannel EG/Tau candidates to EG and Tau
//    if(ecal > rctParameters_->eActivityCut()) {
//      if((hcal/ecal) < rctParameters_->hOeCut()) {
//	aBit = true;
//      }
//    }
        if(fgbit ||  ((ecal)>(rctParameters_->eActivityCut())&&hcal/(ecal+hcal)>rctParameters_->hOeCut()) || ((ecal)<=(rctParameters_->eActivityCut()) && hcal > rctParameters_->hActivityCut())){
        aBit = true;
      }
  }
  return aBit;
}

// uses etScale
unsigned int L1RCTLookupTables::emRank(unsigned short energy) const 
{
  if(etScale_)
    {
      return etScale_->rank(energy);
    }
  else
    //    edm::LogInfo("L1RegionalCaloTrigger") 
    //      << "CaloEtScale was not used - energy instead of rank" << std::endl;
  return energy;
}

// converts compressed ecal energy to linear (real) scale
float L1RCTLookupTables::convertEcal(unsigned short ecal, unsigned short iAbsEta, short sign) const
{
  if(ecalScale_)
    {
      //std::cout << "[luts] energy " << ecal << " sign " << sign 
      //<< " iAbsEta " << iAbsEta << " iPhi "	<< iPhi << std::endl;
      float dummy = 0;
      dummy = float (ecalScale_->et( ecal, iAbsEta, sign ));
      /*
      if (ecal > 0)
	{
	  std::cout << "[luts] ecal converted from " << ecal << " to " 
		    << dummy << " with iAbsEta " << iAbsEta << std::endl;
	}
      */
      return dummy;
    }
  //else if(rctParameters_ == 0)
  //  {
  //    throw cms::Exception("L1RCTParameters Invalid")
  //	<< "L1RCTParameters should be set every event" << rctParameters_;
  //  }
  else
    {
      return ((float) ecal) * rctParameters_->eGammaLSB();
    }
}

// converts compressed hcal energy to linear (real) scale
float L1RCTLookupTables::convertHcal(unsigned short hcal, unsigned short iAbsEta, short sign) const
{
  if (hcalScale_ != 0)
    {
      return (hcalScale_->et( hcal, iAbsEta, sign ));
    }
  else
    {
      //      edm::LogInfo("L1RegionalCaloTrigger") 
      //	<< "CaloTPGTranscoder was not used" << std::endl;
      return ((float) hcal) * rctParameters_->jetMETLSB();
    }
}

// integerize given an LSB and set maximum value of 2^precision-1
unsigned long L1RCTLookupTables::convertToInteger(float et, 
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

unsigned int L1RCTLookupTables::eGammaETCode(float ecal, float hcal, int iAbsEta) const
{
  if(rctParameters_ == 0)
    throw cms::Exception("L1RCTParameters Invalid")
      << "L1RCTParameters should be set every event" << rctParameters_;
  float etLinear = rctParameters_->EGammaTPGSum(ecal,hcal,iAbsEta);
  return convertToInteger(etLinear, rctParameters_->eGammaLSB(), 7);
}

unsigned int L1RCTLookupTables::jetMETETCode(float ecal, float hcal, int iAbsEta) const
{
  if(rctParameters_ == 0)
    throw cms::Exception("L1RCTParameters Invalid")
      << "L1RCTParameters should be set every event" << rctParameters_;
  float etLinear = rctParameters_->JetMETTPGSum(ecal,hcal,iAbsEta);
  return convertToInteger(etLinear, rctParameters_->jetMETLSB(), 9);
}
