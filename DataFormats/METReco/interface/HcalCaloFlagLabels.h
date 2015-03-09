#ifndef DataFormats_METReco_HcalCaloFlagLabels_h
#define DataFormats_METReco_HcalCaloFlagLabels_h

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include <string>

// Create alias names for all status bits
// These aliases are valid for only the _current release_
// Use the HcalCaloFlagTool for full interpretation
namespace HcalCaloFlagLabels
{
  //subdetector-specific bits defined here (bits 0-15)
  enum HBHEStatusFlag{HBHEHpdHitMultiplicity=0,
                      HBHEPulseShape=1,
		      HSCP_R1R2=2,
		      HSCP_FracLeader=3,
		      HSCP_OuterEnergy=4,
		      HSCP_ExpFit=5,
                      HBHETimingTrustBits=6, // 2-bit counter; not yet in use
                      HBHETimingShapedCutsBits=8, // 3-bit counter
            HBHENegativeNoise=27,
		      HBHEIsolatedNoise=11,
		      HBHEFlatNoise=12,
		      HBHESpikeNoise=13,
		      HBHETriangleNoise=14,
		      HBHETS4TS5Noise=15
  };

  enum HFTimingTrustFlag{HFTimingTrustBits=6};

  enum HOStatusFlag{HOBit=0};

  enum HFStatusFlag{HFLongShort=0,
		    HFDigiTime=1,
		    HFInTimeWindow=2, // requires hit be within certain time window
		    HFS8S1Ratio=3,
		    HFPET=4
  };


  enum ZDCStatusFlag{ZDCBit=0};

  enum CalibrationFlag{CalibrationBit=0};

  // Bit definitions that apply to all subdetectors (bits 16-31)
  enum CommonFlag {TimingSubtractedBit=16, // latency shift correction, recovered
		   TimingAddedBit=17,      // latency shift correction, recovered
		   TimingErrorBit=18,      // latency shift error, unrecovered
		   ADCSaturationBit=19,
                   Fraction2TS=20, // should deprecate this at some point
		   PresampleADC=20, // uses 7 bits to store ADC from presample
		   // This bit is not yet in use (as of March 2012), but can be used to mark sim hits to which noise has been intentionally added
		   AddedSimHcalNoise=28,
		   // The following bits are all user-defined; reverse-order them so that UserDefinedBit0 will be the last removed
		   UserDefinedBit2 = 29,
		   UserDefinedBit1 = 30,
		   UserDefinedBit0 = 31
}; 
  
}

#endif //DataFormats_METReco_HcalCaloFlagLabels_h
