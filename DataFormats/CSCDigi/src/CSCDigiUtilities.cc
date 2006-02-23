#include <DataFormats/CSCDigi/interface/CSCDigiUtilities.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigi.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigi.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

/// Taken from L1CSCTrigger/L1MuCSCMotherboard
int CSCDigiUtilities::correlatedQuality(const CSCCLCTDigi& cLCT, 
					const CSCALCTDigi& aLCT)
{ /// This might need to be updated!!!! Needs reality check 23/2/05

  
  unsigned int quality = 0;
  bool isDistrip = (cLCT.getStriptype() == 0);
  /*
  if (aLCT.getValid() && !(cLCT.getValid())) {    // no CLCT
    if (accelerator(aLCT)) {quality =  1;}
    else                       {quality =  3;}
  }


  else if (!(aLCT.getValid()) && cLCT.getValid()) { // no ALCT
    if (isDistrip)             {quality =  4;}
    else                       {quality =  5;}
  }
  else if (aLCT.getValid() && cLCT.getValid()) { // both ALCT and CLCT
    if (accelerator(aLCT)) {quality =  2;} // accelerator muon
    else {                                 // collision muon
      int sumQual = aLCT.getQuality() + cLCT.getQuality();
      if (sumQual < 1 || sumQual > 6) {
	edm::LogWarning("CSCDigiUtilities::correlatedQuality") << "+++ sumQual = " << sumQual 
							       << " is out of range.";
      }
      if (isDistrip) { // distrip pattern
	if (sumQual == 2)      {quality =  6;}
	else if (sumQual == 3) {quality =  7;}
	else if (sumQual == 4) {quality =  8;}
	else if (sumQual == 5) {quality =  9;}
	else if (sumQual == 6) {quality = 10;}
      }
      else {            // halfstrip pattern
	if (sumQual == 2)      {quality = 11;}
	else if (sumQual == 3) {quality = 12;}
	else if (sumQual == 4) {quality = 13;}
	else if (sumQual == 5) {quality = 14;}
	else if (sumQual == 6) {quality = 15;}
      }
    }
  }
  */
  return quality;
  
}

///Taken from CSCTrigger/L1MuCSCCorrelatedLCT
int CSCDigiUtilities::lctQuality(const CSCCorrelatedLCTDigi& CorrLCT)
{
  int quality = CorrLCT.getQuality();
  int alct_clct_quality = 0;
  if (quality == 10 || quality == 15) 
    alct_clct_quality = 3; // exact assignment
  else if (quality == 6 || quality == 11) 
    alct_clct_quality = 1; // assuming 4-layers minimum
  else 
    alct_clct_quality = 2; // pure guessing; no way to know
  
  return alct_clct_quality;
}

///Taken from CSCTrigger/L1MuCSCCorrelatedLCT
bool CSCDigiUtilities::validCLCT(const CSCCorrelatedLCTDigi& CorrLCT)
{
  int quality = CorrLCT.getQuality();
  return (quality != 1 && quality != 3);
}

///Taken from CSCTrigger/L1MuCSCCorrelatedLCT
bool CSCDigiUtilities::validALCT(const CSCCorrelatedLCTDigi& CorrLCT)
{
  int quality = CorrLCT.getQuality();
  return (quality != 4 && quality != 5);
}

///Taken from CSCTrigger/L1MuCSCCorrelatedLCT
bool CSCDigiUtilities::accelerator(const CSCCorrelatedLCTDigi& CorrLCT)
{
  int quality = CorrLCT.getQuality();
  return (quality == 1 || quality == 2);
}

///Taken from CSCTrigger/L1MuCSCCorrelatedLCT
bool CSCDigiUtilities::accelerator(const CSCALCTDigi& aLCT)
{/// This needs a reality check too!!! 23/2/05
  return ((aLCT.getPattern() & 0x1) == 0);
}
