#include "DQM/PhysicsHWW/interface/MITConversionUtilities.h"
#include "TMath.h"

namespace HWWFunctions {

  bool isMITConversion(HWW& hww, unsigned int elidx, 
           int nWrongHitsMax, 
           float probMin,
           float dlMin,
           bool matchCTF,
           bool requireArbitratedMerged) {

    unsigned int nconvs = hww.convs_isConverted().size();
    if(nconvs == 0) 
      return false;
    bool isGoodConversion = false;

    for(unsigned int iconv = 0; iconv < nconvs; iconv++) {
      
      bool conversionMatchFound = false;
      for(unsigned int itk = 0; itk < hww.convs_tkidx().at(iconv).size(); itk++) {

        if(hww.convs_tkalgo().at(iconv)[itk] == 29 && hww.convs_tkidx().at(iconv)[itk] == hww.els_gsftrkidx().at(elidx))
    conversionMatchFound = true;
        if(matchCTF) {
    if(hww.convs_tkalgo().at(iconv)[itk] > 3 && hww.convs_tkalgo().at(iconv)[itk] < 14 && hww.convs_tkalgo().at(iconv)[itk] != 12 && hww.convs_tkidx().at(iconv)[itk] == hww.els_trkidx().at(elidx))
      conversionMatchFound = true;
        }
      
        if(conversionMatchFound)
    break;
      }
      
      
      if(conversionMatchFound==false)
        continue;
      
      if( TMath::Prob( hww.convs_chi2().at(iconv), (Int_t)hww.convs_ndof().at(iconv) )  > probMin && hww.convs_dl().at(iconv) > dlMin ) isGoodConversion = true;
      if(requireArbitratedMerged) {
        if(hww.convs_quality().at(iconv) & 4)
    isGoodConversion = true;
        else 
    isGoodConversion = false;
      }

      for(unsigned int j = 0; j < hww.convs_nHitsBeforeVtx().at(iconv).size(); j++) {
        if(hww.convs_nHitsBeforeVtx().at(iconv)[j] > nWrongHitsMax)
    isGoodConversion = false;
      }
        
      if(isGoodConversion)
        break;
        
        
    }//loop over convserions


    return isGoodConversion;
  }
}
