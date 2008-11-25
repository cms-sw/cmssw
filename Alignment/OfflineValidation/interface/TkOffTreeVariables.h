#ifndef Alignment_OfflineValidation_TkOffTreeVariables_h
#define Alignment_OfflineValidation_TkOffTreeVariables_h

#include <string>
// For ROOT types with '_t':
#include <Rtypes.h>

/// container to hold data to be written into TTree
struct TkOffTreeVariables
{
  /// constructor initialises to empty values
  TkOffTreeVariables()  { this->clear();}

  /// set to empty values
  void clear() {
    // First clear things that are changing if TTrees are merged:
    this->clearMergeAffectedPart();

    // Now the rest:
    // Float_t's
    posR = posPhi = posEta = posX = posY = posZ = 0.;
    // Int_t's
    moduleId = subDetId
      = layer = side = rod 
      = ring = petal 
      = blade = panel 
      = outerInner = 0;
    // Bool_t's
    isDoubleSide = false;
    // std::string's
    histNameLocalX = histNameNormLocalX 
      = histNameX = histNameNormX
      = histNameY = histNameNormY = "";
  }
  /// set those values to empty that are affected by merging
  void clearMergeAffectedPart()
  {
    // variable Float_t's
    meanLocalX = meanNormLocalX = meanX = meanNormX = meanY = meanNormY 
      = chi2PerDof 
      = rmsLocalX = rmsNormLocalX = rmsX = rmsNormX = rmsY = rmsNormY
      = sigmaX = sigmaNormX
      = fitMeanX = fitSigmaX = fitMeanNormX = fitSigmaNormX  
      = numberOfUnderflows = numberOfOverflows = numberOfOutliers = 0.;

    // variable Int_t's
    entries = 0;
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Data members:
  // They do not follow convention to have '_' at the end since they will appear 
  // as such in the TTree and that is ugly.
  ///////////////////////////////////////////////////////////////////////////////
  Float_t meanLocalX, meanNormLocalX, 
    meanX, meanNormX,    //mean value read out from modul histograms
    meanY, meanNormY, chi2PerDof,
    rmsLocalX, rmsNormLocalX, rmsX, rmsNormX,  //rms value read out from modul histograms
    rmsY, rmsNormY,sigmaX,sigmaNormX,
    fitMeanX, fitSigmaX, fitMeanNormX, fitSigmaNormX,
    posR, posPhi, posEta,                     //global coordiantes    
    posX, posY, posZ,             //global coordiantes 
    numberOfUnderflows, numberOfOverflows, numberOfOutliers;
  UInt_t  entries, moduleId, subDetId, //number of entries for each modul //modul Id = detId and subdetector Id
    layer, side, rod, 
    ring, petal, 
    blade, panel, 
    outerInner; //orientation of modules in TIB:1/2= int/ext string, TID:1/2=back/front ring, TEC 1/2=back/front petal 
  Bool_t isDoubleSide;
  std::string histNameLocalX, histNameNormLocalX, histNameX, histNameNormX,
    histNameY, histNameNormY;    
};
  
#endif
