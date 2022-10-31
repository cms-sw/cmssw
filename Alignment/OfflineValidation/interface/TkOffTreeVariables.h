#ifndef Alignment_OfflineValidation_TkOffTreeVariables_h
#define Alignment_OfflineValidation_TkOffTreeVariables_h

#include <string>
// For ROOT types with '_t':
#include "Rtypes.h"

/// container to hold data to be written into TTree
struct TkOffTreeVariables {
  /// constructor initialises to empty values
  TkOffTreeVariables() { this->clear(); }

  /// set to empty values
  void clear() {
    // First clear things that are changing if TTrees are merged:
    this->clearMergeAffectedPart();

    // Now the rest:
    // Float_t's
    posR = posPhi = posEta = posX = posY = posZ = rDirection = phiDirection = zDirection = rOrZDirection = 0.;
    // Int_t's
    moduleId = subDetId = layer = side = half = rod = ring = petal = blade = panel = outerInner = module = 0;
    // Bool_t's
    isDoubleSide = isStereo = false;
    // std::string's
    histNameLocalX = histNameNormLocalX = histNameLocalY /* = histNameNormLocalY */
        = histNameX = histNameNormX = histNameY = histNameNormY = "";
    profileNameResXvsX = profileNameResXvsY = profileNameResYvsX = profileNameResYvsY = "";
  }
  /// set those values to empty that are affected by merging
  void clearMergeAffectedPart() {
    // variable Float_t's
    meanLocalX = meanNormLocalX = meanX = meanNormX = meanY = meanNormY = medianX = medianY = chi2PerDofX =
        chi2PerDofY = rmsLocalX = rmsNormLocalX = rmsX = rmsNormX = rmsY = rmsNormY = sigmaX = sigmaNormX = fitMeanX =
            fitSigmaX = fitMeanNormX = fitSigmaNormX = fitMeanY = fitSigmaY = fitMeanNormY = fitSigmaNormY =
                numberOfUnderflows = numberOfOverflows = numberOfOutliers = 0.;

    meanResXvsX = meanResXvsY = meanResYvsX = meanResYvsY = rmsResXvsX = rmsResXvsY = rmsResYvsX = rmsResYvsY = 0.;

    // variable Int_t's
    entries = 0;
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Data members:
  // They do not follow convention to have '_' at the end since they will appear
  // as such in the TTree and that is ugly.
  ///////////////////////////////////////////////////////////////////////////////

  Float_t meanLocalX, meanNormLocalX, meanX, meanNormX,                    //mean value read out from module histograms
      meanY, meanNormY, medianX, medianY,                                  //median read out from module histograms
      chi2PerDofX, chi2PerDofY, rmsLocalX, rmsNormLocalX, rmsX, rmsNormX,  //rms value read out from modul histograms
      rmsY, rmsNormY, sigmaX, sigmaNormX, fitMeanX, fitSigmaX, fitMeanNormX, fitSigmaNormX, fitMeanY, fitSigmaY,
      fitMeanNormY, fitSigmaNormY, posR, posPhi, posEta,  //global coordiantes
      posX, posY, posZ,                                   //global coordiantes
      numberOfUnderflows, numberOfOverflows, numberOfOutliers, rDirection, phiDirection, zDirection, rOrZDirection;

  UInt_t entries;              // number of entries for each module
  UInt_t moduleId, subDetId,   //moduleId == detId
      layer, side, half, rod,  // half = TPB: halfBarrel, TPE: halfCylinder, TIB: halfShell
      ring, petal, blade, panel, outerInner,
      module;  //orientation of modules in TIB:1/2= int/ext string, TID:1/2=back/front ring, TEC 1/2=back/front petal

  Bool_t isDoubleSide;  // (!isDoubleSide) is a detUnit, (isDoubleSide) is a Det (glued Modules)
  Bool_t isStereo;      // (!isStereo) is a rPhi-module, (isStereo) is the stereo module from a Det

  std::string histNameLocalX, histNameNormLocalX, histNameLocalY; /* histNameNormLocalY, */
  std::string histNameX, histNameNormX, histNameY, histNameNormY;

  Float_t meanResXvsX, meanResXvsY, meanResYvsX, meanResYvsY;
  Float_t rmsResXvsX, rmsResXvsY, rmsResYvsX, rmsResYvsY;

  std::string profileNameResXvsX, profileNameResXvsY, profileNameResYvsX, profileNameResYvsY;
};

#endif
