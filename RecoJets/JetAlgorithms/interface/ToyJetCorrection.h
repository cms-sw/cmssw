#ifndef JetAlgorithms_ToyJetCorrection_h
#define JetAlgorithms_ToyJetCorrection_h

/** \class ToyJetCorrection
      Template algorithm to correct jet
    \author F.Ratnikov (UMd)
    Mar 2, 2006
    $Id: ToyJetCorrection.h,v 1.3 2007/04/18 22:04:31 fedor Exp $
 */

#include "DataFormats/JetReco/interface/CaloJet.h"

class ToyJetCorrection {
 public:
  /** Constructor
   \param fScale Scale factor for jet correction
  */
  ToyJetCorrection (double fScale = 1.) : mScale (fScale) {}
  ~ToyJetCorrection () {}
  reco::CaloJet applyCorrection (const reco::CaloJet& fJet);
 private:
  double mScale;
};

#endif
