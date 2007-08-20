#ifndef JetAlgorithms_ToyJetCorrection_h
#define JetAlgorithms_ToyJetCorrection_h

/** \class ToyJetCorrection
      Template algorithm to correct jet
    \author F.Ratnikov (UMd)
    Mar 2, 2006
    $Id: ToyJetCorrection.h,v 1.4 2007/08/15 17:43:14 fedor Exp $
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
