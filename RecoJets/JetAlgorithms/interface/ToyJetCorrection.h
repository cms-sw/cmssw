#ifndef JetAlgorithms_ToyJetCorrection_h
#define JetAlgorithms_ToyJetCorrection_h

/** \class ToyJetCorrection
      Template algorithm to correct jet
    \author F.Ratnikov (UMd)
    Mar 2, 2006
    $Id: ToyJetCorrection.h,v 1.6 2007/03/26 20:42:26 fedor Exp $
 */

#include "DataFormats/JetReco/interface/CaloJetfwd.h"

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
