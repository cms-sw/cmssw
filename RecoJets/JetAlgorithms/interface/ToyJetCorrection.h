#ifndef JetAlgorithms_ToyJetCorrection_h
#define JetAlgorithms_ToyJetCorrection_h

/* Template algorithm to correct jet
    F.Ratnikov (UMd)
    Mar 2, 2006
*/

#include "DataFormats/JetReco/interface/CaloJetfwd.h"

class ToyJetCorrection {
 public:
  ToyJetCorrection (double fScale = 1.) : mScale (fScale) {}
  ~ToyJetCorrection () {}
  reco::CaloJet applyCorrection (const reco::CaloJet& fJet);
 private:
  double mScale;
};

#endif
