#ifndef JetAlgorithms_ToyJetCorrection_h
#define JetAlgorithms_ToyJetCorrection_h

/* Template algorithm to correct jet
    F.Ratnikov (UMd)
    Mar 2, 2006
*/

class CaloJet;

class ToyJetCorrection {
 public:
  ToyJetCorrection (double fScale = 1.) : mScale (fScale) {}
  ~ToyJetCorrection () {}
  CaloJet applyCorrection (const CaloJet& fJet);
 private:
  double mScale;
};

#endif
