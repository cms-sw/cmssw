#ifndef CorrectJet_H
#define CorrectJet_H

#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

/** \class CorrectJet
 *
 *  Correct jets
 *
 */

class CorrectJet {

 public:
  CorrectJet() {}

  /// Returns the corrected jet
  void setCorrector(const reco::JetCorrector *corrector) {m_corrector=corrector;}
  reco::Jet operator() (const reco::Jet & jet) const
  {
    reco::Jet correctedJet(jet.p4(), jet.vertex());
    if (m_corrector)
      correctedJet.scaleEnergy(m_corrector->correction(jet));
    return correctedJet;
  }

 private:
  const reco::JetCorrector *m_corrector;
};

#endif
