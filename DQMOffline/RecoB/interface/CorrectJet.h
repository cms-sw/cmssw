#ifndef CorrectJet_H
#define CorrectJet_H

#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

/** \class CorrectJet
 *
 *  Correct jets
 *
 */

class CorrectJet {

 public:
  CorrectJet() {}
  CorrectJet(const std::string &corrector) :
  			m_corrector(0), m_correctorName(corrector) {}

  void setEventSetup(const edm::EventSetup & es)
  {
    if (!m_correctorName.empty())
      m_corrector = JetCorrector::getJetCorrector(m_correctorName, es);
    else
      m_corrector = 0;
  }

  /// Returns the corrected jet
  reco::Jet operator() (const reco::Jet & jet) const
  {
    reco::Jet correctedJet(jet.p4(), jet.vertex());
    if (m_corrector)
      correctedJet.scaleEnergy(m_corrector->correction(jet));
    return correctedJet;
  }

 private:
  const JetCorrector *m_corrector;
  std::string m_correctorName;
};

#endif
