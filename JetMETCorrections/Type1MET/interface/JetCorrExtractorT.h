#ifndef JetMETCorrections_Type1MET_JetCorrExtractorT_h
#define JetMETCorrections_Type1MET_JetCorrExtractorT_h

/** \class JetCorrExtractorT
 *
 * Retrieve jet energy correction factor for
 *  o reco::CaloJets
 *  o reco::PFJets
 *  o pat::Jets (of either PF-type or Calo-type)
 *
 * NOTE: this "general" implementation is to be used for reco::CaloJets and reco::PFJets, **not** for pat::Jets
 *
 * \author Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace jetcorrextractor
{
  // never heard of copysign?
  inline double sign(double x)
  {
    if      ( x > 0. ) return +1.;
    else if ( x < 0. ) return -1.;
    else               return  0.;
  }
}

template <typename T>
class JetCorrExtractorT
{
 public:

  reco::Candidate::LorentzVector operator()(const T& rawJet, const reco::JetCorrector* jetCorr,
					    double jetCorrEtaMax = 9.9,
					    const reco::Candidate::LorentzVector * const rawJetP4_specified = nullptr) const
  {

    // allow to specify four-vector to be used as "raw" (uncorrected) jet momentum,
    // call 'rawJet.p4()' in case four-vector not specified explicitely
    reco::Candidate::LorentzVector rawJetP4 = ( rawJetP4_specified ) ?
      (*rawJetP4_specified) : rawJet.p4();

    double jetCorrFactor = 1.;
    if ( fabs(rawJetP4.eta()) < jetCorrEtaMax ) {
      jetCorrFactor = getCorrection(rawJet, jetCorr);
    } else {
      reco::Candidate::PolarLorentzVector modJetPolarP4(rawJetP4);
      modJetPolarP4.SetEta(jetcorrextractor::sign(rawJetP4.eta())*jetCorrEtaMax);

      reco::Candidate::LorentzVector modJetP4(modJetPolarP4);

      T modJet(rawJet);
      modJet.setP4(modJetP4);

      jetCorrFactor = getCorrection(modJet, jetCorr);
      if(jetCorrFactor<0) {
	edm::LogWarning("JetCorrExtractor") << "Negative jet energy scale correction noticed" << ".\n";
      }
    }

    reco::Candidate::LorentzVector corrJetP4 = rawJetP4;
    corrJetP4 *= jetCorrFactor;

    return corrJetP4;
  }

  reco::Candidate::LorentzVector operator()(const T& rawJet, const std::string& jetCorrLabel,
					    double jetCorrEtaMax = 9.9,
					    const reco::Candidate::LorentzVector * const rawJetP4_specified = nullptr) const
  {
    edm::LogWarning("JetCorrExtractor") << "JetCorrExtractorT<T>::operator(const T&, const std::string&, ...) is deprecated.\n"
      << "Please use JetCorrExtractorT<T>::operator(const T&, const reco::JetCorrector*, ...) instead.\n"
      << "Jet remains uncorrected!";
    reco::Candidate::LorentzVector rawJetP4 = ( rawJetP4_specified ) ?
      (*rawJetP4_specified) : rawJet.p4();
    return rawJetP4;
  }

    private:

  static double getCorrection(const T& jet, const reco::JetCorrector* jetCorr)
  {
    return jetCorr->correction(jet);
  }


};

#endif
