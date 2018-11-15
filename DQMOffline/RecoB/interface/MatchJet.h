#ifndef MatchJet_H
#define MatchJet_H

#include <vector>
#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DQMOffline/RecoB/interface/CorrectJet.h"

/** \class MatchJet
 *
 *  Match jets
 *
 */

class MatchJet {

 public:
  MatchJet() {}
  MatchJet(const edm::ParameterSet& pSet);

  void setThreshold(const double& energy) { threshold = energy; }

  /// match the collections
  void matchCollections(const edm::RefToBaseVector<reco::Jet> & refJets,
                        const edm::RefToBaseVector<reco::Jet> & recJets,
			const reco::JetCorrector * corrector
                        );

  /// Returns the matched "reference" jet
  edm::RefToBase<reco::Jet>
  operator() (const edm::RefToBase<reco::Jet> & recJet) const;

 private:
  std::vector<int> refToRec, recToRef;
  edm::RefToBaseVector<reco::Jet> refJets;
  edm::RefToBaseVector<reco::Jet> recJets;

  CorrectJet refJetCorrector;
  CorrectJet recJetCorrector;

  double maxChi2;
  double sigmaDeltaR2;
  double sigmaDeltaE2;
  double threshold;
};

#endif
