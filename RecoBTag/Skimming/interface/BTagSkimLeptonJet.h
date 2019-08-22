#ifndef BTagSkimLeptonJet_h
#define BTagSkimLeptonJet_h

/** \class BtagSkimLeptonJet
 *
 *
 *
 *
 * \author Francisco Yumiceva, FERMILAB
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class BTagSkimLeptonJet : public edm::EDFilter {
public:
  explicit BTagSkimLeptonJet(const edm::ParameterSet&);
  ~BTagSkimLeptonJet() override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  edm::InputTag CaloJetInput_;
  double MinCaloJetPt_;
  double MaxCaloJetEta_;
  int MinNLeptonJet_;
  std::string LeptonType_;
  edm::InputTag LeptonInput_;
  double MinLeptonPt_;
  double MaxLeptonEta_;
  double MaxDeltaR_;
  double MinPtRel_;

  unsigned int nEvents_;
  unsigned int nAccepted_;

  class PtSorter {
  public:
    template <class T>
    bool operator()(const T& a, const T& b) {
      return (a.pt() > b.pt());
    }
  };
};

#endif
