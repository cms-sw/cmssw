#ifndef IsolatedTauJetSelector_H
#define IsolatedTauJetSelector_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"


class IsolatedTauJetsSelector: public edm::EDProducer {
 public:
  explicit IsolatedTauJetsSelector(const edm::ParameterSet&);
  ~IsolatedTauJetsSelector();
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  typedef std::vector<edm::InputTag> vtag;
  vtag jetSrc;
  std::vector<edm::EDGetTokenT<reco::IsolatedTauTagInfoCollection> > jetSrcToken;
  double pt_min_leadTrack;
  bool useInHLTOpen;
  bool useIsolationDiscriminator;
  

};
#endif
