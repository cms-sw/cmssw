#ifndef IsolatedTauJetSelector_H
#define IsolatedTauJetSelector_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"



class IsolatedTauJetsSelector: public edm::EDProducer {
 public:
  explicit IsolatedTauJetsSelector(const edm::ParameterSet&);
  ~IsolatedTauJetsSelector();
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  typedef std::vector<edm::InputTag> vtag;
  vtag jetSrc;
 double matching_cone;
 double signal_cone;
 double isolation_cone;
 double pt_min_isolation;
 double pt_min_leadTrack;
 double dZ_vertex;
 int n_tracks_isolation_ring;

 
  

};
#endif
