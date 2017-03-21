#ifndef PhysicsTools_TagAndProbe_ElectronMatchedCandidateProducer_h
#define PhysicsTools_TagAndProbe_ElectronMatchedCandidateProducer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"


// forward declarations

class ElectronMatchedCandidateProducer : public edm::EDProducer
{
 public:
  explicit ElectronMatchedCandidateProducer(const edm::ParameterSet&);
  ~ElectronMatchedCandidateProducer();

 private:
  virtual void beginJob() override ;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob()  override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<reco::GsfElectron> > electronCollectionToken_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > scCollectionToken_;
  double delRMatchingCut_;
};

#endif
