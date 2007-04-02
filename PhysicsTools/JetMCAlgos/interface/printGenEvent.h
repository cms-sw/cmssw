#ifndef printGenEvent_H
#define printGenEvent_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

class printGenEvent : public edm::EDAnalyzer {
  public:
    explicit printGenEvent(const edm::ParameterSet & );
    ~printGenEvent() {};
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  private:

    edm::InputTag source_;

    edm::Handle<reco::CandidateCollection> particles;
    edm::Handle<reco::CandidateCollection> genJets;

    edm::ESHandle<ParticleDataTable> pdt_;

};

#endif
