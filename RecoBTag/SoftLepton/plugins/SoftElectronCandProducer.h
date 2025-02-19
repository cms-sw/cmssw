#ifndef RecoBTag_SoftLepton_SoftElectronCandProducer_h
#define RecoBTag_SoftLepton_SoftElectronCandProducer_h


#include <vector>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

// SoftElectronCandProducer:  the SoftElectronCandProducer takes
// a PFCandidateCollection as input and produces a ValueMap
// to point out the likely soft electrons in this collection.

class SoftElectronCandProducer : public edm::EDProducer
{

  public:

    SoftElectronCandProducer (const edm::ParameterSet& conf);
    ~SoftElectronCandProducer();

  private:

    virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);
    bool isClean(const reco::GsfElectron& gsfcandidate);

    edm::InputTag gsfElectronTag_;

    std::vector<double> barrelPtCuts_;
    std::vector<double> barreldRGsfTrackElectronCuts_;
    std::vector<double> barrelEemPinRatioCuts_;
    std::vector<double> barrelMVACuts_;

    std::vector<double> forwardPtCuts_;
    std::vector<double> forwardInverseFBremCuts_;
    std::vector<double> forwarddRGsfTrackElectronCuts_;
    std::vector<double> forwardMVACuts_;
};

#endif
