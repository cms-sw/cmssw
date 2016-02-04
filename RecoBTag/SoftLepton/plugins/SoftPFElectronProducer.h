#ifndef RecoBTag_SoftLepton_SoftPFElectronProducer_h
#define RecoBTag_SoftLepton_SoftPFElectronProducer_h


#include <vector>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

// SoftPFElectronProducer:  the SoftPFElectronProducer takes
// a PFCandidateCollection as input and produces a RefVector
// to the likely soft electrons in this collection.

class SoftPFElectronProducer : public edm::EDProducer
{

  public:

    SoftPFElectronProducer (const edm::ParameterSet& conf);
    ~SoftPFElectronProducer();

  private:

    virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);
    bool isClean(const reco::GsfElectron& gsfcandidate);

    edm::InputTag gsfElectronTag_;

    std::vector<double> barrelPtCuts_;
    std::vector<double> barreldRGsfTrackElectronCuts_;
    std::vector<double> barrelEemPinRatioCuts_;
    std::vector<double> barrelMVACuts_;
    std::vector<double> barrelInversedRFirstLastHitCuts_;
    std::vector<double> barrelRadiusFirstHitCuts_;
    std::vector<double> barrelZFirstHitCuts_;

    std::vector<double> forwardPtCuts_;
    std::vector<double> forwardInverseFBremCuts_;
    std::vector<double> forwarddRGsfTrackElectronCuts_;
    std::vector<double> forwardRadiusFirstHitCuts_;
    std::vector<double> forwardZFirstHitCuts_;
    std::vector<double> forwardMVACuts_;
};

#endif
