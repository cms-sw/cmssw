#ifndef L3MuonIsolationProducer_L3MuonSumCaloPFIsolationProducer_h
#define L3MuonIsolationProducer_L3MuonSumCaloPFIsolationProducer_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateIsolation.h"



namespace edm {
    class ConfigurationDescriptions;
}

class L3MuonSumCaloPFIsolationProducer : public edm::global::EDProducer<> {
public:
    explicit L3MuonSumCaloPFIsolationProducer(const edm::ParameterSet&);
    ~L3MuonSumCaloPFIsolationProducer();
    
    virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    
private:
    
    edm::EDGetTokenT<reco::RecoChargedCandidateCollection> recoChargedCandidateProducer_;
    edm::EDGetTokenT<reco::RecoChargedCandidateIsolationMap> pfEcalClusterProducer_;
    edm::EDGetTokenT<reco::RecoChargedCandidateIsolationMap> pfHcalClusterProducer_;
    

};

#endif
