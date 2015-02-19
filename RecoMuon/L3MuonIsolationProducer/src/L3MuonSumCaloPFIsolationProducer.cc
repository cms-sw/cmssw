#include "L3MuonSumCaloPFIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateIsolation.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;

/// constructor with config
L3MuonSumCaloPFIsolationProducer::L3MuonSumCaloPFIsolationProducer(const edm::ParameterSet& config) {
    
    recoChargedCandidateProducer_ = consumes<reco::RecoChargedCandidateCollection>(config.getParameter<edm::InputTag>("recoChargedCandidateProducer"));
    pfEcalClusterProducer_         = consumes<reco::RecoChargedCandidateIsolationMap>(config.getParameter<edm::InputTag>("pfEcalClusterProducer"));
    pfHcalClusterProducer_         = consumes<reco::RecoChargedCandidateIsolationMap>(config.getParameter<edm::InputTag>("pfHcalClusterProducer"));
    
    produces < edm::ValueMap<float> >();
    
}

L3MuonSumCaloPFIsolationProducer::~L3MuonSumCaloPFIsolationProducer()
{}

void L3MuonSumCaloPFIsolationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("recoChargedCandidateProducer", edm::InputTag("hltL1SeededRecoChargedCandidatePF"));
    desc.add<edm::InputTag>("pfEcalClusterProducer", edm::InputTag("hltParticleFlowClusterECAL"));
    desc.add<edm::InputTag>("pfHcalClusterProducer", edm::InputTag("hltParticleFlowClusterHCAL"));
    descriptions.add(("hltL3MuonSumCaloPFIsolationProducer"), desc);
}

void L3MuonSumCaloPFIsolationProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const{
    
    edm::Handle<reco::RecoChargedCandidateCollection> recochargedcandHandle;
    iEvent.getByToken(recoChargedCandidateProducer_,recochargedcandHandle);
    
    edm::Handle<reco::RecoChargedCandidateIsolationMap> ecalIsolation;
    iEvent.getByToken (pfEcalClusterProducer_,ecalIsolation);
    
    edm::Handle<reco::RecoChargedCandidateIsolationMap> hcalIsolation;
    iEvent.getByToken (pfHcalClusterProducer_,hcalIsolation);
    
    std::auto_ptr<edm::ValueMap<float> > caloIsoMap( new edm::ValueMap<float> ());
    std::vector<float> isoFloats(recochargedcandHandle->size(), 0);
    
    for (unsigned int iReco = 0; iReco < recochargedcandHandle->size(); iReco++) {
        reco::RecoChargedCandidateRef candRef(recochargedcandHandle, iReco);
        reco::RecoChargedCandidateIsolationMap::const_iterator mapiECAL = (*ecalIsolation).find( candRef );
        float valisoECAL = mapiECAL->val;
        reco::RecoChargedCandidateIsolationMap::const_iterator mapiHCAL = (*hcalIsolation).find( candRef );
        float valisoHCAL = mapiHCAL->val;
        float caloIso = valisoECAL + valisoHCAL;
        isoFloats[iReco] = caloIso;
    }
    
    edm::ValueMap<float> ::Filler isoFloatFiller(*caloIsoMap);
    isoFloatFiller.insert(recochargedcandHandle, isoFloats.begin(), isoFloats.end());
    isoFloatFiller.fill();
    iEvent.put(caloIsoMap);

}
