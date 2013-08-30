#ifndef RecoBTag_SoftLepton_SoftPFElectronTagInfoProducer_h
#define RecoBTag_SoftLepton_SoftPFElectronTagInfoProducer_h


#include <vector>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidate.h>
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "RecoEgamma/EgammaTools/interface/ConversionTools.h" 

// Vertex
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
// SoftPFElectronTagInfoProducer:  the SoftPFElectronTagInfoProducer takes
// a PFCandidateCollection as input and produces a RefVector
// to the likely soft electrons in this collection.

class SoftPFElectronTagInfoProducer : public edm::EDProducer
{

  public:

    SoftPFElectronTagInfoProducer (const edm::ParameterSet& conf);
    ~SoftPFElectronTagInfoProducer();
    reco::SoftLeptonTagInfo tagElec (const edm::RefToBase<reco::Jet> &, reco::PFCandidateCollection &) ;
  private:

    virtual void produce(edm::Event&, const edm::EventSetup&);
    reco::SoftLeptonProperties fillElecProperties(const reco::PFCandidate&, const reco::Jet&);
    bool isElecClean(edm::Event&,const reco::PFCandidate*);
    
    // service used to make transient tracks from tracks
    const TransientTrackBuilder* transientTrackBuilder;
    edm::InputTag PVerTag_,PFJet_;
    bool goodvertex;

    const reco::Vertex* vertex;

};


#endif
