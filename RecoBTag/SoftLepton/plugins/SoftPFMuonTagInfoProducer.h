#ifndef RecoBTag_SoftLepton_SoftPFMuonTagInfoProducer_h
#define RecoBTag_SoftLepton_SoftPFMuonTagInfoProducer_h


#include <vector>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidate.h>
#include "DataFormats/JetReco/interface/PFJetCollection.h"
// Muons
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "RecoEgamma/EgammaTools/interface/ConversionTools.h" 

// Vertex
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
// SoftPFMuonTagInfoProducer:  the SoftPFMuonTagInfoProducer takes
// a PFCandidateCollection as input and produces a RefVector
// to the likely soft muons in this collection.

class SoftPFMuonTagInfoProducer : public edm::EDProducer
{

  public:

    SoftPFMuonTagInfoProducer (const edm::ParameterSet& conf);
    ~SoftPFMuonTagInfoProducer();
    reco::SoftLeptonTagInfo tagMuon (const edm::RefToBase<reco::Jet> &, reco::PFCandidateCollection &) ;
  private:

    virtual void produce(edm::Event&, const edm::EventSetup&);
    reco::SoftLeptonProperties fillMuonProperties(const reco::PFCandidate&, const reco::Jet&);	 
    bool isMuonClean(edm::Event&,const reco::PFCandidate*);
    
    bool  isLooseMuon(const reco::Muon*);
    bool  isSoftMuon (const reco::Muon*);
    bool  isTightMuon(const reco::Muon*);	
	
    // service used to make transient tracks from tracks
    const TransientTrackBuilder* transientTrackBuilder;
    edm::InputTag PVerTag_,PFJet_;
    bool goodvertex;
    int MuonId_,muonId;

    const reco::Vertex* vertex;

};


#endif
