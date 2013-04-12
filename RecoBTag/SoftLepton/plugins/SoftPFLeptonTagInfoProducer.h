#ifndef RecoBTag_SoftLepton_SoftPFLeptonTagInfoProducer_h
#define RecoBTag_SoftLepton_SoftPFLeptonTagInfoProducer_h


#include <vector>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
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
// SoftPFLeptonTagInfoProducer:  the SoftPFLeptonTagInfoProducer takes
// a PFCandidateCollection as input and produces a RefVector
// to the likely soft electrons in this collection.

class SoftPFLeptonTagInfoProducer : public edm::EDProducer
{

  public:

    SoftPFLeptonTagInfoProducer (const edm::ParameterSet& conf);
    ~SoftPFLeptonTagInfoProducer();
    reco::SoftLeptonTagInfo tagElec (const edm::RefToBase<reco::Jet> &, reco::PFCandidateCollection &) ;
    reco::SoftLeptonTagInfo tagMuon (const edm::RefToBase<reco::Jet> &, reco::PFCandidateCollection &) ;
  private:

    virtual void produce(edm::Event&, const edm::EventSetup&);
    reco::SoftLeptonProperties fillElecProperties(const reco::PFCandidate&, const reco::Jet&);
    reco::SoftLeptonProperties fillMuonProperties(const reco::PFCandidate&, const reco::Jet&);	 
    bool isMuonClean(edm::Event&,const reco::PFCandidate*);
    bool isElecClean(edm::Event&,const reco::PFCandidate*);
    
    bool  isLooseMuon(const reco::Muon*);
    bool  isSoftMuon (const reco::Muon*);
    bool  isTightMuon(const reco::Muon*);	
	
    // service used to make transient tracks from tracks
    const TransientTrackBuilder* transientTrackBuilder;
    edm::InputTag PVerTag_,PFJet_;
    bool goodvertex;
    int MuonId_,muonId;
    std::string SPFELabel_ ,SPFMLabel_,SPFETILabel_ ,SPFMTILabel_;

    const reco::Vertex* vertex;

};


#endif
