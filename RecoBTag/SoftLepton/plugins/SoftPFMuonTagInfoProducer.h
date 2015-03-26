// * Author: Alberto Zucchetta
// * Mail: a.zucchetta@cern.ch
// * January 16, 2015

#ifndef RecoBTag_SoftLepton_SoftPFMuonTagInfoProducer_h
#define RecoBTag_SoftLepton_SoftPFMuonTagInfoProducer_h


#include <vector>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"

class SoftPFMuonTagInfoProducer : public edm::stream::EDProducer<> {

  public:

    SoftPFMuonTagInfoProducer(const edm::ParameterSet& conf);
    ~SoftPFMuonTagInfoProducer();
    
  private:

    virtual void produce(edm::Event&, const edm::EventSetup&);
    virtual float boostedPPar(const math::XYZVector&, const math::XYZVector&);

    edm::EDGetTokenT<edm::View<reco::Jet> > jetToken;
    edm::EDGetTokenT<edm::View<reco::Muon> > muonToken;
    edm::EDGetTokenT<reco::VertexCollection> vertexToken;
    float pTcut, SIPcut, IPcut, ratio1cut, ratio2cut;
    bool useFilter;
};


#endif
