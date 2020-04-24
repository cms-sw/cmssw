#include <string>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/libminifloat.h"

namespace pat {
  class PATVertexSlimmer : public edm::global::EDProducer<> {
  public:
    explicit PATVertexSlimmer(const edm::ParameterSet&);
    ~PATVertexSlimmer() override;
    
    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  private:
    const edm::EDGetTokenT<std::vector<reco::Vertex> > src_;
    const edm::EDGetTokenT<edm::ValueMap<float> > score_;
    const bool rekeyScores_;
  };
}

pat::PATVertexSlimmer::PATVertexSlimmer(const edm::ParameterSet& iConfig) :
    src_(consumes<std::vector<reco::Vertex> >(iConfig.getParameter<edm::InputTag>("src"))),
    score_(mayConsume<edm::ValueMap<float>>(iConfig.existsAs<edm::InputTag>("score")?iConfig.getParameter<edm::InputTag>("score"):edm::InputTag())),
    rekeyScores_(iConfig.existsAs<edm::InputTag>("score"))
{
  produces<std::vector<reco::Vertex> >();
  if(rekeyScores_) produces<edm::ValueMap<float> >();
}

pat::PATVertexSlimmer::~PATVertexSlimmer() {}

void pat::PATVertexSlimmer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
    edm::Handle<std::vector<reco::Vertex> > vertices;
    iEvent.getByToken(src_, vertices);
    auto outPtr = std::make_unique<std::vector<reco::Vertex>>();

    outPtr->reserve(vertices->size());
    for (unsigned int i = 0, n = vertices->size(); i < n; ++i) {
        const reco::Vertex &v = (*vertices)[i];
        auto co = v.covariance4D();
        if(i>0) {
          for(size_t j=0;j<4;j++){
            for(size_t k=j;k<4;k++){
              co(j,k) = MiniFloatConverter::reduceMantissaToNbits<10>( co(j,k) );
            }
          }
        }
        outPtr->push_back(reco::Vertex(v.position(), co, v.t(), v.chi2(), v.ndof(), 0));
    }

    auto oh = iEvent.put(std::move(outPtr));
    if(rekeyScores_) {
      edm::Handle<edm::ValueMap<float> > scores;
      iEvent.getByToken(score_, scores);
      auto vertexScoreOutput = std::make_unique<edm::ValueMap<float>>();
      edm::ValueMap<float>::const_iterator idIt=scores->begin();
      for(;idIt!=scores->end();idIt++) {
          if(idIt.id() ==  vertices.id()) break;
      }
      // std::find_if(scores->begin(), scores->end(), [vertices] (const edm::ValueMap<float>::const_iterator& s) { return s.id() == vertices.id(); } );
      edm::ValueMap<float>::Filler vertexScoreFiller(*vertexScoreOutput);
      vertexScoreFiller.insert(oh,idIt.begin(),idIt.end());
      vertexScoreFiller.fill();
      iEvent.put(std::move(vertexScoreOutput));
    }
}

using pat::PATVertexSlimmer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATVertexSlimmer);
