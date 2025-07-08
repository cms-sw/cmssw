#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/transform.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TLorentzVector.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <vector>
#include <iostream>

class GenParticlesExtendedTableProducer : public edm::stream::EDProducer<> {
protected:
  const edm::InputTag genparticlesLabel;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> genparticlesTag_;
  
public:
  GenParticlesExtendedTableProducer(edm::ParameterSet const& params)
    :
    genparticlesLabel(params.getParameter<edm::InputTag>("genparticles")),
    genparticlesTag_(consumes<std::vector<reco::GenParticle>>(genparticlesLabel))
    {
    produces<nanoaod::FlatTable>("GenPart");
    
  }

  ~GenParticlesExtendedTableProducer() override {}

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    

    edm::Handle<std::vector<reco::GenParticle>> genparticlesHandle;
    iEvent.getByToken(genparticlesTag_, genparticlesHandle);
    auto genparticlesTab = std::make_unique<nanoaod::FlatTable>(genparticlesHandle->size(), "GenPart", false, true);

    std::vector<float> vx,vy,vz;
    std::vector<float> px,py,pz;    

    for(auto genparticle : *genparticlesHandle) {

      vx.push_back(genparticle.vx());
      vy.push_back(genparticle.vy());
      vz.push_back(genparticle.vz());

      px.push_back(genparticle.px());
      py.push_back(genparticle.py());
      pz.push_back(genparticle.pz());
    }

    genparticlesTab->addColumn<float>("vx", vx, "");
    genparticlesTab->addColumn<float>("vy", vy, "");
    genparticlesTab->addColumn<float>("vz", vz, "");

    genparticlesTab->addColumn<float>("px", px, "");
    genparticlesTab->addColumn<float>("py", py, "");
    genparticlesTab->addColumn<float>("pz", pz, "");

    iEvent.put(std::move(genparticlesTab), "GenPart");
  }

};
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GenParticlesExtendedTableProducer);
