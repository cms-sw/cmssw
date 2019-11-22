
#ifndef GEDGsfElectronProducer_h
#define GEDGsfElectronProducer_h

#include "GsfElectronBaseProducer.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include <map>

class GEDGsfElectronProducer : public GsfElectronBaseProducer {
public:
  explicit GEDGsfElectronProducer(const edm::ParameterSet&, const GsfElectronAlgo::HeavyObjectCache*);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<reco::PFCandidateCollection> egmPFCandidateCollection_;
  std::map<reco::GsfTrackRef, reco::GsfElectron::MvaInput> gsfMVAInputMap_;
  std::map<reco::GsfTrackRef, reco::GsfElectron::MvaOutput> gsfMVAOutputMap_;

private:
  void fillGsfElectronValueMap(edm::Event& event, edm::ValueMap<reco::GsfElectronRef>::Filler& filler);
  void matchWithPFCandidates(edm::Event& event);
  void setMVAOutputs(reco::GsfElectronCollection& electrons,
                     const GsfElectronAlgo::HeavyObjectCache*,
                     const std::map<reco::GsfTrackRef, reco::GsfElectron::MvaOutput>& mvaOutputs,
                     reco::VertexCollection const& vertices) const;
};

#endif
