#ifndef RecoTauTag_RecoTau_PFTauPrimaryVertexProducerBase_H_
#define RecoTauTag_RecoTau_PFTauPrimaryVertexProducerBase_H_

/* class PFTauPrimaryVertexProducerBase
 * EDProducer of the 
 * authors: Ian M. Nugent
 * This work is based on the impact parameter work by Rosamaria Venditti and reconstructing the 3 prong taus.
 * The idea of the fully reconstructing the tau using a kinematic fit comes from
 * Lars Perchalla and Philip Sauerland Theses under Achim Stahl supervision. This
 * work was continued by Ian M. Nugent and Vladimir Cherepanov.
 * Thanks goes to Christian Veelken and Evan Klose Friis for their help and suggestions.
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"

#include <TFormula.h>

class PFTauPrimaryVertexProducerBase : public edm::stream::EDProducer<> {
public:
  enum Alg { useInputPV = 0, useFrontPV };

  struct DiscCutPair {
    DiscCutPair() : discr_(nullptr), cutFormula_(nullptr) {}
    ~DiscCutPair() { delete cutFormula_; }
    const reco::PFTauDiscriminator* discr_;
    edm::EDGetTokenT<reco::PFTauDiscriminator> inputToken_;
    double cut_;
    TFormula* cutFormula_;
  };
  typedef std::vector<DiscCutPair*> DiscCutPairVec;

  explicit PFTauPrimaryVertexProducerBase(const edm::ParameterSet& iConfig);
  ~PFTauPrimaryVertexProducerBase() override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  static edm::ParameterSetDescription getDescriptionsBase();

  // called at the beginning of every event - override if necessary
  virtual void beginEvent(const edm::Event&, const edm::EventSetup&) {}

protected:
  // abstract function implemented in derived classes
  virtual void nonTauTracksInPV(const reco::VertexRef&,
                                const std::vector<edm::Ptr<reco::TrackBase> >&,
                                std::vector<const reco::Track*>&) = 0;

private:
  edm::EDGetTokenT<std::vector<reco::PFTau> > pftauToken_;
  edm::EDGetTokenT<edm::View<reco::Electron> > electronToken_;
  edm::EDGetTokenT<edm::View<reco::Muon> > muonToken_;
  edm::EDGetTokenT<reco::VertexCollection> pvToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  int algorithm_;
  edm::ParameterSet qualityCutsPSet_;
  bool useBeamSpot_;
  bool useSelectedTaus_;
  bool removeMuonTracks_;
  bool removeElectronTracks_;
  DiscCutPairVec discriminators_;
  std::unique_ptr<StringCutObjectSelector<reco::PFTau> > cut_;
  std::unique_ptr<reco::tau::RecoTauVertexAssociator> vertexAssociator_;
};

#endif
