#ifndef RecoBTag_SoftLepton_SoftLepton_h
#define RecoBTag_SoftLepton_SoftLepton_h

// -*- C++ -*-
//
// Package:    SoftLepton
// Class:      SoftLepton
//
/**\class SoftLepton SoftLepton.h RecoBTag/SoftLepton/plugin/SoftLepton.h

 Description: CMSSW EDProducer wrapper for sot lepton b tagging.

 Implementation:
     The actual tagging is performed by SoftLeptonAlgorithm.
*/
//
// Original Author:  fwyzard
//         Created:  Wed Oct 18 18:02:07 CEST 2006
//

// system include files
#include <memory>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"


class TransientTrackBuilder;

class SoftLepton : public edm::stream::EDProducer<> {
public:
  explicit SoftLepton(const edm::ParameterSet& iConfig);
  ~SoftLepton();

  struct TrackCompare :
    public std::binary_function<edm::RefToBase<reco::Track>,
                                edm::RefToBase<reco::Track>, bool> {
    inline bool operator () (const edm::RefToBase<reco::Track> &t1,
                             const edm::RefToBase<reco::Track> &t2) const
    { return t1.key() < t2.key();}
  };

  typedef std::map<unsigned int, float> LeptonIds;
  typedef std::map<edm::RefToBase<reco::Track>, LeptonIds, TrackCompare> Leptons;

  // generic interface, using a TrackRefVector for lepton tracks
  reco::SoftLeptonTagInfo tag (
      const edm::RefToBase<reco::Jet> & jet,
      const reco::TrackRefVector      & tracks,
      const Leptons                   & leptons,
      const reco::Vertex              & primaryVertex
  ) const;

protected:
  // generic interface, using a TrackRefVector for lepton tracks

  GlobalVector refineJetAxis (
      const edm::RefToBase<reco::Jet>   & jet,
      const reco::TrackRefVector        & tracks,
      const edm::RefToBase<reco::Track> & exclude = edm::RefToBase<reco::Track>()
  ) const;

  static double relativeEta(
      const math::XYZVector& vector,
      const math::XYZVector& axis
  );

  static double boostedPPar(
      const math::XYZVector& vector,
      const math::XYZVector& axis
  );

private:
  virtual void produce(edm::Event & event, const edm::EventSetup & setup);

  // configuration
  const edm::InputTag                                           m_jets;
  const edm::EDGetTokenT<reco::JetTracksAssociationCollection>  token_jtas;
  const edm::EDGetTokenT<edm::View<reco::Jet> >                 token_jets;
  const edm::EDGetTokenT<reco::VertexCollection>                token_primaryVertex;
  const edm::InputTag                                           m_leptons;
  const edm::EDGetTokenT<edm::View<reco::GsfElectron> >         token_gsfElectrons;
  const edm::EDGetTokenT<edm::View<reco::Electron> >            token_electrons;
  const edm::EDGetTokenT<reco::PFCandidateCollection>           token_pfElectrons;
  const edm::EDGetTokenT<edm::View<reco::Muon> >                token_muons;
  const edm::EDGetTokenT<edm::View<reco::Track> >               token_tracks;
  const edm::InputTag                                           m_leptonCands;
  const edm::EDGetTokenT<edm::ValueMap<float> >                  token_leptonCands;
  const edm::InputTag                                           m_leptonId;
  const edm::EDGetTokenT<edm::ValueMap<float> >                  token_leptonId;

  // service used to make transient tracks from tracks
  const TransientTrackBuilder * m_transientTrackBuilder;

  // algorithm configuration
  unsigned int  m_refineJetAxis;
  double        m_deltaRCut;
  double        m_chi2Cut;

  // specific for reco::Muons
  muon::SelectionType m_muonSelection;

  // nominal beam spot position
  static const reco::Vertex s_nominalBeamSpot;
};

#endif // RecoBTag_SoftLepton_SoftLepton_h
