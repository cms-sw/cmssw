// -*- C++ -*-
//
// Package:    SoftLepton
// Class:      SoftLepton
//
/**\class SoftLepton SoftLepton.cc RecoBTag/SoftLepton/src/SoftLepton.cc

 Description: CMSSW EDProducer for soft lepton b tagging.

 Implementation:
     The actual tagging is performed by SoftLeptonAlgorithm.
*/

// Original Author:  fwyzard
//         Created:  Wed Oct 18 18:02:07 CEST 2006

#include <memory>
#include <string>
#include <utility>
#include <cmath>
#include <map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
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
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// ROOT::Math vectors (aka math::XYZVector)
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Math/GenVector/PxPyPzM4D.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/Boost.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

class SoftLepton : public edm::global::EDProducer<> {
public:
  explicit SoftLepton(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  struct TrackCompare {
    inline bool operator()(const edm::RefToBase<reco::Track> &t1, const edm::RefToBase<reco::Track> &t2) const {
      return t1.key() < t2.key();
    }
  };

  using LeptonIds = std::map<unsigned int, float>;
  using Leptons = std::map<edm::RefToBase<reco::Track>, LeptonIds, TrackCompare>;

  // generic interface, using a TrackRefVector for lepton tracks
  reco::SoftLeptonTagInfo tag(const edm::RefToBase<reco::Jet> &jet,
                              const reco::TrackRefVector &tracks,
                              const Leptons &leptons,
                              const reco::Vertex &primaryVertex,
                              const TransientTrackBuilder &builder) const;

protected:
  // generic interface, using a TrackRefVector for lepton tracks

  GlobalVector refineJetAxis(const edm::RefToBase<reco::Jet> &jet,
                             const reco::TrackRefVector &tracks,
                             const edm::RefToBase<reco::Track> &exclude = edm::RefToBase<reco::Track>()) const;

  static double relativeEta(const math::XYZVector &vector, const math::XYZVector &axis);

  static double boostedPPar(const math::XYZVector &vector, const math::XYZVector &axis);

private:
  void produce(edm::StreamID, edm::Event &event, const edm::EventSetup &setup) const final;

  // configuration
  const edm::InputTag m_jets;
  const edm::EDGetTokenT<reco::JetTracksAssociationCollection> token_jtas;
  const edm::EDGetTokenT<edm::View<reco::Jet> > token_jets;
  const edm::EDGetTokenT<reco::VertexCollection> token_primaryVertex;
  const edm::InputTag m_leptons;
  const edm::EDGetTokenT<edm::View<reco::GsfElectron> > token_gsfElectrons;
  const edm::EDGetTokenT<edm::View<reco::Electron> > token_electrons;
  const edm::EDGetTokenT<reco::PFCandidateCollection> token_pfElectrons;
  const edm::EDGetTokenT<edm::View<reco::Muon> > token_muons;
  const edm::EDGetTokenT<edm::View<reco::Track> > token_tracks;
  const edm::InputTag m_leptonCands;
  const edm::EDGetTokenT<edm::ValueMap<float> > token_leptonCands;
  const edm::InputTag m_leptonId;
  const edm::EDGetTokenT<edm::ValueMap<float> > token_leptonId;

  // service used to make transient tracks from tracks
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> token_builder;

  const edm::EDPutTokenT<reco::SoftLeptonTagInfoCollection> token_put;
  // algorithm configuration
  const unsigned int m_refineJetAxis;
  const double m_deltaRCut;
  const double m_chi2Cut;

  // specific for reco::Muons
  const muon::SelectionType m_muonSelection;

  // nominal beam spot position
  static const reco::Vertex s_nominalBeamSpot;
};

enum AxisType {
  AXIS_CALORIMETRIC = 0,              // use the calorimietric jet axis
  AXIS_CHARGED_AVERAGE = 1,           // refine jet axis using charged tracks: use a pT-weighted average of (eta, phi)
  AXIS_CHARGED_AVERAGE_NOLEPTON = 2,  // as above, without the tagging lepton track
  AXIS_CHARGED_SUM = 3,               // refine jet axis using charged tracks: use the sum of tracks momentum
  AXIS_CHARGED_SUM_NOLEPTON = 4,      // as above, without the tagging lepton track
  AXIS_CALORIMETRIC_NOLEPTON = 5      // use the calorimetric jet axis minus the lepton momentum
};

using namespace std;
using namespace edm;
using namespace reco;
using namespace ROOT::Math::VectorUtil;

typedef edm::View<reco::GsfElectron> GsfElectronView;
typedef edm::View<reco::Electron> ElectronView;
typedef edm::View<reco::Muon> MuonView;

// ------------ static copy of the nominal beamspot --------------------------------------
const reco::Vertex SoftLepton::s_nominalBeamSpot(
    reco::Vertex::Point(0, 0, 0),
    reco::Vertex::Error(ROOT::Math::SVector<double, 6>(0.0015 * 0.0015,  //          0.0,        0.0
                                                       0.0,
                                                       0.0015 * 0.0015,  //     0.0
                                                       0.0,
                                                       0.0,
                                                       15. * 15.)),
    1,
    1,
    0);

// ------------ c'tor --------------------------------------------------------------------
SoftLepton::SoftLepton(const edm::ParameterSet &iConfig)
    : m_jets(iConfig.getParameter<edm::InputTag>("jets")),
      token_jtas(mayConsume<reco::JetTracksAssociationCollection>(m_jets)),
      token_jets(mayConsume<edm::View<reco::Jet> >(m_jets)),
      token_primaryVertex(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertex"))),
      m_leptons(iConfig.getParameter<edm::InputTag>("leptons")),
      token_gsfElectrons(mayConsume<GsfElectronView>(m_leptons)),
      token_electrons(mayConsume<ElectronView>(m_leptons)),
      token_pfElectrons(mayConsume<reco::PFCandidateCollection>(m_leptons)),
      token_muons(mayConsume<MuonView>(m_leptons)),
      token_tracks(mayConsume<edm::View<reco::Track> >(m_leptons)),
      m_leptonCands(iConfig.getParameter<edm::InputTag>("leptonCands")),
      token_leptonCands(mayConsume<edm::ValueMap<float> >(m_leptonCands)),
      m_leptonId(iConfig.getParameter<edm::InputTag>("leptonId")),
      token_leptonId(mayConsume<edm::ValueMap<float> >(m_leptonId)),
      token_builder(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      token_put(produces()),
      m_refineJetAxis(iConfig.getParameter<unsigned int>("refineJetAxis")),
      m_deltaRCut(iConfig.getParameter<double>("leptonDeltaRCut")),
      m_chi2Cut(iConfig.getParameter<double>("leptonChi2Cut")),
      m_muonSelection((muon::SelectionType)iConfig.getParameter<unsigned int>("muonSelection")) {}

// ------------ method called once per event during the event loop -----------------------
void SoftLepton::produce(edm::StreamID, edm::Event &event, const edm::EventSetup &setup) const {
  // grab a TransientTrack helper from the Event Setup
  auto const &transientTrackBuilder = setup.getData(token_builder);

  // input objects

  // input jets (and possibly tracks)
  ProductID jets_id;
  std::vector<edm::RefToBase<reco::Jet> > jets;
  std::vector<reco::TrackRefVector> tracks;
  do {
    {
      // look for a JetTracksAssociationCollection
      edm::Handle<reco::JetTracksAssociationCollection> h_jtas = event.getHandle(token_jtas);
      if (h_jtas.isValid()) {
        unsigned int size = h_jtas->size();
        jets.resize(size);
        tracks.resize(size);
        for (unsigned int i = 0; i < size; ++i) {
          jets[i] = (*h_jtas)[i].first;
          tracks[i] = (*h_jtas)[i].second;
        }
        break;
      }
    }
    {  // else...
      // look for a View<Jet>
      edm::Handle<edm::View<reco::Jet> > h_jets = event.getHandle(token_jets);
      if (h_jets.isValid()) {
        unsigned int size = h_jets->size();
        jets.resize(size);
        tracks.resize(size);
        for (unsigned int i = 0; i < h_jets->size(); i++)
          jets[i] = h_jets->refAt(i);
        break;
      }
    }
    {  // else...
      throw edm::Exception(edm::errors::NotFound)
          << "Object " << m_jets
          << " of type among (\"reco::JetTracksAssociationCollection\", \"edm::View<reco::Jet>\") not found";
    }
  } while (false);

  // input primary vetex
  reco::Vertex vertex;
  Handle<reco::VertexCollection> h_primaryVertex = event.getHandle(token_primaryVertex);
  if (h_primaryVertex.isValid() and not h_primaryVertex->empty())
    vertex = h_primaryVertex->front();
  else
    // fall back to nominal beam spot
    vertex = s_nominalBeamSpot;

  // input leptons (can be of different types)
  Leptons leptons;

  Handle<edm::ValueMap<float> > h_leptonCands;
  bool haveLeptonCands = !(m_leptonCands == edm::InputTag());
  if (haveLeptonCands)
    h_leptonCands = event.getHandle(token_leptonCands);

  // try to access the input collection as a collection of GsfElectrons, Muons or Tracks

  unsigned int leptonId = SoftLeptonProperties::Quality::leptonId;
  do {
    {
      // look for View<GsfElectron>
      Handle<GsfElectronView> h_electrons = event.getHandle(token_gsfElectrons);

      if (h_electrons.isValid()) {
        leptonId = SoftLeptonProperties::Quality::egammaElectronId;
        for (GsfElectronView::const_iterator electron = h_electrons->begin(); electron != h_electrons->end();
             ++electron) {
          LeptonIds &id = leptons[reco::TrackBaseRef(electron->gsfTrack())];
          id[SoftLeptonProperties::Quality::pfElectronId] = electron->mva_e_pi();
          if (haveLeptonCands)
            id[SoftLeptonProperties::Quality::btagElectronCands] =
                (*h_leptonCands)[h_electrons->refAt(electron - h_electrons->begin())];
        }
        break;
      }
    }
    {  // else
      // look for View<Electron>
      // FIXME: is this obsolete?
      Handle<ElectronView> h_electrons = event.getHandle(token_electrons);
      if (h_electrons.isValid()) {
        leptonId = SoftLeptonProperties::Quality::egammaElectronId;
        for (ElectronView::const_iterator electron = h_electrons->begin(); electron != h_electrons->end(); ++electron) {
          LeptonIds &id = leptons[reco::TrackBaseRef(electron->track())];
          if (haveLeptonCands)
            id[SoftLeptonProperties::Quality::btagElectronCands] =
                (*h_leptonCands)[h_electrons->refAt(electron - h_electrons->begin())];
        }
        break;
      }
    }
    {  // else
      // look for PFElectrons
      // FIXME: is this obsolete?
      Handle<reco::PFCandidateCollection> h_electrons = event.getHandle(token_pfElectrons);
      if (h_electrons.isValid()) {
        leptonId = SoftLeptonProperties::Quality::egammaElectronId;
        for (reco::PFCandidateCollection::const_iterator electron = h_electrons->begin();
             electron != h_electrons->end();
             ++electron) {
          LeptonIds *id;
          if (electron->gsfTrackRef().isNonnull())
            id = &leptons[reco::TrackBaseRef(electron->gsfTrackRef())];
          else if (electron->trackRef().isNonnull())
            id = &leptons[reco::TrackBaseRef(electron->trackRef())];
          else
            continue;
          (*id)[SoftLeptonProperties::Quality::pfElectronId] = electron->mva_e_pi();
          if (haveLeptonCands)
            (*id)[SoftLeptonProperties::Quality::btagElectronCands] =
                (*h_leptonCands)[reco::PFCandidateRef(h_electrons, electron - h_electrons->begin())];
        }
        break;
      }
    }
    {  // else
      // look for View<Muon>
      Handle<MuonView> h_muons = event.getHandle(token_muons);
      if (h_muons.isValid()) {
        for (MuonView::const_iterator muon = h_muons->begin(); muon != h_muons->end(); ++muon) {
          // FIXME -> turn this selection into a muonCands input?
          if (muon::isGoodMuon(*muon, m_muonSelection)) {
            LeptonIds *id;
            if (muon->globalTrack().isNonnull())
              id = &leptons[reco::TrackBaseRef(muon->globalTrack())];
            else if (muon->innerTrack().isNonnull())
              id = &leptons[reco::TrackBaseRef(muon->innerTrack())];
            else if (muon->outerTrack().isNonnull())
              // does this makes sense ?
              id = &leptons[reco::TrackBaseRef(muon->outerTrack())];
            else
              continue;
            if (haveLeptonCands)
              (*id)[SoftLeptonProperties::Quality::btagMuonCands] =
                  (*h_leptonCands)[h_muons->refAt(muon - h_muons->begin())];
          }
        }
        break;
      }
    }
    {  // else
      // look for edm::View<Track>
      Handle<edm::View<reco::Track> > h_tracks = event.getHandle(token_tracks);
      if (h_tracks.isValid()) {
        for (unsigned int i = 0; i < h_tracks->size(); i++) {
          LeptonIds &id = leptons[h_tracks->refAt(i)];
          if (haveLeptonCands)
            id[SoftLeptonProperties::Quality::btagLeptonCands] = (*h_leptonCands)[h_tracks->refAt(i)];
        }
        break;
      }
    }
    {  // else
      throw edm::Exception(edm::errors::NotFound) << "Object " << m_leptons
                                                  << " of type among (\"edm::View<reco::GsfElectron>\", "
                                                     "\"edm::View<reco::Muon>\", \"edm::View<reco::Track>\") !found";
    }
  } while (false);

  if (!(m_leptonId == edm::InputTag())) {
    edm::ValueMap<float> const &h_leptonId = event.get(token_leptonId);

    for (Leptons::iterator lepton = leptons.begin(); lepton != leptons.end(); ++lepton)
      lepton->second[leptonId] = h_leptonId[lepton->first];
  }

  // output collections
  reco::SoftLeptonTagInfoCollection outputCollection;
  for (unsigned int i = 0; i < jets.size(); ++i) {
    reco::SoftLeptonTagInfo result = tag(jets[i], tracks[i], leptons, vertex, transientTrackBuilder);
    outputCollection.push_back(result);
  }
  event.emplace(token_put, std::move(outputCollection));
}

// ---------------------------------------------------------------------------------------
reco::SoftLeptonTagInfo SoftLepton::tag(const edm::RefToBase<reco::Jet> &jet,
                                        const reco::TrackRefVector &tracks,
                                        const Leptons &leptons,
                                        const reco::Vertex &primaryVertex,
                                        const TransientTrackBuilder &transientTrackBuilder) const {
  reco::SoftLeptonTagInfo info;
  info.setJetRef(jet);

  for (Leptons::const_iterator lepton = leptons.begin(); lepton != leptons.end(); ++lepton) {
    const math::XYZVector &lepton_momentum = lepton->first->momentum();
    if (m_chi2Cut > 0.0 && lepton->first->normalizedChi2() > m_chi2Cut)
      continue;

    const GlobalVector jetAxis = refineJetAxis(jet, tracks, lepton->first);
    const math::XYZVector axis(jetAxis.x(), jetAxis.y(), jetAxis.z());
    float deltaR = Geom::deltaR(lepton_momentum, axis);
    if (deltaR > m_deltaRCut)
      continue;

    reco::SoftLeptonProperties properties;

    reco::TransientTrack transientTrack = transientTrackBuilder.build(*lepton->first);
    Measurement1D ip2d = IPTools::signedTransverseImpactParameter(transientTrack, jetAxis, primaryVertex).second;
    Measurement1D ip3d = IPTools::signedImpactParameter3D(transientTrack, jetAxis, primaryVertex).second;
    properties.sip2dsig = ip2d.significance();
    properties.sip3dsig = ip3d.significance();
    properties.sip2d = ip2d.value();
    properties.sip3d = ip3d.value();
    properties.deltaR = deltaR;
    properties.ptRel = Perp(lepton_momentum, axis);
    properties.p0Par = boostedPPar(lepton_momentum, axis);
    properties.etaRel = relativeEta(lepton_momentum, axis);
    properties.ratio = lepton_momentum.R() / axis.R();
    properties.ratioRel = lepton_momentum.Dot(axis) / axis.Mag2();

    for (LeptonIds::const_iterator iter = lepton->second.begin(); iter != lepton->second.end(); ++iter)
      properties.setQuality(static_cast<SoftLeptonProperties::Quality::Generic>(iter->first), iter->second);

    info.insert(lepton->first, properties);
  }

  return info;
}

// ---------------------------------------------------------------------------------------
GlobalVector SoftLepton::refineJetAxis(const edm::RefToBase<reco::Jet> &jet,
                                       const reco::TrackRefVector &tracks,
                                       const reco::TrackBaseRef &exclude /* = reco::TrackBaseRef() */
) const {
  math::XYZVector axis = jet->momentum();

  if (m_refineJetAxis == AXIS_CHARGED_AVERAGE or m_refineJetAxis == AXIS_CHARGED_AVERAGE_NOLEPTON) {
    double sum_pT = 0.;
    double sum_eta_by_pT = 0.;
    double sum_phi_by_pT = 0.;

    double perp;
    double phi_rel;
    double eta_rel;

    // refine jet eta and phi with charged tracks measurements, if available
    for (reco::TrackRefVector::const_iterator track_it = tracks.begin(); track_it != tracks.end(); ++track_it) {
      const reco::Track &track = **track_it;

      perp = track.pt();
      eta_rel = (double)track.eta() - axis.eta();
      phi_rel = (double)track.phi() - axis.phi();
      while (phi_rel < -M_PI)
        phi_rel += 2 * M_PI;
      while (phi_rel > M_PI)
        phi_rel -= 2 * M_PI;

      sum_pT += perp;
      sum_phi_by_pT += perp * phi_rel;
      sum_eta_by_pT += perp * eta_rel;
    }

    // "remove" excluded track
    if (m_refineJetAxis == AXIS_CHARGED_AVERAGE_NOLEPTON and exclude.isNonnull()) {
      const reco::Track &track = *exclude;

      perp = track.pt();
      eta_rel = (double)track.eta() - axis.eta();
      phi_rel = (double)track.phi() - axis.phi();
      while (phi_rel < -M_PI)
        phi_rel += 2 * M_PI;
      while (phi_rel > M_PI)
        phi_rel -= 2 * M_PI;

      sum_pT -= perp;
      sum_phi_by_pT -= perp * phi_rel;
      sum_eta_by_pT -= perp * eta_rel;
    }

    if (sum_pT > 1.)  // avoid the case of only the lepton-track with small rounding errors
      axis =
          math::RhoEtaPhiVector(axis.rho(), axis.eta() + sum_eta_by_pT / sum_pT, axis.phi() + sum_phi_by_pT / sum_pT);

  } else if (m_refineJetAxis == AXIS_CHARGED_SUM or m_refineJetAxis == AXIS_CHARGED_SUM_NOLEPTON) {
    math::XYZVector sum;

    // recalculate the jet direction as the sum of charget tracks momenta
    for (reco::TrackRefVector::const_iterator track_it = tracks.begin(); track_it != tracks.end(); ++track_it) {
      const reco::Track &track = **track_it;
      sum += track.momentum();
    }

    // "remove" excluded track
    if (m_refineJetAxis == AXIS_CHARGED_SUM_NOLEPTON and exclude.isNonnull()) {
      const reco::Track &track = *exclude;
      sum -= track.momentum();
    }

    if (sum.R() > 1.)  // avoid the case of only the lepton-track with small rounding errors
      axis = sum;
  } else if (m_refineJetAxis == AXIS_CALORIMETRIC_NOLEPTON) {
    axis -= exclude->momentum();
  }

  return GlobalVector(axis.x(), axis.y(), axis.z());
}

double SoftLepton::relativeEta(const math::XYZVector &vector, const math::XYZVector &axis) {
  double mag = vector.r() * axis.r();
  double dot = vector.Dot(axis);
  return -log((mag - dot) / (mag + dot)) / 2;
}

// compute the lepton momentum along the jet axis, in the jet rest frame
double SoftLepton::boostedPPar(const math::XYZVector &vector, const math::XYZVector &axis) {
  static const double lepton_mass = 0.00;  // assume a massless (ultrarelativistic) lepton
  static const double jet_mass = 5.279;    // use B±/B0 mass as the jet rest mass [PDG 2007 updates]
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > lepton(
      vector.Dot(axis) / axis.r(), Perp(vector, axis), 0., lepton_mass);
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > jet(axis.r(), 0., 0., jet_mass);
  ROOT::Math::BoostX boost(-jet.Beta());
  return boost(lepton).x();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void SoftLepton::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<unsigned int>("muonSelection", 1);
  desc.add<edm::InputTag>("leptons", edm::InputTag("muons"));
  desc.add<edm::InputTag>("primaryVertex", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("leptonCands", edm::InputTag());
  desc.add<edm::InputTag>("leptonId", edm::InputTag());
  desc.add<unsigned int>("refineJetAxis", 0);
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<double>("leptonDeltaRCut", 0.4);
  desc.add<double>("leptonChi2Cut", 9999.0);
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(SoftLepton);
