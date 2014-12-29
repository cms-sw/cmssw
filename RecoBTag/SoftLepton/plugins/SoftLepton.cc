// -*- C++ -*-
//
// Package:    SoftLepton
// Class:      SoftLepton
//
/**\class SoftLepton SoftLepton.cc RecoBTag/SoftLepton/src/SoftLepton.cc

 Description: CMSSW EDProducer for soft lepton b tagging.

 Implementation:
*/

// Original Author:  fwyzard
//         Created:  Wed Oct 18 18:02:07 CEST 2006


#include <memory>
#include <string>
#include <utility>
#include <cmath>

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

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
#include "SoftLepton.h"

enum AxisType {
  AXIS_CALORIMETRIC             = 0,  // use the calorimietric jet axis
  AXIS_CHARGED_AVERAGE          = 1,  // refine jet axis using charged tracks: use a pT-weighted average of (eta, phi)
  AXIS_CHARGED_AVERAGE_NOLEPTON = 2,  // as above, without the tagging lepton track
  AXIS_CHARGED_SUM              = 3,  // refine jet axis using charged tracks: use the sum of tracks momentum
  AXIS_CHARGED_SUM_NOLEPTON     = 4,  // as above, without the tagging lepton track
  AXIS_CALORIMETRIC_NOLEPTON    = 5   // use the calorimetric jet axis minus the lepton momentum
};

using namespace std;
using namespace edm;
using namespace reco;
using namespace ROOT::Math::VectorUtil;

typedef edm::View<reco::GsfElectron> GsfElectronView;
typedef edm::View<reco::Electron>    ElectronView;
typedef edm::View<reco::Muon>        MuonView;

// ------------ static copy of the nominal beamspot --------------------------------------
const reco::Vertex SoftLepton::s_nominalBeamSpot(
  reco::Vertex::Point( 0, 0, 0 ),
  reco::Vertex::Error( ROOT::Math::SVector<double,6>( 0.0015 * 0.0015, //          0.0,        0.0
                                                                  0.0, 0.0015 * 0.0015, //     0.0
                                                                  0.0,             0.0, 15. * 15. ) ),
  1, 1, 0 );

// ------------ c'tor --------------------------------------------------------------------
SoftLepton::SoftLepton(const edm::ParameterSet & iConfig) :
  m_jets(              iConfig.getParameter<edm::InputTag>( "jets" ) ),
  token_jtas(          mayConsume<reco::JetTracksAssociationCollection>( iConfig.getParameter<edm::InputTag>( "jets" ) ) ),
  token_jets(          mayConsume<edm::View<reco::Jet> >( iConfig.getParameter<edm::InputTag>( "jets" ) ) ),
  token_primaryVertex( consumes<reco::VertexCollection>( iConfig.getParameter<edm::InputTag>( "primaryVertex" ) ) ),
  m_leptons(           iConfig.getParameter<edm::InputTag>( "leptons" ) ),
  token_gsfElectrons(  mayConsume<GsfElectronView>( iConfig.getParameter<edm::InputTag>( "leptons" ) ) ),
  token_electrons(     mayConsume<ElectronView>( iConfig.getParameter<edm::InputTag>( "leptons" ) ) ),
  token_pfElectrons(   mayConsume<reco::PFCandidateCollection>( iConfig.getParameter<edm::InputTag>( "leptons" ) ) ),
  token_muons(         mayConsume<MuonView>( iConfig.getParameter<edm::InputTag>( "leptons" ) ) ),
  token_tracks(        mayConsume<edm::View<reco::Track> >( iConfig.getParameter<edm::InputTag>( "leptons" ) ) ),
  m_leptonCands(       iConfig.exists("leptonCands") ? iConfig.getParameter<edm::InputTag>( "leptonCands" ) : edm::InputTag() ),
  token_leptonCands(   mayConsume<edm::ValueMap<float> >( iConfig.exists("leptonCands") ? iConfig.getParameter<edm::InputTag>( "leptonCands" ) : edm::InputTag() ) ),
  m_leptonId(          iConfig.exists("leptonId") ? iConfig.getParameter<edm::InputTag>( "leptonId" ) : edm::InputTag() ),
  token_leptonId(      mayConsume<edm::ValueMap<float> >( iConfig.exists("leptonId") ? iConfig.getParameter<edm::InputTag>( "leptonId" ) : edm::InputTag() ) ),
  m_transientTrackBuilder( 0 ),
  m_refineJetAxis(     iConfig.getParameter<unsigned int>( "refineJetAxis" ) ),
  m_deltaRCut(         iConfig.getParameter<double>( "leptonDeltaRCut" ) ),
  m_chi2Cut(           iConfig.getParameter<double>( "leptonChi2Cut" ) ),
  m_muonSelection(     (muon::SelectionType) iConfig.getParameter<unsigned int>( "muonSelection" ) )
{
  produces<reco::SoftLeptonTagInfoCollection>();
}

// ------------ d'tor --------------------------------------------------------------------
SoftLepton::~SoftLepton(void) {
}

// ------------ method called once per event during the event loop -----------------------
void
SoftLepton::produce(edm::Event & event, const edm::EventSetup & setup) {

  // grab a TransientTrack helper from the Event Setup
  edm::ESHandle<TransientTrackBuilder> builder;
  setup.get<TransientTrackRecord>().get( "TransientTrackBuilder", builder );
  m_transientTrackBuilder = builder.product();

  // input objects

  // input jets (and possibly tracks)
  ProductID jets_id;
  std::vector<edm::RefToBase<reco::Jet> > jets;
  std::vector<reco::TrackRefVector>       tracks;
  do { {
    // look for a JetTracksAssociationCollection
    edm::Handle<reco::JetTracksAssociationCollection> h_jtas;
    event.getByToken(token_jtas, h_jtas);
    if (h_jtas.isValid()) {
      unsigned int size = h_jtas->size();
      jets.resize(size);
      tracks.resize(size);
      for (unsigned int i = 0; i < size; ++i) {
        jets[i]   = (*h_jtas)[i].first;
        tracks[i] = (*h_jtas)[i].second;
      }
      break;
    }
  } { // else...
    // look for a View<Jet>
    edm::Handle<edm::View<reco::Jet> > h_jets;
    event.getByToken(token_jets, h_jets);
    if (h_jets.isValid()) {
      unsigned int size = h_jets->size();
      jets.resize(size);
      tracks.resize(size);
      for (unsigned int i = 0; i < h_jets->size(); i++)
        jets[i] = h_jets->refAt(i);
      break;
    }
  } { // else...
    throw edm::Exception(edm::errors::NotFound) << "Object " << m_jets << " of type among (\"reco::JetTracksAssociationCollection\", \"edm::View<reco::Jet>\") not found";
  } } while (false);

  // input primary vetex
  reco::Vertex vertex;
  Handle<reco::VertexCollection> h_primaryVertex;
  event.getByToken(token_primaryVertex, h_primaryVertex);
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
    event.getByToken(token_leptonCands, h_leptonCands);

  // try to access the input collection as a collection of GsfElectrons, Muons or Tracks

  unsigned int leptonId = SoftLeptonProperties::Quality::leptonId;
  do { {
    // look for View<GsfElectron>
    Handle<GsfElectronView> h_electrons;
    event.getByToken(token_gsfElectrons, h_electrons);

    if (h_electrons.isValid()) {
      leptonId = SoftLeptonProperties::Quality::egammaElectronId;
      for (GsfElectronView::const_iterator electron = h_electrons->begin(); electron != h_electrons->end(); ++electron) {
        LeptonIds &id = leptons[reco::TrackBaseRef(electron->gsfTrack())];
        id[SoftLeptonProperties::Quality::pfElectronId] = electron->mva_e_pi();
        if (haveLeptonCands)
          id[SoftLeptonProperties::Quality::btagElectronCands] = (*h_leptonCands)[h_electrons->refAt(electron - h_electrons->begin())];
      }
      break;
    }
  } { // else
    // look for View<Electron>
    // FIXME: is this obsolete?
    Handle<ElectronView> h_electrons;
    event.getByToken(token_electrons, h_electrons);
    if (h_electrons.isValid()) {
      leptonId = SoftLeptonProperties::Quality::egammaElectronId;
      for (ElectronView::const_iterator electron = h_electrons->begin(); electron != h_electrons->end(); ++electron) {
        LeptonIds &id = leptons[reco::TrackBaseRef(electron->track())];
        if (haveLeptonCands)
          id[SoftLeptonProperties::Quality::btagElectronCands] = (*h_leptonCands)[h_electrons->refAt(electron - h_electrons->begin())];
      }
      break;
    }
  } { // else
    // look for PFElectrons
    // FIXME: is this obsolete?
    Handle<reco::PFCandidateCollection> h_electrons;
    event.getByToken(token_pfElectrons, h_electrons);
    if (h_electrons.isValid()) {
      leptonId = SoftLeptonProperties::Quality::egammaElectronId;
      for (reco::PFCandidateCollection::const_iterator electron = h_electrons->begin(); electron != h_electrons->end(); ++electron) {
        LeptonIds *id;
        if (electron->gsfTrackRef().isNonnull())
          id = &leptons[reco::TrackBaseRef(electron->gsfTrackRef())];
        else if (electron->trackRef().isNonnull())
          id = &leptons[reco::TrackBaseRef(electron->trackRef())];
        else
          continue;
        (*id)[SoftLeptonProperties::Quality::pfElectronId] = electron->mva_e_pi();
        if (haveLeptonCands)
          (*id)[SoftLeptonProperties::Quality::btagElectronCands] = (*h_leptonCands)[reco::PFCandidateRef(h_electrons, electron - h_electrons->begin())];
      }
      break;
    }
  } { // else
    // look for View<Muon>
    Handle<MuonView> h_muons;
    event.getByToken(token_muons, h_muons);
    if (h_muons.isValid()) {
      for (MuonView::const_iterator muon = h_muons->begin(); muon != h_muons->end(); ++muon) {
        // FIXME -> turn this selection into a muonCands input?
        if (muon::isGoodMuon( *muon, m_muonSelection )) {
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
            (*id)[SoftLeptonProperties::Quality::btagMuonCands] = (*h_leptonCands)[h_muons->refAt(muon - h_muons->begin())];
        }
      }
      break;
    }
  } { // else
    // look for edm::View<Track>
    Handle<edm::View<reco::Track> > h_tracks;
    event.getByToken(token_tracks, h_tracks);
    if (h_tracks.isValid()) {
      for (unsigned int i = 0; i < h_tracks->size(); i++) {
        LeptonIds &id = leptons[h_tracks->refAt(i)];
        if (haveLeptonCands)
          id[SoftLeptonProperties::Quality::btagLeptonCands] = (*h_leptonCands)[h_tracks->refAt(i)];
      }
      break;
    }
  } { // else
    throw edm::Exception(edm::errors::NotFound) << "Object " << m_leptons << " of type among (\"edm::View<reco::GsfElectron>\", \"edm::View<reco::Muon>\", \"edm::View<reco::Track>\") !found";
  } } while (false);

  if (!(m_leptonId == edm::InputTag())) {
    Handle<edm::ValueMap<float> > h_leptonId;
    event.getByToken(token_leptonId, h_leptonId);

    for (Leptons::iterator lepton = leptons.begin(); lepton != leptons.end(); ++lepton)
      lepton->second[leptonId] = (*h_leptonId)[lepton->first];
  }

  // output collections
  std::auto_ptr<reco::SoftLeptonTagInfoCollection> outputCollection(  new reco::SoftLeptonTagInfoCollection() );
  for (unsigned int i = 0; i < jets.size(); ++i) {
    reco::SoftLeptonTagInfo result = tag( jets[i], tracks[i], leptons, vertex );
    outputCollection->push_back( result );
  }
  event.put( outputCollection );
}

// ---------------------------------------------------------------------------------------
reco::SoftLeptonTagInfo SoftLepton::tag (
    const edm::RefToBase<reco::Jet> & jet,
    const reco::TrackRefVector      & tracks,
    const Leptons                   & leptons,
    const reco::Vertex              & primaryVertex
) const {
  reco::SoftLeptonTagInfo info;
  info.setJetRef( jet );

  for(Leptons::const_iterator lepton = leptons.begin(); lepton != leptons.end(); ++lepton) {
    const math::XYZVector & lepton_momentum = lepton->first->momentum();
    if (m_chi2Cut > 0.0 && lepton->first->normalizedChi2() > m_chi2Cut)
      continue;

    const GlobalVector jetAxis = refineJetAxis( jet, tracks, lepton->first );
    const math::XYZVector axis( jetAxis.x(), jetAxis.y(), jetAxis.z());
    float deltaR = Geom::deltaR(lepton_momentum, axis);
    if (deltaR > m_deltaRCut)
      continue;

    reco::SoftLeptonProperties properties;

    reco::TransientTrack transientTrack = m_transientTrackBuilder->build(*lepton->first);
    properties.sip2d    = IPTools::signedTransverseImpactParameter( transientTrack, jetAxis, primaryVertex ).second.significance();
    properties.sip3d    = IPTools::signedImpactParameter3D( transientTrack, jetAxis, primaryVertex ).second.significance();
    properties.deltaR   = deltaR;
    properties.ptRel    = Perp( lepton_momentum, axis );
    properties.p0Par    = boostedPPar( lepton_momentum, axis );
    properties.etaRel   = relativeEta( lepton_momentum, axis );
    properties.ratio    = lepton_momentum.R() / axis.R();
    properties.ratioRel = lepton_momentum.Dot(axis) / axis.Mag2();

    for(LeptonIds::const_iterator iter = lepton->second.begin(); iter != lepton->second.end(); ++iter)
      properties.setQuality(static_cast<SoftLeptonProperties::Quality::Generic>(iter->first), iter->second);

    info.insert( lepton->first, properties );
  }

  return info;
}


// ---------------------------------------------------------------------------------------
GlobalVector SoftLepton::refineJetAxis (
    const edm::RefToBase<reco::Jet>   & jet,
    const reco::TrackRefVector        & tracks,
    const reco::TrackBaseRef & exclude /* = reco::TrackBaseRef() */
) const {
  math::XYZVector axis = jet->momentum();

  if (m_refineJetAxis == AXIS_CHARGED_AVERAGE or
      m_refineJetAxis == AXIS_CHARGED_AVERAGE_NOLEPTON) {

    double sum_pT        = 0.;
    double sum_eta_by_pT = 0.;
    double sum_phi_by_pT = 0.;

    double perp;
    double phi_rel;
    double eta_rel;

    // refine jet eta and phi with charged tracks measurements, if available
    for (reco::TrackRefVector::const_iterator track_it = tracks.begin(); track_it != tracks.end(); ++track_it ) {
      const reco::Track & track = **track_it;

      perp = track.pt();
      eta_rel = (double) track.eta() - axis.eta();
      phi_rel = (double) track.phi() - axis.phi();
      while (phi_rel < -M_PI) phi_rel += 2*M_PI;
      while (phi_rel >  M_PI) phi_rel -= 2*M_PI;

      sum_pT        += perp;
      sum_phi_by_pT += perp * phi_rel;
      sum_eta_by_pT += perp * eta_rel;
    }

    // "remove" excluded track
    if (m_refineJetAxis == AXIS_CHARGED_AVERAGE_NOLEPTON and exclude.isNonnull()) {
      const reco::Track & track = *exclude;

      perp = track.pt();
      eta_rel = (double) track.eta() - axis.eta();
      phi_rel = (double) track.phi() - axis.phi();
      while (phi_rel < -M_PI) phi_rel += 2*M_PI;
      while (phi_rel >  M_PI) phi_rel -= 2*M_PI;

      sum_pT        -= perp;
      sum_phi_by_pT -= perp * phi_rel;
      sum_eta_by_pT -= perp * eta_rel;
    }

    if (sum_pT > 1.)    // avoid the case of only the lepton-track with small rounding errors
      axis = math::RhoEtaPhiVector( axis.rho(), axis.eta() + sum_eta_by_pT / sum_pT, axis.phi() + sum_phi_by_pT / sum_pT);

  } else if (m_refineJetAxis == AXIS_CHARGED_SUM or
             m_refineJetAxis == AXIS_CHARGED_SUM_NOLEPTON) {
    math::XYZVector sum;

    // recalculate the jet direction as the sum of charget tracks momenta
    for (reco::TrackRefVector::const_iterator track_it = tracks.begin(); track_it != tracks.end(); ++track_it ) {
      const reco::Track & track = **track_it;
      sum += track.momentum();
    }

    // "remove" excluded track
    if (m_refineJetAxis == AXIS_CHARGED_SUM_NOLEPTON and exclude.isNonnull()) {
      const reco::Track & track = *exclude;
      sum -= track.momentum();
    }

    if (sum.R() > 1.) // avoid the case of only the lepton-track with small rounding errors
      axis = sum;
  } else if(m_refineJetAxis == AXIS_CALORIMETRIC_NOLEPTON) {
    axis -= exclude->momentum();
  }

  return GlobalVector(axis.x(), axis.y(), axis.z());
}

double SoftLepton::relativeEta(const math::XYZVector& vector, const math::XYZVector& axis) {
  double mag = vector.r() * axis.r();
  double dot = vector.Dot(axis);
  return -log((mag - dot)/(mag + dot)) / 2;
}

// compute the lepton momentum along the jet axis, in the jet rest frame
double SoftLepton::boostedPPar(const math::XYZVector& vector, const math::XYZVector& axis) {
  static const double lepton_mass = 0.00;       // assume a massless (ultrarelativistic) lepton
  static const double jet_mass    = 5.279;      // use B±/B0 mass as the jet rest mass [PDG 2007 updates]
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > lepton(vector.Dot(axis) / axis.r(), Perp(vector, axis), 0., lepton_mass);
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > jet( axis.r(), 0., 0., jet_mass );
  ROOT::Math::BoostX boost( -jet.Beta() );
  return boost(lepton).x();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void
SoftLepton::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<unsigned int>("muonSelection",1);
  desc.add<edm::InputTag>("leptons",edm::InputTag("muons"));
  desc.add<edm::InputTag>("primaryVertex",edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("leptonCands",edm::InputTag(""));
  desc.add<edm::InputTag>("leptonId",edm::InputTag(""));
  desc.add<unsigned int>("refineJetAxis",0);
  desc.add<edm::InputTag>("jets",edm::InputTag("ak4PFJetsCHS"));
  desc.add<double>("leptonDeltaRCut",0.4);
  desc.add<double>("leptonChi2Cut",9999.0);
  descriptions.addDefault(desc);
}
