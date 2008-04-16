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
// $Id: SoftLepton.cc,v 1.17 2008/03/03 10:52:26 fwyzard Exp $


#include <memory>
#include <string>
#include <utility>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/RefToBase.h"

// ROOT::Math vectors (aka math::XYZVector)
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/PxPyPzM4D.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/Boost.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexSorter.h"
#include "SoftLepton.h"

enum {
  AXIS_CALORIMETRIC             = 0,  // use the calorimietric jet axis
  AXIS_CHARGED_AVERAGE          = 1,  // refine jet axis using charged tracks: use a pT-weighted average of (eta, phi)
  AXIS_CHARGED_AVERAGE_NOLEPTON = 2,  // as above, without the tagging lepton track
  AXIS_CHARGED_SUM              = 3,  // refine jet axis using charged tracks: use the sum of tracks momentum
  AXIS_CHARGED_SUM_NOLEPTON     = 4   // as above, without the tagging lepton track
};

using namespace std;
using namespace edm;
using namespace reco;
using namespace ROOT::Math::VectorUtil;

const reco::Vertex SoftLepton::s_nominalBeamSpot(
  reco::Vertex::Point( 0, 0, 0 ),
  reco::Vertex::Error( ROOT::Math::SVector<double,6>( 0.0015 * 0.0015, //          0.0,        0.0
                                                                  0.0, 0.0015 * 0.0015, //     0.0  
                                                                  0.0,             0.0, 15. * 15. ) ),
  1, 1, 0 );

SoftLepton::SoftLepton(const edm::ParameterSet & iConfig) :
  m_jets(          iConfig.getParameter<edm::InputTag>( "jets" ) ),
  m_primaryVertex( iConfig.getParameter<edm::InputTag>( "primaryVertex" ) ),
  m_leptons(       iConfig.getParameter<edm::InputTag>( "leptons" ) ),
  m_transientTrackBuilder( NULL ),
  m_refineJetAxis( iConfig.getParameter<unsigned int>( "refineJetAxis" ) ),
  m_deltaRCut(     iConfig.getParameter<double>( "leptonDeltaRCut" ) ),
  m_chi2Cut(       iConfig.getParameter<double>( "leptonChi2Cut" ) ),
  m_qualityCut(    iConfig.getParameter<double>( "leptonQualityCut" ) )
{
  produces<reco::SoftLeptonTagInfoCollection>();
}

SoftLepton::~SoftLepton() {
}

void
SoftLepton::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // input objects

  // input jets (and possibly tracks)
  ProductID jets_id;
  std::vector<edm::RefToBase<reco::Jet> > jets;
  std::vector<reco::TrackRefVector>       tracks;
  do { {
    // look for a JetTracksAssociationCollection
    edm::Handle<reco::JetTracksAssociationCollection> h_jtas;
    iEvent.getByLabel(m_jets, h_jtas);
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
    iEvent.getByLabel(m_jets, h_jets);
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
  
  // input primary vetex (optional, can be "nominal" or "beamspot")
  reco::Vertex vertex;
  Handle<reco::VertexCollection> h_primaryVertex;
  if (m_primaryVertex.label() == "nominal") {
    vertex = s_nominalBeamSpot;
  } else
  if (m_primaryVertex.label() == "beamspot") {
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByType(recoBeamSpotHandle);
    vertex = reco::Vertex(recoBeamSpotHandle->position(), recoBeamSpotHandle->covariance3D(), 1, 1, 0);
  } else {
    iEvent.getByLabel(m_primaryVertex, h_primaryVertex);
    if (h_primaryVertex->size()) {
      vertex = h_primaryVertex->front();
    } else {
      // fall back to nominal beam spot
      vertex = s_nominalBeamSpot;
    }
  }

  // input leptons (can be of different types)
  ProductID leptons_id;
  std::vector<edm::RefToBase<reco::Track> > leptons;
  // try to access the input collection as a collection of Electrons, Muons or Tracks
  do { {
    // look for vector<Electron>
    Handle<reco::ElectronCollection> h_electrons;
    iEvent.getByLabel(m_leptons, h_electrons);
    if (h_electrons.isValid()) {
      for (reco::ElectronCollection::const_iterator electron = h_electrons->begin(); electron != h_electrons->end(); ++electron)
        leptons.push_back(edm::RefToBase<reco::Track>( electron->track() ));
      break;
    }
  } { // else
    // look for vector<PixelMatchElectron>
    Handle<reco::PixelMatchGsfElectronCollection> h_electrons;
    iEvent.getByLabel(m_leptons, h_electrons);
    if (h_electrons.isValid()) {
      for (reco::PixelMatchGsfElectronCollection::const_iterator electron = h_electrons->begin(); electron != h_electrons->end(); ++electron)
        leptons.push_back(edm::RefToBase<reco::Track>( electron->track() ));
      break;
    }
  } { // else
    // look for vector<GsfElectron>
    Handle<reco::GsfElectronCollection> h_electrons;
    iEvent.getByLabel(m_leptons, h_electrons);
    if (h_electrons.isValid()) {
      for (reco::GsfElectronCollection::const_iterator electron = h_electrons->begin(); electron != h_electrons->end(); ++electron)
        leptons.push_back(edm::RefToBase<reco::Track>( electron->gsfTrack() ));
      break;
    }
  } { // else
    // look for vetor<Muon>
    Handle<reco::MuonCollection> h_muons;
    iEvent.getByLabel(m_leptons, h_muons);
    if (h_muons.isValid()) {
      for (reco::MuonCollection::const_iterator muon = h_muons->begin(); muon != h_muons->end(); ++muon) {
        if (! muon->combinedMuon().isNull() and muon->caloCompatibility() > m_qualityCut)
          leptons.push_back(edm::RefToBase<reco::Track>( muon->combinedMuon() ));
        else 
        if (! muon->track().isNull() and muon->caloCompatibility() > m_qualityCut)
          leptons.push_back(edm::RefToBase<reco::Track>( muon->track() ));
      }
      break;
    }
  } { // else
    // look for edm::View<Track> 
    Handle<edm::View<reco::Track> > h_tracks;
    iEvent.getByLabel(m_leptons, h_tracks);
    if (h_tracks.isValid()) {
      for (unsigned int i = 0; i < h_tracks->size(); i++)
        leptons.push_back(h_tracks->refAt(i));
      break;
    }
  } { // else
    throw edm::Exception(edm::errors::NotFound) << "Object " << m_leptons << " of type among (\"reco::ElectronCollection\", \"reco::PixelMatchGsfElectronCollection\", \"reco::GsfElectronCollection\", \"reco::MuonCollection\", \"edm::View<reco::Track>\") not found";
  } } while (false);

  // output collections
  std::auto_ptr<reco::SoftLeptonTagInfoCollection> outputCollection(  new reco::SoftLeptonTagInfoCollection() );
  for (unsigned int i = 0; i < jets.size(); ++i) {
    reco::SoftLeptonTagInfo result = tag( jets[i], tracks[i], leptons, vertex );
    outputCollection->push_back( result );
  }
  iEvent.put( outputCollection );
}

// ------------ method called once each job just before starting event loop  ------------
void 
SoftLepton::beginJob(const edm::EventSetup& iSetup) {
  // grab a TransientTrack helper from the Event Setup
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get( "TransientTrackBuilder", builder );
  m_transientTrackBuilder = builder.product();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SoftLepton::endJob(void) {
}

// ---------------------------------------------------------------------------------------
reco::SoftLeptonTagInfo SoftLepton::tag (
    const edm::RefToBase<reco::Jet> & jet,
    const reco::TrackRefVector      & tracks,
    const std::vector<edm::RefToBase<reco::Track> > & leptons,
    const reco::Vertex              & primaryVertex
) const {
  reco::SoftLeptonTagInfo info;
  info.setJetRef( jet );

  for (unsigned int i = 0; i < leptons.size(); i++) {
    const edm::RefToBase<reco::Track> & lepton = leptons[i];
    const math::XYZVector & lepton_momentum = lepton->momentum();
    if ((m_chi2Cut > 0.0) and (lepton->normalizedChi2() > m_chi2Cut))
      continue;

    const GlobalVector jetAxis = refineJetAxis( jet, tracks, lepton );
    const math::XYZVector axis( jetAxis.x(), jetAxis.y(), jetAxis.z());
    if (DeltaR(lepton_momentum, axis) > m_deltaRCut)
      continue;

    reco::SoftLeptonProperties properties;

    const reco::TransientTrack transientTrack = m_transientTrackBuilder->build(*lepton);
    properties.sip2d    = IPTools::signedTransverseImpactParameter( transientTrack, jetAxis, primaryVertex ).second.significance();
    properties.sip3d    = IPTools::signedImpactParameter3D( transientTrack, jetAxis, primaryVertex ).second.significance();
    properties.deltaR   = DeltaR( lepton_momentum, axis );
    properties.ptRel    = Perp( lepton_momentum, axis );
    properties.p0Par    = boostedPPar( lepton_momentum, axis );
    properties.etaRel   = relativeEta( lepton_momentum, axis );
    properties.ratio    = lepton_momentum.R() / axis.R();
    properties.ratioRel = lepton_momentum.Dot(axis) / axis.Mag2();
    info.insert( lepton, properties );
  }

  return info;
}


// ---------------------------------------------------------------------------------------
GlobalVector SoftLepton::refineJetAxis (
    const edm::RefToBase<reco::Jet>   & jet,
    const reco::TrackRefVector        & tracks,
    const edm::RefToBase<reco::Track> & exclude /* = edm::RefToBase<reco::Track>() */
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
