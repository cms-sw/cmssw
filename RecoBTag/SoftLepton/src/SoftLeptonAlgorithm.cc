// -*- C++ -*-
//
// Package:    SoftLepton
// Class:      SoftLeptonAlgorithm
// 
/**\class SoftLepton SoftLeptonAlgorithm.cc RecoBTag/SoftLepton/src/SoftLeptonAlgorithm.cc

 Description: Concrete implementation of soft lepton b tagging algorithm

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  fwyzard
//         Created:  Wed Oct 18 18:02:07 CEST 2006
// $Id: SoftLeptonAlgorithm.cc,v 1.10 2007/03/07 23:40:55 fwyzard Exp $
//

// STL
#include <utility>
#include <iostream>
#include <iomanip>

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"
#include "RecoBTag/SoftLepton/interface/SoftLeptonAlgorithm.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerBase.h"

// ROOT TVector3 - used internally
#include "TVector3.h"
#include "TMath.h"

std::pair < reco::JetTag, reco::SoftLeptonTagInfo >
SoftLeptonAlgorithm::tag(
    const reco::JetTracksAssociationRef & jetTracks,
    const reco::Vertex                  & primaryVertex,
    const reco::TrackRefVector          & leptons
) {
  if (m_transientTrackBuilder == NULL) {
    std::cerr << "SoftLeptonAlgorithm: Missing TransientTrack builder." << std::endl;
    abort(); //FIXME: put an exception here
  }

  if (m_concreteTagger == NULL) {
    std::cerr << "SoftLeptonAlgorithm: Missing concrete soft lepton tagger." << std::endl;
    abort(); //FIXME: put an exception here
  }

  // current implementation of AssociationMap / JetTracksAssociation only stores jets WITH tracks
  /*
  if (jetTracks.product()->numberOfAssociations( jetTracks->key ) == 0) {
    // no tracks associated to this jet - assume no leptons and fail
    reco::JetTag tag(0.0, jetTracks);
    reco::SoftLeptonTagInfo info;
    return std::pair(tag, info);
  }
  */

  const reco::CaloJetRef     jet    = jetTracks->key;
  const reco::TrackRefVector tracks = jetTracks->val;
  #ifdef DEBUG
  using namespace std;
  try {
    cerr << "->   Jet " << setw(2) << jet.index() << " pT: " << setprecision(2) << setw(6) << jet->pt() << " eta: " << setprecision(2) << setw(5) << jet->eta() << " phi: " << setprecision(2) << setw(5) << jet->phi() << " has " << tracks.size() << " tracks:" << endl;
    for (reco::TrackRefVector::const_iterator track = tracks.begin(); track != tracks.end(); ++track)
      cerr << "   Track " << setw(2) << (*track).index() << " pT: " << setprecision(2) << setw(6) << (**track).pt() << " eta: " << setprecision(2) << setw(5) << (**track).eta() << " phi: " << setprecision(2) << setw(5) << (**track).phi() << endl;
  } catch (edm::Exception e) {
    cerr << "->   Jet " << setw(2) << jet.index() << " pT: " << setprecision(2) << setw(6) << jet->pt() << " eta: " << setprecision(2) << setw(5) << jet->eta() << " phi: " << setprecision(2) << setw(5) << jet->phi() << " has 0 tracks" << endl;
    reco::JetTag tag( 0.0, jetTracks );
    reco::SoftLeptonTagInfo info;
    return std::pair<reco::JetTag, reco::SoftLeptonTagInfo>(tag, info);
  }
  #endif // DEBUG

  SignedImpactParameter3D sip3D;
  const GlobalVector chargedAxis = refineJetAxis( jet, tracks );
  double discriminant = 0.0;

  reco::SoftLeptonTagInfo info;
  for (unsigned int i = 0; i < leptons.size(); i++) {
    reco::TrackRef lepton = leptons[i];

    // Temporary TVector3 vecotrs
    TVector3 _original_axis( jet->px(), jet->py(), jet->pz() );
    TVector3 _lepton( lepton->px(), lepton->py(), lepton->pz() );
    if (_lepton.DeltaR(_original_axis) > m_deltaRCut)
      continue;

    reco::SoftLeptonProperties properties;
    properties.axisRefinement = m_refineJetAxis;
    
    GlobalVector jetAxis;
    switch (m_refineJetAxis) {
      case reco::SoftLeptonProperties::AXIS_ORIGINAL :
        jetAxis = GlobalVector(jet->px(), jet->py(), jet->pz());
        break;
      case reco::SoftLeptonProperties::AXIS_CHARGED  :
        jetAxis = chargedAxis;
        break;
      case reco::SoftLeptonProperties::AXIS_EXCLUDED :
        jetAxis = refineJetAxis( jet, tracks, lepton );
        break;
      default:
        jetAxis = GlobalVector(jet->px(), jet->py(), jet->pz());
    }

    // temporary TransientTrack
    const reco::TransientTrack transientTrack = m_transientTrackBuilder->build(&lepton);
    properties.sip3d  = sip3D.apply( transientTrack, jetAxis, primaryVertex ).second.significance();
    
    // Temporary TVector3 vecotrs
    TVector3 _axis( jetAxis.x(), jetAxis.y(), jetAxis.z() );
    properties.deltaR   = _lepton.DeltaR(_axis);
    properties.ptRel    = _lepton.Perp(_axis);
    properties.etaRel   = relativeEta(_lepton, _axis);
    properties.ratio    = _lepton.Mag() / _axis.Mag();
    properties.ratioRel = _lepton.Dot(_axis) / _axis.Mag2();
    properties.tag      = m_concreteTagger->discriminant( _axis, _lepton, properties);
    info.insert( lepton, properties );

    if (properties.tag > discriminant)
      discriminant = properties.tag;
  }
 
  reco::JetTag tag( discriminant, jetTracks );
  return std::pair<reco::JetTag, reco::SoftLeptonTagInfo>( tag, info );
}

GlobalVector
SoftLeptonAlgorithm::refineJetAxis (
    const reco::CaloJetRef     & jet,
    const reco::TrackRefVector & tracks,
    const reco::TrackRef       & excluded /* = reco::TrackRef() */
) {
  math::XYZVector axis = jet->momentum();

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
  if (excluded.isNonnull()) {
    const reco::Track & track = *excluded;

    perp = track.pt();
    eta_rel = (double) track.eta() - axis.eta();
    phi_rel = (double) track.phi() - axis.phi();
    while (phi_rel < -M_PI) phi_rel += 2*M_PI;
    while (phi_rel >  M_PI) phi_rel -= 2*M_PI;

    sum_pT        -= perp;
    sum_phi_by_pT -= perp * phi_rel;
    sum_eta_by_pT -= perp * eta_rel;
  }

  if (sum_pT > 0.)
    axis = math::RhoEtaPhiVector( axis.rho(), axis.eta() + sum_eta_by_pT / sum_pT, axis.phi() + sum_phi_by_pT / sum_pT);

  return GlobalVector(axis.x(), axis.y(), axis.z());
}

double
SoftLeptonAlgorithm::relativeEta(const TVector3& a, const TVector3& b) {
  double mag = a.Mag() * b.Mag();
  double dot = a.Dot( b );
  return - 0.5 * TMath::Log((mag - dot)/(mag + dot));
}
