#include <utility>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"
#include "RecoBTag/SoftLepton/interface/SoftLeptonAlgorithm.h"

// ROOT::Math vectors (aka math::XYZVector)
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/VectorUtil.h"
using namespace ROOT::Math::VectorUtil;


SoftLeptonAlgorithm::SoftLeptonAlgorithm( const edm::ParameterSet & iConfig ) :
    m_transientTrackBuilder( NULL ),
    m_refineJetAxis( iConfig.getParameter<unsigned int>("refineJetAxis") ),
    m_deltaRCut( iConfig.getParameter<double>("leptonDeltaRCut") ),
    m_chi2Cut( iConfig.getParameter<double>("leptonChi2Cut") ),
    m_usePrimaryVertex( iConfig.getParameter<bool>("usePrimaryVertex") )
{
}
  

reco::SoftLeptonTagInfo SoftLeptonAlgorithm::tag (
    const edm::RefToBase<reco::Jet> & jet,
    const reco::TrackRefVector      & tracks,
    const reco::TrackRefVector      & leptons,
    const reco::Vertex              & primaryVertex
) const {

  if (m_usePrimaryVertex && m_transientTrackBuilder == NULL)
    throw cms::Exception("Configuration") << "SoftLeptonAlgorithm: missing TransientTrack builder";

  SignedImpactParameter3D sip3D;

  reco::SoftLeptonTagInfo info;
  info.setJetRef( jet );
  
  for (unsigned int i = 0; i < leptons.size(); i++) {
    const reco::TrackRef  & lepton = leptons[i];
    const math::XYZVector & lepton_momentum = lepton->momentum();
    if ((m_chi2Cut > 0.0) and (lepton->normalizedChi2() > m_chi2Cut))
      continue;

    const GlobalVector jetAxis = refineJetAxis( jet, tracks, lepton );
    const math::XYZVector axis( jetAxis.x(), jetAxis.y(), jetAxis.z());
    if (DeltaR(lepton_momentum, axis) > m_deltaRCut)
      continue;

    reco::SoftLeptonProperties properties;
    properties.axisRefinement = m_refineJetAxis;
   
    if (m_usePrimaryVertex) { 
      const reco::TransientTrack transientTrack = m_transientTrackBuilder->build(&lepton);
      properties.sip3d = sip3D.apply( transientTrack, jetAxis, primaryVertex ).second.significance();
    } else {
      // no primary vertex, don't compute the IP
      properties.sip3d = 0.0;
    }

    properties.deltaR   = DeltaR( lepton_momentum, axis );
    properties.ptRel    = Perp( lepton_momentum, axis );
    properties.etaRel   = relativeEta( lepton_momentum, axis );
    properties.ratio    = lepton_momentum.R() / axis.R();
    properties.ratioRel = lepton_momentum.Dot(axis) / axis.Mag2();
    properties.tag      = 0.0;  // tags should not be in extended collections
    info.insert( lepton, properties );
  }
 
  return info;
}


GlobalVector
SoftLeptonAlgorithm::refineJetAxis (
    const edm::RefToBase<reco::Jet> & jet,
    const reco::TrackRefVector      & tracks,
    const reco::TrackRef            & exclude /* = reco::TrackRef() */
) const {
  math::XYZVector axis = jet->momentum();

  if (m_refineJetAxis == reco::SoftLeptonProperties::AXIS_CHARGED_AVERAGE or 
      m_refineJetAxis == reco::SoftLeptonProperties::AXIS_CHARGED_AVERAGE_NOLEPTON) {
    
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
    if (m_refineJetAxis == reco::SoftLeptonProperties::AXIS_CHARGED_AVERAGE_NOLEPTON and exclude.isNonnull()) {
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
    
  } else if (m_refineJetAxis == reco::SoftLeptonProperties::AXIS_CHARGED_SUM or
             m_refineJetAxis == reco::SoftLeptonProperties::AXIS_CHARGED_SUM_NOLEPTON) {
    math::XYZVector sum;
    
    // recalculate the jet direction as the sum of charget tracks momenta
    for (reco::TrackRefVector::const_iterator track_it = tracks.begin(); track_it != tracks.end(); ++track_it ) {
      const reco::Track & track = **track_it;
      sum += track.momentum();
    }

    // "remove" excluded track
    if (m_refineJetAxis == reco::SoftLeptonProperties::AXIS_CHARGED_SUM_NOLEPTON and exclude.isNonnull()) {
      const reco::Track & track = *exclude;
      sum -= track.momentum();
    }

    if (sum.R() > 1.) // avoid the case of only the lepton-track with small rounding errors
      axis = sum;
  }

  return GlobalVector(axis.x(), axis.y(), axis.z());
}

double SoftLeptonAlgorithm::relativeEta(const math::XYZVector& vector, const math::XYZVector& axis) {
  double mag = vector.R() * axis.R();
  double dot = vector.Dot(axis);
  return - log((mag - dot)/(mag + dot)) / 2;
}
