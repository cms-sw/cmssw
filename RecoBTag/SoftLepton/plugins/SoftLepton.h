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
// $Id: SoftLepton.h,v 1.9 2009/10/26 15:23:43 saout Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

class TransientTrackBuilder;

class SoftLepton : public edm::EDProducer {
public:
  explicit SoftLepton(const edm::ParameterSet& iConfig);
  ~SoftLepton();

  // generic interface, using a TrackRefVector for lepton tracks
  reco::SoftLeptonTagInfo tag (
      const edm::RefToBase<reco::Jet> & jet,
      const reco::TrackRefVector      & tracks,
      const std::vector<edm::RefToBase<reco::Track> > & leptons,
      const reco::Vertex              & primaryVertex
  ) const;

  enum VertexType {
    VERTEX_NOMINAL,         // nominal beamspot
    VERTEX_BEAMSPOT,        // reconstructed beamspot
    VERTEX_PRIMARY          // reconstructed primary vertex
  };

protected:
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
  const edm::InputTag m_jets;
  const edm::InputTag m_primaryVertex;
  const edm::InputTag m_leptons;

  // service used to make transient tracks from tracks
  const TransientTrackBuilder * m_transientTrackBuilder;

  // algorithm configuration
  unsigned int  m_refineJetAxis;
  double        m_deltaRCut;
  double        m_chi2Cut;
  double        m_qualityCut;
  VertexType    m_pvType;       // vertex type
  
  // specific for reco::Muons
  muon::SelectionType m_muonSelection;

  // nominal beam spot position
  static const reco::Vertex s_nominalBeamSpot;
};

#endif // RecoBTag_SoftLepton_SoftLepton_h
