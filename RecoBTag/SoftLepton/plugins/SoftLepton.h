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
// $Id: SoftLepton.h,v 1.2 2007/10/08 16:16:47 fwyzard Exp $
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
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

class edm::EventSetup;
class edm::Event;
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
  virtual void beginJob(const edm::EventSetup& iSetup);
  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void endJob(void);

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

  // nominal beam spot position
  static const reco::Vertex s_nominalBeamSpot;
};

#endif // RecoBTag_SoftLepton_SoftLepton_h
