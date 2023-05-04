// -*- C++ -*-
//
// Package:    JetPlusTrack
// Class:      JetPlusTrack
//
/**\class JetPlusTrackProducer JetPlusTrackProducer.cc JetPlusTrackProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Olga Kodolova,40 R-A12,+41227671273,
//         Created:  Fri Feb 19 10:14:02 CET 2010
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetPlusTracks/plugins/JetPlusTrackProducerAA.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

//=>
#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationXtrpCalo.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
//=>

#include <string>

using namespace std;
using namespace jpt;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
JetPlusTrackProducerAA::JetPlusTrackProducerAA(const edm::ParameterSet& iConfig) {
  //register your products
  src = iConfig.getParameter<edm::InputTag>("src");
  alias = iConfig.getUntrackedParameter<string>("alias");
  mTracks = iConfig.getParameter<edm::InputTag>("tracks");
  srcPVs_ = iConfig.getParameter<edm::InputTag>("srcPVs");
  vectorial_ = iConfig.getParameter<bool>("VectorialCorrection");
  useZSP = iConfig.getParameter<bool>("UseZSP");
  std::string tq = iConfig.getParameter<std::string>("TrackQuality");
  trackQuality_ = reco::TrackBase::qualityByName(tq);
  mConeSize = iConfig.getParameter<double>("coneSize");
  //=>
  mExtrapolations = iConfig.getParameter<edm::InputTag>("extrapolations");
  //=>
  mJPTalgo = new JetPlusTrackCorrector(iConfig, consumesCollector());
  if (useZSP)
    mZSPalgo = new ZSPJPTJetCorrector(iConfig);

  produces<reco::JPTJetCollection>().setBranchAlias(alias);

  input_jets_token_ = consumes<edm::View<reco::CaloJet> >(src);
  input_vertex_token_ = consumes<reco::VertexCollection>(srcPVs_);
  input_tracks_token_ = consumes<reco::TrackCollection>(mTracks);
  input_extrapolations_token_ = consumes<std::vector<reco::TrackExtrapolation> >(mExtrapolations);
}

JetPlusTrackProducerAA::~JetPlusTrackProducerAA() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void JetPlusTrackProducerAA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // get stuff from Event

  edm::Handle<reco::TrackCollection> tracks_h;
  iEvent.getByToken(input_tracks_token_, tracks_h);

  auto const& jets_h = iEvent.get(input_jets_token_);

  std::vector<reco::TrackRef> fTracks;
  fTracks.reserve(tracks_h->size());
  for (unsigned i = 0; i < tracks_h->size(); ++i) {
    fTracks.push_back(reco::TrackRef(tracks_h, i));
  }

  edm::Handle<std::vector<reco::TrackExtrapolation> > extrapolations_h;
  iEvent.getByToken(input_extrapolations_token_, extrapolations_h);

  auto pOut = std::make_unique<reco::JPTJetCollection>();

  reco::JPTJetCollection tmpColl;

  int iJet = 0;
  for (auto const& oldjet : jets_h) {
    reco::CaloJet corrected = oldjet;

    // ZSP corrections

    double factorZSP = 1.;
    if (useZSP)
      factorZSP = mZSPalgo->correction(corrected, iEvent, iSetup);

    corrected.scaleEnergy(factorZSP);

    // JPT corrections

    double scaleJPT = 1.;

    math::XYZTLorentzVector p4;

    // Construct JPTJet constituent
    jpt::MatchedTracks pions;
    jpt::MatchedTracks muons;
    jpt::MatchedTracks elecs;
    bool validMatches = false;

    if (!vectorial_) {
      scaleJPT = mJPTalgo->correction(corrected, oldjet, iEvent, iSetup, pions, muons, elecs, validMatches);
      p4 = math::XYZTLorentzVector(corrected.px() * scaleJPT,
                                   corrected.py() * scaleJPT,
                                   corrected.pz() * scaleJPT,
                                   corrected.energy() * scaleJPT);
    } else {
      scaleJPT = mJPTalgo->correction(corrected, oldjet, iEvent, iSetup, p4, pions, muons, elecs, validMatches);
    }

    reco::JPTJet::Specific specific;

    if (validMatches) {
      specific.pionsInVertexInCalo = pions.inVertexInCalo_;
      specific.pionsInVertexOutCalo = pions.inVertexOutOfCalo_;
      specific.pionsOutVertexInCalo = pions.outOfVertexInCalo_;
      specific.muonsInVertexInCalo = muons.inVertexInCalo_;
      specific.muonsInVertexOutCalo = muons.inVertexOutOfCalo_;
      specific.muonsOutVertexInCalo = muons.outOfVertexInCalo_;
      specific.elecsInVertexInCalo = elecs.inVertexInCalo_;
      specific.elecsInVertexOutCalo = elecs.inVertexOutOfCalo_;
      specific.elecsOutVertexInCalo = elecs.outOfVertexInCalo_;
    }

    // Fill JPT Specific
    specific.theCaloJetRef = edm::RefToBase<reco::Jet>(jets_h.refAt(iJet));
    specific.mResponseOfChargedWithEff = (float)mJPTalgo->getResponseOfChargedWithEff();
    specific.mResponseOfChargedWithoutEff = (float)mJPTalgo->getResponseOfChargedWithoutEff();
    specific.mSumPtOfChargedWithEff = (float)mJPTalgo->getSumPtWithEff();
    specific.mSumPtOfChargedWithoutEff = (float)mJPTalgo->getSumPtWithoutEff();
    specific.mSumEnergyOfChargedWithEff = (float)mJPTalgo->getSumEnergyWithEff();
    specific.mSumEnergyOfChargedWithoutEff = (float)mJPTalgo->getSumEnergyWithoutEff();
    specific.mChargedHadronEnergy = (float)mJPTalgo->getSumEnergyWithoutEff();

    // Fill Charged Jet shape parameters
    double deR2Tr = 0.;
    double deEta2Tr = 0.;
    double dePhi2Tr = 0.;
    double Zch = 0.;
    double Pout2 = 0.;
    double Pout = 0.;
    double denominator_tracks = 0.;
    int ntracks = 0;

    for (reco::TrackRefVector::const_iterator it = pions.inVertexInCalo_.begin(); it != pions.inVertexInCalo_.end();
         it++) {
      double deR = deltaR((*it)->eta(), (*it)->phi(), p4.eta(), p4.phi());
      double deEta = (*it)->eta() - p4.eta();
      double dePhi = deltaPhi((*it)->phi(), p4.phi());
      if ((**it).ptError() / (**it).pt() < 0.1) {
        deR2Tr = deR2Tr + deR * deR * (*it)->pt();
        deEta2Tr = deEta2Tr + deEta * deEta * (*it)->pt();
        dePhi2Tr = dePhi2Tr + dePhi * dePhi * (*it)->pt();
        denominator_tracks = denominator_tracks + (*it)->pt();
        Zch = Zch + (*it)->pt();

        Pout2 = Pout2 + (**it).p() * (**it).p() - (Zch * p4.P()) * (Zch * p4.P());
        ntracks++;
      }
    }
    for (reco::TrackRefVector::const_iterator it = muons.inVertexInCalo_.begin(); it != muons.inVertexInCalo_.end();
         it++) {
      double deR = deltaR((*it)->eta(), (*it)->phi(), p4.eta(), p4.phi());
      double deEta = (*it)->eta() - p4.eta();
      double dePhi = deltaPhi((*it)->phi(), p4.phi());
      if ((**it).ptError() / (**it).pt() < 0.1) {
        deR2Tr = deR2Tr + deR * deR * (*it)->pt();
        deEta2Tr = deEta2Tr + deEta * deEta * (*it)->pt();
        dePhi2Tr = dePhi2Tr + dePhi * dePhi * (*it)->pt();
        denominator_tracks = denominator_tracks + (*it)->pt();
        Zch = Zch + (*it)->pt();

        Pout2 = Pout2 + (**it).p() * (**it).p() - (Zch * p4.P()) * (Zch * p4.P());
        ntracks++;
      }
    }
    for (reco::TrackRefVector::const_iterator it = elecs.inVertexInCalo_.begin(); it != elecs.inVertexInCalo_.end();
         it++) {
      double deR = deltaR((*it)->eta(), (*it)->phi(), p4.eta(), p4.phi());
      double deEta = (*it)->eta() - p4.eta();
      double dePhi = deltaPhi((*it)->phi(), p4.phi());
      if ((**it).ptError() / (**it).pt() < 0.1) {
        deR2Tr = deR2Tr + deR * deR * (*it)->pt();
        deEta2Tr = deEta2Tr + deEta * deEta * (*it)->pt();
        dePhi2Tr = dePhi2Tr + dePhi * dePhi * (*it)->pt();
        denominator_tracks = denominator_tracks + (*it)->pt();
        Zch = Zch + (*it)->pt();

        Pout2 = Pout2 + (**it).p() * (**it).p() - (Zch * p4.P()) * (Zch * p4.P());
        ntracks++;
      }
    }
    for (reco::TrackRefVector::const_iterator it = pions.inVertexOutOfCalo_.begin();
         it != pions.inVertexOutOfCalo_.end();
         it++) {
      Zch = Zch + (*it)->pt();
    }
    for (reco::TrackRefVector::const_iterator it = muons.inVertexOutOfCalo_.begin();
         it != muons.inVertexOutOfCalo_.end();
         it++) {
      Zch = Zch + (*it)->pt();
    }
    for (reco::TrackRefVector::const_iterator it = elecs.inVertexOutOfCalo_.begin();
         it != elecs.inVertexOutOfCalo_.end();
         it++) {
      Zch = Zch + (*it)->pt();
    }

    if (mJPTalgo->getSumPtForBeta() > 0.)
      Zch = Zch / mJPTalgo->getSumPtForBeta();

    if (ntracks > 0) {
      Pout = sqrt(fabs(Pout2)) / ntracks;
    }

    if (denominator_tracks != 0) {
      deR2Tr = deR2Tr / denominator_tracks;
      deEta2Tr = deEta2Tr / denominator_tracks;
      dePhi2Tr = dePhi2Tr / denominator_tracks;
    }

    specific.R2momtr = deR2Tr;
    specific.Eta2momtr = deEta2Tr;
    specific.Phi2momtr = dePhi2Tr;
    specific.Pout = Pout;
    specific.Zch = Zch;

    // Create JPT jet
    reco::Particle::Point vertex_ = reco::Jet::Point(0, 0, 0);

    // If we add primary vertex
    edm::Handle<reco::VertexCollection> pvCollection;
    iEvent.getByToken(input_vertex_token_, pvCollection);
    if (pvCollection.isValid() && !pvCollection->empty())
      vertex_ = pvCollection->begin()->position();

    reco::JPTJet fJet(p4, vertex_, specific, corrected.getJetConstituents());

    // Temporarily collection before correction for background

    iJet++;
    tmpColl.push_back(fJet);
  }

  //=======================================================================================================>
  // Correction for background

  reco::TrackRefVector trBgOutOfCalo;
  reco::TrackRefVector trBgOutOfVertex = calculateBGtracksJet(tmpColl, fTracks, extrapolations_h, trBgOutOfCalo);

  //===> Area without Jets
  std::map<reco::JPTJetCollection::iterator, double> AreaNonJet;

  for (reco::JPTJetCollection::iterator ij1 = tmpColl.begin(); ij1 != tmpColl.end(); ij1++) {
    int nj1 = 1;
    for (reco::JPTJetCollection::iterator ij2 = tmpColl.begin(); ij2 != tmpColl.end(); ij2++) {
      if (ij2 == ij1)
        continue;
      if (fabs((*ij1).eta() - (*ij2).eta()) > 0.5)
        continue;
      nj1++;
    }

    AreaNonJet[ij1] = 4 * M_PI * mConeSize - nj1 * 4 * mConeSize * mConeSize;
  }

  //===>

  for (reco::JPTJetCollection::iterator ij = tmpColl.begin(); ij != tmpColl.end(); ij++) {
    // Correct JPTjet for background tracks

    const reco::TrackRefVector pioninin = (*ij).getPionsInVertexInCalo();
    const reco::TrackRefVector pioninout = (*ij).getPionsInVertexOutCalo();

    double ja = (AreaNonJet.find(ij))->second;

    double factorPU = mJPTalgo->correctAA(*ij, trBgOutOfVertex, mConeSize, pioninin, pioninout, ja, trBgOutOfCalo);

    (*ij).scaleEnergy(factorPU);

    // Output module
    pOut->push_back(*ij);
  }

  iEvent.put(std::move(pOut));
}
// -----------------------------------------------
// ------------ calculateBGtracksJet  ------------
// ------------ Tracks not included in jets ------
// -----------------------------------------------
reco::TrackRefVector JetPlusTrackProducerAA::calculateBGtracksJet(
    reco::JPTJetCollection& fJets,
    std::vector<reco::TrackRef>& fTracks,
    edm::Handle<std::vector<reco::TrackExtrapolation> >& extrapolations_h,
    reco::TrackRefVector& trBgOutOfCalo) {
  reco::TrackRefVector trBgOutOfVertex;

  for (unsigned t = 0; t < fTracks.size(); ++t) {
    int track_bg = 0;

    const reco::Track* track = &*(fTracks[t]);
    double trackEta = track->eta();
    double trackPhi = track->phi();

    //loop on jets
    for (unsigned j = 0; j < fJets.size(); ++j) {
      const reco::Jet* jet = &(fJets[j]);
      double jetEta = jet->eta();
      double jetPhi = jet->phi();

      if (fabs(jetEta - trackEta) < mConeSize) {
        double dphiTrackJet = deltaPhi(trackPhi, jetPhi);
        if (dphiTrackJet < mConeSize) {
          track_bg = 1;
        }
      }
    }  //jets

    if (track_bg == 0) {
      trBgOutOfVertex.push_back(fTracks[t]);
    }

  }  //tracks

  //=====> Propagate BG tracks to calo
  for (std::vector<reco::TrackExtrapolation>::const_iterator xtrpBegin = extrapolations_h->begin(),
                                                             xtrpEnd = extrapolations_h->end(),
                                                             ixtrp = xtrpBegin;
       ixtrp != xtrpEnd;
       ++ixtrp) {
    reco::TrackRefVector::iterator it = find(trBgOutOfVertex.begin(), trBgOutOfVertex.end(), (*ixtrp).track());

    if (it != trBgOutOfVertex.end()) {
      trBgOutOfCalo.push_back(*it);
    }
  }

  return trBgOutOfVertex;
}

//define this as a plug-in
//DEFINE_FWK_MODULE(JetPlusTrackProducerAA);
