// -*- C++ -*-
//
// Package:    JetPlusTracks
// Class:      JetPlusTrackProducer
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

#include "RecoJets/JetPlusTracks/plugins/JetPlusTrackProducer.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
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
JetPlusTrackProducer::JetPlusTrackProducer(const edm::ParameterSet& iConfig) {
  //register your products
  src_ = iConfig.getParameter<edm::InputTag>("src");
  srcTrackJets_ = iConfig.getParameter<edm::InputTag>("srcTrackJets");
  alias_ = iConfig.getUntrackedParameter<string>("alias");
  srcPVs_ = iConfig.getParameter<edm::InputTag>("srcPVs");
  vectorial_ = iConfig.getParameter<bool>("VectorialCorrection");
  useZSP_ = iConfig.getParameter<bool>("UseZSP");
  ptCUT_ = iConfig.getParameter<double>("ptCUT");
  dRcone_ = iConfig.getParameter<double>("dRcone");
  usePAT_ = iConfig.getParameter<bool>("UsePAT");

  mJPTalgo = new JetPlusTrackCorrector(iConfig, consumesCollector());
  if (useZSP_)
    mZSPalgo = new ZSPJPTJetCorrector(iConfig);

  produces<reco::JPTJetCollection>().setBranchAlias(alias_);
  produces<reco::CaloJetCollection>().setBranchAlias("ak4CaloJetsJPT");

  input_jets_token_ = consumes<edm::View<reco::CaloJet> >(src_);
  input_addjets_token_ = consumes<edm::View<reco::CaloJet> >(iConfig.getParameter<edm::InputTag>("srcAddCaloJets"));
  input_trackjets_token_ = consumes<edm::View<reco::TrackJet> >(srcTrackJets_);
  input_vertex_token_ = consumes<reco::VertexCollection>(srcPVs_);
  mExtrapolations_ =
      consumes<std::vector<reco::TrackExtrapolation> >(iConfig.getParameter<edm::InputTag>("extrapolations"));
}

JetPlusTrackProducer::~JetPlusTrackProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//
bool sort_by_pt(const reco::JPTJet& a, const reco::JPTJet& b) { return (a.pt() > b.pt()); }

// ------------ method called to produce the data  ------------
void JetPlusTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  auto const& jets_h = iEvent.get(input_jets_token_);
  auto const& addjets_h = iEvent.get(input_addjets_token_);
  auto const& iExtrapolations = iEvent.get(mExtrapolations_);
  edm::RefProd<reco::CaloJetCollection> pOut1RefProd = iEvent.getRefBeforePut<reco::CaloJetCollection>();
  edm::Ref<reco::CaloJetCollection>::key_type idxCaloJet = 0;

  auto pOut = std::make_unique<reco::JPTJetCollection>();
  auto pOut1 = std::make_unique<reco::CaloJetCollection>();

  double scaleJPT = 1.;
  for (auto const& jet : iEvent.get(input_trackjets_token_)) {
    int icalo = -1;
    int i = 0;
    for (auto const& oldjet : addjets_h) {
      double dr2 = deltaR2(jet, oldjet);
      if (dr2 <= dRcone_ * dRcone_) {
        icalo = i;
      }
      i++;
    }  // Calojets
    if (icalo < 0)
      continue;
    auto const& mycalo = addjets_h[icalo];
    std::vector<edm::Ptr<reco::Track> > tracksinjet = jet.tracks();
    reco::TrackRefVector tracksincalo;
    reco::TrackRefVector tracksinvert;
    for (auto const& itrack : tracksinjet) {
      for (auto const& ixtrp : iExtrapolations) {
        if (ixtrp.positions().empty())
          continue;
        if (usePAT_) {
          double mydphi = deltaPhi(ixtrp.track()->phi(), itrack->phi());
          if (fabs(ixtrp.track()->pt() - itrack->pt()) > 0.001 || fabs(ixtrp.track()->eta() - itrack->eta()) > 0.001 ||
              mydphi > 0.001)
            continue;
        } else {
          if (itrack.id() != ixtrp.track().id() || itrack.key() != ixtrp.track().key())
            continue;
        }
        tracksinvert.push_back(ixtrp.track());
        reco::TrackBase::Point const& point = ixtrp.positions().at(0);
        double dr2 = deltaR2(jet, point);
        if (dr2 <= dRcone_ * dRcone_) {
          tracksincalo.push_back(ixtrp.track());
        }
      }  // Track extrapolations
    }    // tracks

    const reco::TrackJet& corrected = jet;
    math::XYZTLorentzVector p4;
    jpt::MatchedTracks pions;
    jpt::MatchedTracks muons;
    jpt::MatchedTracks elecs;

    scaleJPT =
        mJPTalgo->correction(corrected, mycalo, iEvent, iSetup, tracksinvert, tracksincalo, p4, pions, muons, elecs);
    if (p4.pt() > ptCUT_) {
      reco::JPTJet::Specific jptspe;
      jptspe.pionsInVertexInCalo = pions.inVertexInCalo_;
      jptspe.pionsInVertexOutCalo = pions.inVertexOutOfCalo_;
      jptspe.pionsOutVertexInCalo = pions.outOfVertexInCalo_;
      jptspe.muonsInVertexInCalo = muons.inVertexInCalo_;
      jptspe.muonsInVertexOutCalo = muons.inVertexOutOfCalo_;
      jptspe.muonsOutVertexInCalo = muons.outOfVertexInCalo_;
      jptspe.elecsInVertexInCalo = elecs.inVertexInCalo_;
      jptspe.elecsInVertexOutCalo = elecs.inVertexOutOfCalo_;
      jptspe.elecsOutVertexInCalo = elecs.outOfVertexInCalo_;
      reco::CaloJetRef myjet(pOut1RefProd, idxCaloJet++);
      jptspe.theCaloJetRef = edm::RefToBase<reco::Jet>(myjet);
      jptspe.JPTSeed = 1;
      reco::JPTJet fJet(p4, jet.primaryVertex()->position(), jptspe, mycalo.getJetConstituents());
      pOut->push_back(fJet);
      pOut1->push_back(mycalo);
    }
  }  // trackjets

  int iJet = 0;
  for (auto const& oldjet : jets_h) {
    reco::CaloJet corrected = oldjet;

    // ZSP corrections
    double factorZSP = 1.;
    if (useZSP_)
      factorZSP = mZSPalgo->correction(corrected, iEvent, iSetup);
    corrected.scaleEnergy(factorZSP);

    // JPT corrections
    scaleJPT = 1.;

    math::XYZTLorentzVector p4;

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
    iJet++;

    // Output module
    if (fJet.pt() > ptCUT_)
      pOut->push_back(fJet);
  }
  std::sort(pOut->begin(), pOut->end(), sort_by_pt);
  iEvent.put(std::move(pOut1));
  iEvent.put(std::move(pOut));
}

//define this as a plug-in
//DEFINE_FWK_MODULE(JetPlusTrackProducer);
