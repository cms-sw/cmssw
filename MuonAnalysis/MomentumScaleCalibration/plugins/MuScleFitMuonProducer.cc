// -*- C++ -*-
//
// Package:    MuScleFitMuonProducer
// Class:      MuScleFitMuonProducer
//
/**
 * Produce a new muon collection with corrected Pt. <br>
 * It is also possible to apply a smearing to the muons Pt.
 */
//
// Original Author:  Marco De Mattia,40 3-B32,+41227671551,
//         Created:  Tue Jun 22 13:50:22 CEST 2010
//
//

// system include files
#include <memory>
#include <string>

// user include files
#include "CondFormats/DataRecord/interface/MuScleFitDBobjectRcd.h"
#include "CondFormats/RecoMuonObjects/interface/MuScleFitDBobject.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"

class MuScleFitMuonProducer : public edm::stream::EDProducer<> {
public:
  explicit MuScleFitMuonProducer(const edm::ParameterSet&);
  ~MuScleFitMuonProducer() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  template <class T>
  std::unique_ptr<T> applyCorrection(const edm::Handle<T>& allMuons);
  const edm::ESGetToken<MuScleFitDBobject, MuScleFitDBobjectRcd> muToken_;
  const edm::InputTag theMuonLabel_;
  const edm::EDGetTokenT<pat::MuonCollection> thePatMuonToken_;
  const edm::EDGetTokenT<reco::MuonCollection> theRecoMuonToken_;
  const bool patMuons_;

  edm::ESHandle<MuScleFitDBobject> dbObject_;
  unsigned long long dbObjectCacheId_;
  std::shared_ptr<MomentumScaleCorrector> corrector_;
};

MuScleFitMuonProducer::MuScleFitMuonProducer(const edm::ParameterSet& iConfig)
    : muToken_(esConsumes(edm::ESInputTag("", iConfig.getUntrackedParameter<std::string>("DbObjectLabel", "")))),
      theMuonLabel_(iConfig.getParameter<edm::InputTag>("MuonLabel")),
      thePatMuonToken_(mayConsume<pat::MuonCollection>(theMuonLabel_)),
      theRecoMuonToken_(mayConsume<reco::MuonCollection>(theMuonLabel_)),
      patMuons_(iConfig.getParameter<bool>("PatMuons")),
      dbObjectCacheId_(0) {
  if (patMuons_ == true) {
    produces<pat::MuonCollection>();
  } else {
    produces<reco::MuonCollection>();
  }
}

MuScleFitMuonProducer::~MuScleFitMuonProducer() = default;

template <class T>
std::unique_ptr<T> MuScleFitMuonProducer::applyCorrection(const edm::Handle<T>& allMuons) {
  std::unique_ptr<T> pOut(new T);

  // Apply the correction and produce the new muons
  for (typename T::const_iterator muon = allMuons->begin(); muon != allMuons->end(); ++muon) {
    //std::cout << "Pt before correction = " << muon->pt() << std::endl;
    double pt = (*corrector_)(*muon);
    //std::cout << "Pt after correction = " << pt << std::endl;
    double eta = muon->eta();
    double phi = muon->phi();

    typename T::value_type* newMuon = muon->clone();
    newMuon->setP4(reco::Particle::PolarLorentzVector(pt, eta, phi, muon->mass()));

    pOut->push_back(*newMuon);
  }
  return pOut;
}

// ------------ method called to produce the data  ------------
void MuScleFitMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  unsigned long long dbObjectCacheId = iSetup.get<MuScleFitDBobjectRcd>().cacheIdentifier();
  if (dbObjectCacheId != dbObjectCacheId_) {
    dbObject_ = iSetup.getHandle(muToken_);
  }

  //std::cout << "identifiers size from dbObject = " << dbObject_->identifiers.size() << std::endl;
  //std::cout << "parameters size from dbObject = " << dbObject_->parameters.size() << std::endl;;

  // Create the corrector and set the parameters
  corrector_ = std::make_shared<MomentumScaleCorrector>(dbObject_.product());

  if (patMuons_ == true) {
    edm::Handle<pat::MuonCollection> allMuons;
    iEvent.getByToken(thePatMuonToken_, allMuons);
    iEvent.put(applyCorrection(allMuons));
  } else {
    edm::Handle<reco::MuonCollection> allMuons;
    iEvent.getByToken(theRecoMuonToken_, allMuons);
    iEvent.put(applyCorrection(allMuons));
  }

  // put into the Event
  // iEvent.put(std::move(pOut));
  // iEvent.put(applyCorrection(allMuons);

  /*  std::unique_ptr<reco::MuonCollection> pOut(new reco::MuonCollection);

  // Apply the correction and produce the new muons
  for( std::vector<reco::Muon>::const_iterator muon = allMuons->begin(); muon != allMuons->end(); ++muon ) {

    double pt = (*corrector_)(*muon);
    double eta = muon->eta();
    double phi = muon->phi();

    reco::Muon * newMuon = muon->clone();
    newMuon->setP4( reco::Particle::PolarLorentzVector( pt, eta, phi, muon->mass() ) );

    pOut->push_back(*newMuon);
  }
*/
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuScleFitMuonProducer);
