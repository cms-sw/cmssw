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
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"

#include "CondFormats/RecoMuonObjects/interface/MuScleFitDBobject.h"
#include "CondFormats/DataRecord/interface/MuScleFitDBobjectRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"

class MuScleFitMuonProducer : public edm::EDProducer {
public:
  explicit MuScleFitMuonProducer(const edm::ParameterSet&);
  ~MuScleFitMuonProducer() override;

private:
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  template <class T>
  std::unique_ptr<T> applyCorrection(const edm::Handle<T>& allMuons);

  edm::InputTag theMuonLabel_;
  edm::EDGetTokenT<pat::MuonCollection> thePatMuonToken_;
  edm::EDGetTokenT<reco::MuonCollection> theRecoMuonToken_;
  bool patMuons_;
  edm::ESHandle<MuScleFitDBobject> dbObject_;
  std::string dbObjectLabel_;
  unsigned long long dbObjectCacheId_;
  std::shared_ptr<MomentumScaleCorrector> corrector_;
};

MuScleFitMuonProducer::MuScleFitMuonProducer(const edm::ParameterSet& iConfig)
    : theMuonLabel_(iConfig.getParameter<edm::InputTag>("MuonLabel")),
      thePatMuonToken_(mayConsume<pat::MuonCollection>(theMuonLabel_)),
      theRecoMuonToken_(mayConsume<reco::MuonCollection>(theMuonLabel_)),
      patMuons_(iConfig.getParameter<bool>("PatMuons")),
      dbObjectLabel_(iConfig.getUntrackedParameter<std::string>("DbObjectLabel", "")),
      dbObjectCacheId_(0) {
  if (patMuons_ == true) {
    produces<pat::MuonCollection>();
  } else {
    produces<reco::MuonCollection>();
  }
}

MuScleFitMuonProducer::~MuScleFitMuonProducer() {}

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
    if (!dbObjectLabel_.empty()) {
      iSetup.get<MuScleFitDBobjectRcd>().get(dbObjectLabel_, dbObject_);
    } else {
      iSetup.get<MuScleFitDBobjectRcd>().get(dbObject_);
    }
  }

  //std::cout << "identifiers size from dbObject = " << dbObject_->identifiers.size() << std::endl;
  //std::cout << "parameters size from dbObject = " << dbObject_->parameters.size() << std::endl;;

  // Create the corrector and set the parameters
  corrector_.reset(new MomentumScaleCorrector(dbObject_.product()));

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

// ------------ method called once each job just before starting event loop  ------------
void MuScleFitMuonProducer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void MuScleFitMuonProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(MuScleFitMuonProducer);
