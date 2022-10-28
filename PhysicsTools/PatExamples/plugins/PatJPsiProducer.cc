// -*- C++ -*-
//
// Package:    PatJPsiProducer
// Class:      PatJPsiProducer
//
/**\class PatJPsiProducer PatJPsiProducer.cc Analysis/PatJPsiProducer/src/PatJPsiProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Mon Sep 28 12:53:57 CDT 2009
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <TLorentzVector.h>

#include <vector>

//
// class decleration
//

class PatJPsiProducer : public edm::global::EDProducer<> {
public:
  explicit PatJPsiProducer(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<pat::Muon>> muonSrcToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PatJPsiProducer::PatJPsiProducer(const edm::ParameterSet& iConfig)
    : muonSrcToken_(consumes<edm::View<pat::Muon>>(iConfig.getParameter<edm::InputTag>("muonSrc"))) {
  produces<std::vector<pat::CompositeCandidate>>();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void PatJPsiProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto jpsiCands = std::make_unique<std::vector<pat::CompositeCandidate>>();
  edm::Handle<edm::View<pat::Muon>> h_muons;
  iEvent.getByToken(muonSrcToken_, h_muons);

  // const double MUON_MASS = 0.106;
  const double JPSI_MASS = 3.097;

  if (h_muons.isValid() && h_muons->size() > 1) {
    for (edm::View<pat::Muon>::const_iterator muonsBegin = h_muons->begin(),
                                              muonsEnd = h_muons->end(),
                                              imuon = muonsBegin;
         imuon != muonsEnd - 1;
         ++imuon) {
      if (imuon->pt() > 1.0) {
        for (edm::View<pat::Muon>::const_iterator jmuon = imuon + 1; jmuon != muonsEnd; ++jmuon) {
          if (imuon->charge() * jmuon->charge() < 0) {
            pat::CompositeCandidate jpsi;
            jpsi.addDaughter(*imuon, "mu1");
            jpsi.addDaughter(*jmuon, "mu2");

            AddFourMomenta addp4;
            addp4.set(jpsi);

            double dR = reco::deltaR<pat::Muon, pat::Muon>(*imuon, *jmuon);

            jpsi.addUserFloat("dR", dR);

            if (fabs(jpsi.mass() - JPSI_MASS) < 1.0) {
              jpsiCands->push_back(jpsi);
            }
          }
        }
      }
    }
  }

  iEvent.put(std::move(jpsiCands));
}

//define this as a plug-in
DEFINE_FWK_MODULE(PatJPsiProducer);
