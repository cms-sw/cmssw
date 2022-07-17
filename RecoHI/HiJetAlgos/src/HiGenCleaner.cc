// -*- C++ -*-
//
// Package:    HiGenCleaner
// Class:      HiGenCleaner
//
/**\class HiGenCleaner HiGenCleaner.cc yetkin/HiGenCleaner/src/HiGenCleaner.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Tue Jul 21 04:26:01 EDT 2009
//
//

// system include files
#include "RecoHI/HiJetAlgos/interface/HiGenCleaner.h"
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace std;
using namespace edm;

//
// class decleration
//

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

template <class T2>
HiGenCleaner<T2>::HiGenCleaner(const edm::ParameterSet& iConfig)
    : jetSrc_(consumes<edm::View<T2> >(iConfig.getParameter<edm::InputTag>("src"))),
      deltaR_(iConfig.getParameter<double>("deltaR")),
      ptCut_(iConfig.getParameter<double>("ptCut")),
      makeNew_(iConfig.getUntrackedParameter<bool>("createNewCollection", true)),
      fillDummy_(iConfig.getUntrackedParameter<bool>("fillDummyEntries", true)) {
  std::string alias = (iConfig.getParameter<InputTag>("src")).label();
  produces<T2Collection>().setBranchAlias(alias);
}

template <class T2>
HiGenCleaner<T2>::~HiGenCleaner() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template <class T2>
void HiGenCleaner<T2>::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  using namespace reco;

  auto jets = std::make_unique<T2Collection>();

  edm::Handle<edm::View<T2> > genjets;
  iEvent.getByToken(jetSrc_, genjets);

  int jetsize = genjets->size();

  vector<int> selection;
  selection.reserve(jetsize);
  for (int ijet = 0; ijet < jetsize; ++ijet) {
    selection.push_back(-1);
  }

  vector<int> selectedIndices;
  vector<int> removedIndices;

  for (int ijet = 0; ijet < jetsize; ++ijet) {
    const T2* jet1 = &((*genjets)[ijet]);

    if (selection[ijet] == -1) {
      selection[ijet] = 1;
      for (int ijet2 = 0; ijet2 < jetsize; ++ijet2) {
        if (ijet2 == ijet)
          continue;

        const T2* jet2 = &((*genjets)[ijet2]);

        if (Geom::deltaR(jet1->momentum(), jet2->momentum()) < deltaR_) {
          if (jet1->et() < jet2->et()) {
            selection[ijet] = 0;
            removedIndices.push_back(ijet);
            break;
          } else {
            selection[ijet2] = 0;
            removedIndices.push_back(ijet2);
          }
        }
      }
    }

    double etjet = ((*genjets)[ijet]).et();

    if (selection[ijet] == 1 && etjet > ptCut_) {
      selectedIndices.push_back(ijet);
      jets->push_back(*jet1);
    }
  }
  iEvent.put(std::move(jets));
}

template class HiGenCleaner<reco::GenParticle>;
template class HiGenCleaner<reco::GenJet>;
