// -*- C++ -*-
//
// Package:    PFCand_NoPU_WithAM
// Class:      PFCand_NoPU_WithAM
//
/**\class PF_PU_AssoMap PFCand_NoPU_WithAM.cc CommonTools/RecoUtils/plugins/PFCand_NoPU_WithAM.cc

 Description: Produces a collection of PFCandidates associated to the first vertex based on the association map

*/
//
// Original Author:  Matthias Geisler,32 4-B20,+41227676487,
//         Created:  Thu Dec  1 16:07:41 CET 2011
// $Id: PFCand_NoPU_WithAM.cc,v 1.5 2012/12/06 14:03:15 mgeisler Exp $
//
//
#include "CommonTools/RecoUtils/interface/PFCand_NoPU_WithAM.h"

// system include files
#include <memory>
#include <vector>

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/View.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

using namespace edm;
using namespace std;
using namespace reco;

//
// constructors and destructor
//
PFCand_NoPU_WithAM::PFCand_NoPU_WithAM(const edm::ParameterSet& iConfig) {
  //now do what ever other initialization is needed

  input_AssociationType_ = iConfig.getParameter<InputTag>("AssociationType");

  token_PFCandToVertexAssMap_ =
      mayConsume<PFCandToVertexAssMap>(iConfig.getParameter<InputTag>("VertexPFCandAssociationMap"));
  token_VertexToPFCandAssMap_ =
      mayConsume<VertexToPFCandAssMap>(iConfig.getParameter<InputTag>("VertexPFCandAssociationMap"));

  token_VertexCollection_ = mayConsume<VertexCollection>(iConfig.getParameter<InputTag>("VertexCollection"));

  input_MinQuality_ = iConfig.getParameter<int>("MinQuality");

  //register your products

  if (input_AssociationType_.label() == "PFCandsToVertex") {
    produces<PFCandidateCollection>("P2V");
  } else {
    if (input_AssociationType_.label() == "VertexToPFCands") {
      produces<PFCandidateCollection>("V2P");
    } else {
      if (input_AssociationType_.label() == "Both") {
        produces<PFCandidateCollection>("P2V");
        produces<PFCandidateCollection>("V2P");
      } else {
        cout << "No correct InputTag for AssociationType!" << endl;
        cout << "Won't produce any PFCandiateCollection!" << endl;
      }
    }
  }
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void PFCand_NoPU_WithAM::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  unique_ptr<PFCandidateCollection> p2v_firstvertex(new PFCandidateCollection());
  unique_ptr<PFCandidateCollection> v2p_firstvertex(new PFCandidateCollection());

  bool p2vassmap = false;
  bool v2passmap = false;

  //get the input vertex<->pf-candidate association map
  Handle<PFCandToVertexAssMap> p2vAM;
  Handle<VertexToPFCandAssMap> v2pAM;

  string asstype = input_AssociationType_.label();

  if ((asstype == "PFCandsToVertex") || (asstype == "Both")) {
    if (iEvent.getByToken(token_PFCandToVertexAssMap_, p2vAM)) {
      p2vassmap = true;
    }
  }

  if ((asstype == "VertexToPFCands") || (asstype == "Both")) {
    if (iEvent.getByToken(token_VertexToPFCandAssMap_, v2pAM)) {
      v2passmap = true;
    }
  }

  if (!p2vassmap && !v2passmap) {
    cout << "No input collection could be found" << endl;
    return;
  }

  int negativeQuality = 0;
  if (input_MinQuality_ >= 2) {
    negativeQuality = -1;
  } else {
    if (input_MinQuality_ == 1) {
      negativeQuality = -2;
    } else {
      negativeQuality = -3;
    }
  }

  if (p2vassmap) {
    const PFCandQualityPairVector pfccoll = p2vAM->begin()->val;

    //get the candidates associated to the first vertex and store them in a pf-candidate collection
    for (unsigned int pfccoll_ite = 0; pfccoll_ite < pfccoll.size(); pfccoll_ite++) {
      PFCandidateRef pfcand = pfccoll[pfccoll_ite].first;
      int quality = pfccoll[pfccoll_ite].second;

      if ((quality >= input_MinQuality_) || ((quality < 0) && (quality >= negativeQuality))) {
        p2v_firstvertex->push_back(*pfcand);
      }
    }

    iEvent.put(std::move(p2v_firstvertex), "P2V");
  }

  if (v2passmap) {
    //get the input vertex collection
    Handle<VertexCollection> input_vtxcollH;
    iEvent.getByToken(token_VertexCollection_, input_vtxcollH);

    VertexRef firstVertexRef(input_vtxcollH, 0);

    VertexToPFCandAssMap::const_iterator v2p_ite;

    for (v2p_ite = v2pAM->begin(); v2p_ite != v2pAM->end(); v2p_ite++) {
      PFCandidateRef pfcand = v2p_ite->key;

      for (unsigned v_ite = 0; v_ite < (v2p_ite->val).size(); v_ite++) {
        VertexRef vtxref = (v2p_ite->val)[v_ite].first;
        int quality = (v2p_ite->val)[v_ite].second;

        if ((vtxref == firstVertexRef) &&
            ((quality >= input_MinQuality_) || ((quality < 0) && (quality >= negativeQuality)))) {
          v2p_firstvertex->push_back(*pfcand);
        }
      }
    }

    iEvent.put(std::move(v2p_firstvertex), "V2P");
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PFCand_NoPU_WithAM::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;

  desc.add<InputTag>("AssociationType");
  desc.add<InputTag>("VertexPFCandAssociationMap");
  desc.add<InputTag>("VertexCollection");
  desc.add<int>("MinQuality");

  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFCand_NoPU_WithAM);
