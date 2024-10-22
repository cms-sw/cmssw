#ifndef PFCand_NoPU_WithAM_h
#define PFCand_NoPU_WithAM_h

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
// $Id: PFCand_NoPU_WithAM.h,v 1.2 2012/04/18 15:09:23 mgeisler Exp $
//
//
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

//
// constants, enums and typedefs
//
typedef edm::AssociationMap<edm::OneToManyWithQuality<reco::VertexCollection, reco::PFCandidateCollection, int> >
    PFCandToVertexAssMap;
typedef edm::AssociationMap<edm::OneToManyWithQuality<reco::PFCandidateCollection, reco::VertexCollection, int> >
    VertexToPFCandAssMap;

typedef std::pair<reco::PFCandidateRef, int> PFCandQualityPair;
typedef std::vector<PFCandQualityPair> PFCandQualityPairVector;

//
// class declaration
//

class PFCand_NoPU_WithAM : public edm::global::EDProducer<> {
public:
  explicit PFCand_NoPU_WithAM(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  edm::InputTag input_AssociationType_;

  edm::EDGetTokenT<PFCandToVertexAssMap> token_PFCandToVertexAssMap_;
  edm::EDGetTokenT<VertexToPFCandAssMap> token_VertexToPFCandAssMap_;

  edm::EDGetTokenT<reco::VertexCollection> token_VertexCollection_;

  int input_MinQuality_;
};

#endif
