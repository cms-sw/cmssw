#ifndef PFCand_AssoMap_h
#define PFCand_AssoMap_h

// -*- C++ -*-
//
// Package:    PFCand_AssoMap
// Class:      PFCand_AssoMap
// 
/**\class PFCand_AssoMap PFCand_AssoMap.cc CommonTools/RecoUtils/plugins/PFCand_AssoMap.cc

  Description: Produces a map with association between pf candidates and their particular most probable vertex with a quality of this association
*/
//
// Original Author:  Matthias Geisler
//         Created:  Wed Apr 18 14:48:37 CEST 2012
// $Id$
//
//
#include "CommonTools/RecoUtils/interface/PFCand_NoPU_WithAM.h"


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidateFwd.h"


//
// class declaration
//

class PFCand_AssoMap : public edm::EDProducer {
   public:
      explicit PFCand_AssoMap(const edm::ParameterSet&);
      ~PFCand_AssoMap();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual bool TrackMatch(reco::TrackRef,reco::TrackRef);

      // ----------member data ---------------------------

      edm::InputTag input_PFCandidates_;
      edm::InputTag input_VertexCollection_;
      edm::InputTag input_VertexTrackAssociationMap_;

      edm::InputTag ConversionsCollection_;

      edm::InputTag KshortCollection_;
      edm::InputTag LambdaCollection_;

      edm::InputTag NIVertexCollection_;
};


#endif
