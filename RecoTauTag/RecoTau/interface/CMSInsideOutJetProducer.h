#ifndef RecoTau_InsideOutProducer_H
#define RecoTau_InsideOutProducer_H
/*
 * CMS Inside-Out jet producer
 *
 * Produces jets build w/ the inside-out algorithm to seed
 * the PFRecoTauAlgorithm
 *
 * Author:  Evan Friis, UC Davis evan.friis@cern.ch
 *
 * Adapted from code in RecoJets/JetProducers::BaseJetProducer
 *
 */

#include <memory>

// Framework stuff
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Jet types
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "RecoTauTag/RecoTau/interface/CMSInsideOutAlgorithm.h"

using namespace reco;

class CMSInsideOutProducer : public edm::EDProducer
{
   public:
      CMSInsideOutProducer(const edm::ParameterSet& ps);
      ~CMSInsideOutProducer(){};

      void produce(edm::Event&, const edm::EventSetup&);

   private:
      CMSInsideOutAlgorithm alg_;
      edm::InputTag mSrc;
      bool mVerbose;
      double mEtInputCut;
      double mEInputCut;
      double mJetPtMin;
};

#endif
