// -*- C++ -*-
//
// Package:    JetSubstructurePacker
// Class:      JetSubstructurePacker
// 
// \class JetSubstructurePacker JetSubstructurePacker.h PhysicsTools/PatUtils/interface/JetSubstructurePacker.h
// Description: Class to pack subjet information from various pat::Jet collections into a single one.
//
// Original Author:  "Salvatore Rappoccio"
// $Id: JetSubstructurePacker.cc,v 1.1 2013/03/07 20:13:55 srappocc Exp $
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
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/Utilities/interface/transform.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
//
// class decleration
//


class JetSubstructurePacker : public edm::stream::EDProducer<> {
   public:

      explicit JetSubstructurePacker(const edm::ParameterSet&);
      ~JetSubstructurePacker();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      
      // ----------member data ---------------------------

      // data labels
      float                                        distMax_;      
      edm::EDGetTokenT<edm::View<pat::Jet> >       jetToken_;
      std::vector<std::string>                     algoLabels_;
      std::vector<edm::InputTag>                   algoTags_;
      std::vector< edm::EDGetTokenT< edm::View<pat::Jet> > >   algoTokens_;
      bool fixDaughters_;
      edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> pf2pc_;
      edm::EDGetTokenT<edm::Association<reco::PFCandidateCollection   >> pc2pf_;
};
