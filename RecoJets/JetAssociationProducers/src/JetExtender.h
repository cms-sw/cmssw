// \class JetExtender JetExtender.cc 
//
// Combines different Jet associations into single compact object
// which extends basic Jet information
// Fedor Ratnikov Sep. 10, 2007
//
//
#ifndef JetExtender_h
#define JetExtender_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

class JetExtender : public edm::stream::EDProducer<> {
   public:
      JetExtender(const edm::ParameterSet&);
      virtual ~JetExtender();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     edm::EDGetTokenT<edm::View <reco::Jet>> token_mJets;
     edm::EDGetTokenT<reco::JetTracksAssociation::Container > token_mJet2TracksAtVX;
     edm::EDGetTokenT<reco::JetTracksAssociation::Container > token_mJet2TracksAtCALO;
     edm::InputTag mJets;
     edm::InputTag mJet2TracksAtVX;
     edm::InputTag mJet2TracksAtCALO;
};

#endif
