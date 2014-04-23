// \class JetTracksAssociatorExplicit JetTracksAssociatorExplicit.cc 
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// Accommodated for Jet Package by: Fedor Ratnikov Jul. 30, 2007
//
//
#ifndef JetTracksAssociatorExplicit_h
#define JetTracksAssociatorExplicit_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationExplicit.h"
#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationExplicit.h"

class JetTracksAssociatorExplicit : public edm::stream::EDProducer<> {
   public:
      JetTracksAssociatorExplicit(const edm::ParameterSet&);
      virtual ~JetTracksAssociatorExplicit();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     edm::EDGetTokenT<edm::View <reco::Jet>> mJets;
     edm::EDGetTokenT<reco::TrackCollection> mTracks;
     JetTracksAssociationExplicit mAssociatorExplicit;
};

#endif
