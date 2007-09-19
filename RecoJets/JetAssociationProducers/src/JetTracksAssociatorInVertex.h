// \class JetTracksAssociatorInVertex JetTracksAssociatorInVertex.cc 
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// Accommodated for Jet Package by: Fedor Ratnikov Jul. 30, 2007
// $Id: JetTracksAssociatorInVertex.h,v 1.1 2007/08/29 17:53:15 fedor Exp $
//
//
#ifndef JetTracksAssociatorInVertex_h
#define JetTracksAssociatorInVertex_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRVertex.h"

class JetTracksAssociatorInVertex : public edm::EDProducer {
   public:
      JetTracksAssociatorInVertex(const edm::ParameterSet&);
      virtual ~JetTracksAssociatorInVertex();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     edm::InputTag mJets;
     edm::InputTag mTracks;
     JetTracksAssociationDRVertex mAssociator;
};

#endif
