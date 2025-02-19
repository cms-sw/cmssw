// \class JetTracksAssociatorAtVertex JetTracksAssociatorAtVertex.cc 
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// Accommodated for Jet Package by: Fedor Ratnikov Jul. 30, 2007
// $Id: JetTracksAssociatorAtVertex.h,v 1.4 2011/11/11 18:58:02 srappocc Exp $
//
//
#ifndef JetTracksAssociatorAtVertex_h
#define JetTracksAssociatorAtVertex_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRVertex.h"
#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRVertexAssigned.h"

class JetTracksAssociatorAtVertex : public edm::EDProducer {
   public:
      JetTracksAssociatorAtVertex(const edm::ParameterSet&);
      virtual ~JetTracksAssociatorAtVertex();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     edm::InputTag mJets;
     edm::InputTag mTracks;
     int mTrackQuality;
     JetTracksAssociationDRVertex mAssociator;
     JetTracksAssociationDRVertexAssigned mAssociatorAssigned;
     bool useAssigned;   /// if true, use the track/jet association with vertex assignment to tracks
     edm::InputTag pvSrc; /// if useAssigned, will read this PV collection. 
};

#endif
