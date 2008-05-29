// \class JetTracksAssociatorAtVertex JetTracksAssociatorAtVertex.cc 
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// Accommodated for Jet Package by: Fedor Ratnikov Jul. 30, 2007
// $Id: JetTracksAssociatorAtVertex.h,v 1.1 2007/09/20 22:32:41 fedor Exp $
//
//
#ifndef JetTracksAssociatorAtVertex_h
#define JetTracksAssociatorAtVertex_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRVertex.h"

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
};

#endif
