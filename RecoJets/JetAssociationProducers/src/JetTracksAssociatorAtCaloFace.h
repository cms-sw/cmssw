// \class JetTracksAssociatorAtCaloFace JetTracksAssociatorAtCaloFace.cc 
// Associate jet with tracks extrapolated to CALO face
// Accommodated for Jet Package by: Fedor Ratnikov Sep.7, 2007
// $Id: JetTracksAssociatorAtCaloFace.h,v 1.1 2007/09/19 18:30:01 fedor Exp $
//
//
#ifndef JetTracksAssociatorAtCaloFace_h
#define JetTracksAssociatorAtCaloFace_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRCalo.h"

class JetTracksAssociatorAtCaloFace : public edm::EDProducer {
   public:
      JetTracksAssociatorAtCaloFace(const edm::ParameterSet&);
      virtual ~JetTracksAssociatorAtCaloFace();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     edm::InputTag mJets;
     edm::InputTag mTracks;
     int mTrackQuality;
     JetTracksAssociationDRCalo mAssociator;
};

#endif
